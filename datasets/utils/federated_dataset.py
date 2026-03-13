from abc import abstractmethod
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch.optim
from torch import nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets
from torchvision.transforms import transforms


class FederatedDataset:
    """
    Federated learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list, model_name='') -> nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


class DatasetWithIndex(Dataset):
    """
    给任意 dataset 包一层，返回 (x, y, global_index)
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        out = self.base_dataset[idx]
        if isinstance(out, tuple) and len(out) >= 2:
            x, y = out[0], out[1]
            return x, y, idx
        raise ValueError(f"Unexpected dataset output: {type(out)}")


def _get_clean_targets(train_dataset):
    """
    优先读取 clean_targets，与 noise-fl 对齐
    """
    if hasattr(train_dataset, 'clean_targets'):
        return np.array(train_dataset.clean_targets)
    elif hasattr(train_dataset, 'targets'):
        return np.array(train_dataset.targets)
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'clean_targets'):
        return np.array(train_dataset.dataset.clean_targets)
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
        return np.array(train_dataset.dataset.targets)
    else:
        raise ValueError("Cannot find clean labels from train_dataset.")


def _set_noisy_targets(train_dataset, y_train_np):
    """
    将污染后的标签写回 dataset
    """
    if hasattr(train_dataset, 'targets'):
        if isinstance(train_dataset.targets, list):
            train_dataset.targets = y_train_np.tolist()
        else:
            train_dataset.targets = y_train_np
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
        if isinstance(train_dataset.dataset.targets, list):
            train_dataset.dataset.targets = y_train_np.tolist()
        else:
            train_dataset.dataset.targets = y_train_np
    else:
        raise ValueError("Cannot write noisy targets back to train_dataset.")


def _dirichlet_equal_partition(y_train, n_participants, n_class, alpha):
    """
    类别 Dirichlet + 客户端样本数尽量一致（严格相等或只差 1）

    思路：
    - 每个客户端先分配容量 cap
    - 对每个类别采样一个 Dirichlet 概率向量
    - 在有剩余容量的客户端中，按该概率顺序分配该类别样本
    """
    N = len(y_train)
    y_train = np.array(y_train)

    # 每个客户端容量：严格相等或只差 1
    client_capacities = np.array(
        [N // n_participants + (1 if i < N % n_participants else 0) for i in range(n_participants)],
        dtype=int
    )
    client_current_sizes = np.zeros(n_participants, dtype=int)

    net_dataidx_map = {i: [] for i in range(n_participants)}

    for c in range(n_class):
        class_idx = np.where(y_train == c)[0]
        np.random.shuffle(class_idx)

        # 该类别对应的 Dirichlet 概率
        class_probs = np.random.dirichlet(alpha * np.ones(n_participants))

        for idx in class_idx:
            remaining_caps = client_capacities - client_current_sizes
            available_clients = np.where(remaining_caps > 0)[0]

            if len(available_clients) == 0:
                raise RuntimeError("No available clients left when assigning samples.")

            # 只在仍有容量的客户端中按该类别 Dirichlet 倾向分配
            avail_probs = class_probs[available_clients].astype(np.float64)

            # 防止极端情况下全 0
            if avail_probs.sum() <= 0:
                avail_probs = np.ones(len(available_clients), dtype=np.float64) / len(available_clients)
            else:
                avail_probs = avail_probs / avail_probs.sum()

            chosen_client = np.random.choice(available_clients, p=avail_probs)
            net_dataidx_map[chosen_client].append(int(idx))
            client_current_sizes[chosen_client] += 1

    # 最后再打乱每个客户端内部顺序
    for k in net_dataidx_map:
        np.random.shuffle(net_dataidx_map[k])

    return net_dataidx_map


def _iid_equal_partition(y_train, n_participants):
    """
    保留一个 IID equal split 版本，方便以后切回去
    """
    N = len(y_train)
    idxs = np.random.permutation(N)
    batch_idxs = np.array_split(idxs, n_participants)
    net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_participants)}
    return net_dataidx_map


def _inject_client_noise(y_train_np, net_dataidx_map, n_class, args):
    """
    在客户端划分完成后，对每个客户端内部注入标签噪声

    支持：
    - uniform noise: 所有客户端同一噪声率 args.noise_rate
    - heterogeneous noise: 每个客户端随机采样 [0.01, args.noise_max]
    """
    y_train_noisy = y_train_np.copy()
    N = len(y_train_np)

    # 与 noise-fl 对齐：True=clean, False=noisy
    noise_or_not = np.ones(N, dtype=bool)
    client_noise_rates = {}

    noise_mode = getattr(args, 'noise_mode', 'uniform')

    for j, idx_j in net_dataidx_map.items():
        if noise_mode == 'uniform':
            tau_i = getattr(args, 'noise_rate', getattr(args, 'noise_max', 0.0))
        elif noise_mode == 'heterogeneous':
            tau_i = np.random.uniform(0.01, args.noise_max)
        else:
            raise ValueError(f"Unsupported noise_mode: {noise_mode}")

        client_noise_rates[j] = float(tau_i)

        n_noisy = int(len(idx_j) * tau_i)
        if n_noisy <= 0:
            continue

        noisy_indices = np.random.choice(idx_j, n_noisy, replace=False)
        noise_or_not[noisy_indices] = False

        for idx in noisy_indices:
            orig_label = int(y_train_noisy[idx])

            if args.noise_type == 'symmetric':
                choices = [l for l in range(n_class) if l != orig_label]
                y_train_noisy[idx] = np.random.choice(choices)

            elif args.noise_type in ['asymmetric', 'pairflip']:
                # 通用 pairflip：label -> (label + 1) % n_class
                y_train_noisy[idx] = (orig_label + 1) % n_class

            else:
                raise ValueError(f"Unsupported noise_type: {args.noise_type}")

    return y_train_noisy, noise_or_not, client_noise_rates


def partition_label_skew_loaders(train_dataset: datasets, test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num

    # 读取 clean labels
    y_train_np = _get_clean_targets(train_dataset)

    # -----------------------------
    # 1) 先做客户端划分
    # -----------------------------
    partition_mode = getattr(setting.args, 'partition_mode', 'dirichlet')
    dir_alpha = getattr(setting.args, 'dir_alpha', 0.3)

    if partition_mode == 'iid':
        net_dataidx_map = _iid_equal_partition(y_train_np, n_participants)
    elif partition_mode == 'dirichlet':
        net_dataidx_map = _dirichlet_equal_partition(
            y_train=y_train_np,
            n_participants=n_participants,
            n_class=n_class,
            alpha=dir_alpha
        )
    else:
        raise ValueError(f"Unsupported partition_mode: {partition_mode}")

    # -----------------------------
    # 2) 再做客户端内部噪声注入
    # -----------------------------
    y_train_noisy, noise_or_not, client_noise_rates = _inject_client_noise(
        y_train_np=y_train_np,
        net_dataidx_map=net_dataidx_map,
        n_class=n_class,
        args=setting.args
    )

    # 写回污染标签
    _set_noisy_targets(train_dataset, y_train_noisy)

    # 状态挂载
    setting.client_noise_rates = client_noise_rates
    setting.noise_or_not = noise_or_not       # True=clean
    setting.is_noisy = ~noise_or_not          # True=noisy
    setting.net_dataidx_map = net_dataidx_map

    # 统计最终标签分布（注意这里统计的是污染后标签）
    net_cls_counts = record_net_data_stats(y_train_noisy, net_dataidx_map, n_class)

    # 用带 index 的 dataset 构建 DataLoader
    train_dataset_with_index = DatasetWithIndex(train_dataset)
    test_dataset_with_index = DatasetWithIndex(test_dataset)

    setting.train_loaders = []

    for j in range(n_participants):
        train_sampler = SubsetRandomSampler(net_dataidx_map[j])
        train_loader = DataLoader(
            train_dataset_with_index,
            batch_size=setting.args.local_batch_size,
            sampler=train_sampler,
            num_workers=1,
            drop_last=False
        )
        setting.train_loaders.append(train_loader)

    test_loader = DataLoader(
        test_dataset_with_index,
        batch_size=setting.args.local_batch_size,
        shuffle=False,
        num_workers=1
    )
    setting.test_loader = test_loader

    return setting.train_loaders, setting.test_loader, net_cls_counts


def record_net_data_stats(y_train, net_dataidx_map, n_class):
    net_cls_counts = {}
    y_train = np.array(y_train)

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {}
        for i in range(n_class):
            if i in unq:
                tmp[i] = int(unq_cnt[unq == i][0])
            else:
                tmp[i] = 0
        net_cls_counts[net_i] = tmp

    data_list = []
    for _, data in net_cls_counts.items():
        n_total = 0
        for _, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)

    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def save_data_stat(net_cls_counts):
    path = 'datastat.csv'
    with open(path, 'w') as f:
        for k1 in net_cls_counts:
            data = net_cls_counts[k1]
            out_str = ''
            for k2 in data:
                out_str += str(data[k2]) + ','
            out_str += '\n'
            f.write(out_str)
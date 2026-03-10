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


def partition_label_skew_loaders(train_dataset: datasets, test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num

    # 优先读取 clean_targets，与 noise-fl 对齐
    if hasattr(train_dataset, 'clean_targets'):
        y_train = train_dataset.clean_targets
    elif hasattr(train_dataset, 'targets'):
        y_train = train_dataset.targets
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'clean_targets'):
        y_train = train_dataset.dataset.clean_targets
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'targets'):
        y_train = train_dataset.dataset.targets
    else:
        raise ValueError("Cannot find clean labels from train_dataset.")

    N = len(y_train)
    y_train_np = np.array(y_train)

    # 1. IID equal split
    idxs = np.random.permutation(N)
    batch_idxs = np.array_split(idxs, n_participants)
    net_dataidx_map = {i: batch_idxs[i].tolist() for i in range(n_participants)}

    # 2. heterogeneous noise injection
    client_noise_rates = {}

    # 与 noise-fl 对齐：True=clean, False=noisy
    noise_or_not = np.ones(N, dtype=bool)

    for j in range(n_participants):
        tau_i = np.random.uniform(0.01, setting.args.noise_max)
        client_noise_rates[j] = tau_i

        idx_j = net_dataidx_map[j]
        n_noisy = int(len(idx_j) * tau_i)

        if n_noisy > 0:
            noisy_indices = np.random.choice(idx_j, n_noisy, replace=False)
            noise_or_not[noisy_indices] = False

            for idx in noisy_indices:
                orig_label = y_train_np[idx]
                if setting.args.noise_type == 'symmetric':
                    choices = [l for l in range(n_class) if l != orig_label]
                    y_train_np[idx] = np.random.choice(choices)
                elif setting.args.noise_type in ['asymmetric', 'pairflip']:
                    y_train_np[idx] = (orig_label + 1) % n_class

    # 3. 写回污染标签
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

    # 状态挂载
    setting.client_noise_rates = client_noise_rates
    setting.noise_or_not = noise_or_not       # True=clean
    setting.is_noisy = ~noise_or_not          # True=noisy
    setting.net_dataidx_map = net_dataidx_map

    # 4. 统计最终标签分布
    net_cls_counts = record_net_data_stats(y_train_np, net_dataidx_map, n_class)

    # 5. 用带 index 的 dataset 构建 DataLoader
    train_dataset_with_index = DatasetWithIndex(train_dataset)
    test_dataset_with_index = DatasetWithIndex(test_dataset)

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
                tmp[i] = unq_cnt[unq == i][0]
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
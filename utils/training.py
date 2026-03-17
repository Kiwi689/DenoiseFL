import datetime
import os
import time

import torch
from argparse import Namespace
from torch.utils.data import DataLoader

from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from utils.logger import CsvWriter


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_str(x):
    s = str(x)
    s = s.replace('/', '-')
    s = s.replace(' ', '')
    s = s.replace(':', '-')
    return s


def build_result_dir_and_files(args: Namespace, model: FederatedModel):
    root_dir = getattr(args, 'result_root', 'results')

    dataset_name = safe_str(args.dataset)
    model_name = safe_str(args.model if hasattr(args, 'model') else model.NAME)

    # 核心实验标签
    parti_mode = "dir" if getattr(args, 'partition_mode', '') == 'dirichlet' else "iid"
    noise_mode = "uni" if getattr(args, 'noise_mode', '') == 'uniform' else "het"
    dir_alpha = safe_str(getattr(args, 'dir_alpha', 'NA'))
    noise_rate = safe_str(getattr(args, 'noise_rate', getattr(args, 'noise_max', 'NA')))
    noise_type = safe_str(getattr(args, 'noise_type', 'NA'))
    strategy = safe_str(getattr(args, 'denoise_strategy', 'NA'))
    drop_rate = safe_str(getattr(args, 'drop_rate', 'NA'))

    # 目录层级里也纳入关键方法参数，避免不同实验混到一起
    short_tag = (
        f"pm-{parti_mode}_a-{dir_alpha}"
        f"_nm-{noise_mode}_nr-{noise_rate}"
        f"_nt-{noise_type}"
        f"_st-{strategy}"
        f"_dr-{drop_rate}"
    )

    result_dir = os.path.join(root_dir, dataset_name, model_name, short_tag)
    ensure_dir(result_dir)

    # 文件名继续保持简洁，但加入更细粒度时间戳和 pid，避免同秒撞车
    file_stem = f"{model_name}_{dataset_name}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    pid = os.getpid()

    txt_path = os.path.join(result_dir, f"{file_stem}_{timestamp}_pid{pid}.txt")
    log_path = os.path.join(result_dir, f"{file_stem}_{timestamp}_pid{pid}.log")

    return result_dir, txt_path, log_path


def write_log(log_path: str, msg: str, also_print: bool = True):
    if also_print:
        print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> float:
    dl = test_dl
    net = model.global_net
    status = net.training
    net.eval()

    total, top1, top5 = 0.0, 0.0, 0.0

    for _, batch in enumerate(dl):
        with torch.no_grad():
            if len(batch) == 3:
                images, labels, client_id = batch
            else:
                images, labels = batch
                client_id = None

            images, labels = images.to(model.device), labels.to(model.device)

            # 只对 FedRDN 做测试时的 client-specific normalization
            if getattr(model, 'NAME', '').lower() == 'fedrdn':
                if hasattr(model, 'normalize_test_images') and client_id is not None:
                    if torch.is_tensor(client_id):
                        # 默认一个 batch 来自同一个 client
                        cid = int(client_id[0].item())
                    else:
                        cid = int(client_id)
                    images = model.normalize_test_images(cid, images)

            outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)

            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)

    top1acc = round(100 * top1 / total, 2)
    _ = round(100 * top5 / total, 2)

    net.train(status)
    return top1acc


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    pri_train_loaders, test_loaders, net_cls_counts = private_dataset.get_data_loaders()
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders
    model.net_cls_counts = net_cls_counts

    if hasattr(private_dataset, 'client_noise_rates'):
        model.client_noise_rates = private_dataset.client_noise_rates

    if hasattr(private_dataset, 'noise_or_not'):
        model.noise_or_not = private_dataset.noise_or_not

    if hasattr(private_dataset, 'is_noisy'):
        model.is_noisy = private_dataset.is_noisy

    if hasattr(private_dataset, 'net_dataidx_map'):
        model.net_dataidx_map = private_dataset.net_dataidx_map

    result_dir, txt_path, log_path = build_result_dir_and_files(args, model)

    write_log(log_path, "=" * 100)
    write_log(log_path, f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, f"Result directory: {result_dir}")
    write_log(log_path, f"Result txt file : {txt_path}")
    write_log(log_path, f"Result log file : {log_path}")
    write_log(log_path, "-" * 100)
    write_log(log_path, f"Model: {getattr(args, 'model', model.NAME)}")
    write_log(log_path, f"Dataset: {getattr(args, 'dataset', 'NA')}")
    write_log(log_path, f"Structure: {getattr(args, 'structure', 'NA')}")
    write_log(log_path, f"Participants: {getattr(args, 'parti_num', 'NA')}")
    write_log(log_path, f"Online ratio: {getattr(args, 'online_ratio', 'NA')}")
    write_log(log_path, f"Communication epochs: {getattr(args, 'communication_epoch', 'NA')}")
    write_log(log_path, f"Local epoch: {getattr(args, 'local_epoch', 'NA')}")
    write_log(log_path, f"Local batch size: {getattr(args, 'local_batch_size', 'NA')}")
    write_log(log_path, f"Local lr: {getattr(args, 'local_lr', 'NA')}")

    # 新实验协议的关键参数
    write_log(log_path, f"Partition mode: {getattr(args, 'partition_mode', 'NA')}")
    write_log(log_path, f"Dirichlet alpha: {getattr(args, 'dir_alpha', 'NA')}")
    write_log(log_path, f"Noise mode: {getattr(args, 'noise_mode', 'NA')}")
    write_log(log_path, f"Noise rate: {getattr(args, 'noise_rate', 'NA')}")
    write_log(log_path, f"Noise type: {getattr(args, 'noise_type', 'NA')}")
    write_log(log_path, f"Noise max: {getattr(args, 'noise_max', 'NA')}")

    # 方法相关参数
    write_log(log_path, f"Denoise score alpha: {getattr(args, 'alpha', 'NA')}")
    write_log(log_path, f"Drop rate: {getattr(args, 'drop_rate', 'NA')}")
    write_log(log_path, f"Denoise strategy: {getattr(args, 'denoise_strategy', 'NA')}")
    write_log(log_path, f"FedProx mu: {getattr(args, 'mu', 'NA')}")
    write_log(log_path, f"FedRDN std: {getattr(args, 'rdn_std', 'NA')}")
    write_log(log_path, f"FedCDA history size: {getattr(args, 'cda_history_size', 'NA')}")
    write_log(log_path, f"FedGLoSS beta: {getattr(args, 'beta', 'NA')}")

    write_log(log_path, f"Averaging: {getattr(args, 'averaing', 'NA')}")
    write_log(log_path, f"Seed: {getattr(args, 'seed', 'NA')}")
    write_log(log_path, "=" * 100)

    if hasattr(model, 'ini'):
        model.ini()

    accs_list = []
    pure_list = []

    Epoch = args.communication_epoch
    option_learning_decay = args.learning_decay

    total_train_start = time.time()

    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write("epoch\tacc\tpure\tround_time_sec\ttotal_time_sec\n")

    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        round_start = time.time()

        epoch_pure = None

        if hasattr(model, 'loc_update'):
            epoch_pure = model.loc_update(pri_train_loaders)

            if option_learning_decay is True:
                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        accs = round(accs, 3)
        accs_list.append(accs)

        if epoch_pure is not None:
            epoch_pure = round(float(epoch_pure), 3)
            pure_list.append(epoch_pure)

        round_time = time.time() - round_start
        total_time = time.time() - total_train_start

        progress_msg = f"[Round {epoch_index + 1:03d}/{Epoch:03d}] Acc={accs:.3f}"
        if epoch_pure is not None:
            progress_msg += f" | Pure={epoch_pure:.3f}"
        progress_msg += (
            f" | RoundTime={round_time:.2f}s"
            f" | TotalTime={total_time:.2f}s"
            f" | Method={model.args.model}"
        )

        write_log(log_path, progress_msg)

        with open(txt_path, 'a', encoding='utf-8') as f:
            pure_str = "None" if epoch_pure is None else str(epoch_pure)
            f.write(f"{epoch_index}\t{accs}\t{pure_str}\t{round_time:.4f}\t{total_time:.4f}\n")

        if getattr(args, 'test_time', False) and epoch_index == 0:
            with open(args.dataset + '_time.csv', 'a', encoding='utf-8') as f:
                f.write(args.model + ',' + f"{round_time:.6f}" + '\n')
            return

    total_elapsed = time.time() - total_train_start

    write_log(log_path, "=" * 100)
    write_log(log_path, f"Training finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, f"Total elapsed time: {total_elapsed:.2f}s")

    if len(accs_list) > 0:
        write_log(log_path, f"Final Acc: {accs_list[-1]:.3f}")
        write_log(log_path, f"Best Acc : {max(accs_list):.3f}")

    if len(pure_list) > 0:
        write_log(log_path, f"Final Pure: {pure_list[-1]:.3f}")
        write_log(log_path, f"Best Pure : {max(pure_list):.3f}")

    write_log(log_path, "=" * 100)

    if args.csv_log:
        csv_writer.write_acc(accs_list)
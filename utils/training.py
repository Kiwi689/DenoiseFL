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


def get_method_specific_tag(args: Namespace) -> str:
    model = getattr(args, 'model', '')

    if model == 'feddenoise_v3':
        teacher_schedule = safe_str(getattr(args, 'teacher_schedule', 'NA'))
        teacher_select_strategy = safe_str(getattr(args, 'teacher_select_strategy', 'NA'))
        teacher_similarity = safe_str(getattr(args, 'teacher_similarity', 'NA'))
        warmup_mode = safe_str(getattr(args, 'warmup_mode', 'NA'))
        warmup_round = safe_str(getattr(args, 'warmup_round', 'NA'))
        stage_round = safe_str(getattr(args, 'stage_round', 'NA'))
        drop_rate = safe_str(getattr(args, 'drop_rate', 'NA'))
        drop_rate_schedule = safe_str(
            getattr(args, 'drop_rate_schedule', 'None')
            if getattr(args, 'drop_rate_schedule', '') != ''
            else 'None'
        )
        return (
            f"_wr-{warmup_round}"
            f"_sr-{stage_round}"
            f"_ts-{teacher_schedule}"
            f"_tss-{teacher_select_strategy}"
            f"_tsim-{teacher_similarity}"
            f"_wm-{warmup_mode}"
            f"_dr-{drop_rate}"
            f"_drs-{drop_rate_schedule}"
        )

    elif model == 'fedprox':
        return f"_mu-{safe_str(getattr(args, 'mu', 'NA'))}"

    elif model == 'fedrdn':
        return (
            f"_std-{safe_str(getattr(args, 'rdn_std', 'NA'))}"
            f"_eps-{safe_str(getattr(args, 'rdn_eps', 'NA'))}"
        )

    elif model == 'fedcda':
        return (
            f"_hs-{safe_str(getattr(args, 'cda_history_size', 'NA'))}"
            f"_bn-{safe_str(getattr(args, 'cda_batch_num', 'NA'))}"
            f"_wr-{safe_str(getattr(args, 'cda_warmup_round', 'NA'))}"
            f"_L-{safe_str(getattr(args, 'cda_L', 'NA'))}"
        )

    elif model == 'fedgloss':
        return (
            f"_rho-{safe_str(getattr(args, 'rho', 'NA'))}"
            f"_beta-{safe_str(getattr(args, 'beta', 'NA'))}"
            f"_slr-{safe_str(getattr(args, 'server_lr', 'NA'))}"
        )

    elif model == 'feddenoise':
        return (
            f"_dr-{safe_str(getattr(args, 'drop_rate', 'NA'))}"
            f"_ds-{safe_str(getattr(args, 'denoise_strategy', 'NA'))}"
            f"_rg-{safe_str(getattr(args, 'refresh_gap', 'NA'))}"
            f"_es-{safe_str(getattr(args, 'evaluator_schedule', 'NA'))}"
            f"_sa-{safe_str(getattr(args, 'score_agg', 'NA'))}"
        )

    return ""


def get_method_specific_log_items(args: Namespace):
    model = getattr(args, 'model', '')

    if model == 'feddenoise_v3':
        return [
            ('Warmup round', getattr(args, 'warmup_round', 'NA')),
            ('Stage round', getattr(args, 'stage_round', 'NA')),
            ('Teacher schedule', getattr(args, 'teacher_schedule', 'NA')),
            ('Teacher select strategy', getattr(args, 'teacher_select_strategy', 'NA')),
            ('Teacher similarity', getattr(args, 'teacher_similarity', 'NA')),
            ('Teacher score mode', getattr(args, 'teacher_score_mode', 'NA')),
            ('Warmup mode', getattr(args, 'warmup_mode', 'NA')),
            ('Drop rate', getattr(args, 'drop_rate', 'NA')),
            ('Drop rate schedule', getattr(args, 'drop_rate_schedule', 'None') if getattr(args, 'drop_rate_schedule', '') != '' else 'None'),
            ('Exclude self teacher', getattr(args, 'exclude_self_teacher', 'NA')),
        ]

    elif model == 'fedprox':
        return [
            ('FedProx mu', getattr(args, 'mu', 'NA')),
        ]

    elif model == 'fedrdn':
        return [
            ('FedRDN std', getattr(args, 'rdn_std', 'NA')),
            ('FedRDN eps', getattr(args, 'rdn_eps', 'NA')),
        ]

    elif model == 'fedcda':
        return [
            ('FedCDA history size', getattr(args, 'cda_history_size', 'NA')),
            ('FedCDA batch num', getattr(args, 'cda_batch_num', 'NA')),
            ('FedCDA warmup round', getattr(args, 'cda_warmup_round', 'NA')),
            ('FedCDA L', getattr(args, 'cda_L', 'NA')),
        ]

    elif model == 'fedgloss':
        return [
            ('FedGLoSS rho', getattr(args, 'rho', 'NA')),
            ('FedGLoSS beta', getattr(args, 'beta', 'NA')),
            ('FedGLoSS server lr', getattr(args, 'server_lr', 'NA')),
        ]

    elif model == 'feddenoise':
        return [
            ('Drop rate', getattr(args, 'drop_rate', 'NA')),
            ('Denoise strategy', getattr(args, 'denoise_strategy', 'NA')),
            ('Refresh gap', getattr(args, 'refresh_gap', 'NA')),
            ('Evaluator schedule', getattr(args, 'evaluator_schedule', 'NA')),
            ('Score aggregation', getattr(args, 'score_agg', 'NA')),
            ('Alpha', getattr(args, 'alpha', 'NA')),
        ]

    return []


def build_result_dir_and_files(args: Namespace, model: FederatedModel):
    root_dir = getattr(args, 'result_root', 'results')

    dataset_name = safe_str(args.dataset)
    model_name = safe_str(args.model if hasattr(args, 'model') else model.NAME)

    # 公共实验标签
    parti_mode = "dir" if getattr(args, 'partition_mode', '') == 'dirichlet' else "iid"
    noise_mode = "uni" if getattr(args, 'noise_mode', '') == 'uniform' else "het"
    dir_alpha = safe_str(getattr(args, 'dir_alpha', 'NA'))
    noise_rate = safe_str(getattr(args, 'noise_rate', getattr(args, 'noise_max', 'NA')))
    noise_type = safe_str(getattr(args, 'noise_type', 'NA'))

    short_tag = (
        f"pm-{parti_mode}_a-{dir_alpha}"
        f"_nm-{noise_mode}_nr-{noise_rate}"
        f"_nt-{noise_type}"
    )

    short_tag += get_method_specific_tag(args)

    result_dir = os.path.join(root_dir, dataset_name, model_name, short_tag)
    ensure_dir(result_dir)

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

            if getattr(model, 'NAME', '').lower() == 'fedrdn':
                if hasattr(model, 'normalize_test_images') and client_id is not None:
                    if torch.is_tensor(client_id):
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

    write_log(log_path, "=" * 120)
    write_log(log_path, f"Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, f"Result directory: {result_dir}")
    write_log(log_path, f"Result txt file : {txt_path}")
    write_log(log_path, f"Result log file : {log_path}")
    write_log(log_path, "-" * 120)

    # 公共参数
    write_log(log_path, f"Model: {getattr(args, 'model', model.NAME)}")
    write_log(log_path, f"Dataset: {getattr(args, 'dataset', 'NA')}")
    write_log(log_path, f"Structure: {getattr(args, 'structure', 'NA')}")
    write_log(log_path, f"Participants: {getattr(args, 'parti_num', 'NA')}")
    write_log(log_path, f"Online ratio: {getattr(args, 'online_ratio', 'NA')}")
    write_log(log_path, f"Communication epochs: {getattr(args, 'communication_epoch', 'NA')}")
    write_log(log_path, f"Local epoch: {getattr(args, 'local_epoch', 'NA')}")
    write_log(log_path, f"Local batch size: {getattr(args, 'local_batch_size', 'NA')}")
    write_log(log_path, f"Local lr: {getattr(args, 'local_lr', 'NA')}")
    write_log(log_path, f"Partition mode: {getattr(args, 'partition_mode', 'NA')}")
    write_log(log_path, f"Dirichlet alpha: {getattr(args, 'dir_alpha', 'NA')}")
    write_log(log_path, f"Noise mode: {getattr(args, 'noise_mode', 'NA')}")
    write_log(log_path, f"Noise rate: {getattr(args, 'noise_rate', 'NA')}")
    write_log(log_path, f"Noise type: {getattr(args, 'noise_type', 'NA')}")
    write_log(log_path, f"Noise max: {getattr(args, 'noise_max', 'NA')}")
    write_log(log_path, f"Averaging: {getattr(args, 'averaing', 'NA')}")
    write_log(log_path, f"Seed: {getattr(args, 'seed', 'NA')}")

    # 方法专属参数
    for k, v in get_method_specific_log_items(args):
        write_log(log_path, f"{k}: {v}")

    write_log(log_path, "=" * 120)

    if hasattr(model, 'ini'):
        model.ini()

    accs_list = []
    stage_pure_list = []

    Epoch = args.communication_epoch
    option_learning_decay = args.learning_decay

    total_train_start = time.time()

    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(
            "epoch\tphase\tstage\tacc\tstage_pure\tclean_precision\tclean_recall\t"
            "noisy_precision\tnoisy_recall\tkeep_ratio\tavg_teacher_similarity\t"
            "round_time_sec\ttotal_time_sec\n"
        )

    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        round_start = time.time()

        phase = 'standard'
        stage_id = -1
        epoch_pure = None
        clean_precision = None
        clean_recall = None
        noisy_precision = None
        noisy_recall = None
        keep_ratio = None
        avg_teacher_similarity = None

        round_stats = None
        if hasattr(model, 'loc_update'):
            round_stats = model.loc_update(pri_train_loaders)

            if option_learning_decay is True:
                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

        if isinstance(round_stats, dict):
            phase = round_stats.get('phase', phase)
            stage_id = round_stats.get('stage_id', stage_id)
            epoch_pure = round_stats.get('round_pure', epoch_pure)
            clean_precision = round_stats.get('clean_precision', clean_precision)
            clean_recall = round_stats.get('clean_recall', clean_recall)
            noisy_precision = round_stats.get('noisy_precision', noisy_precision)
            noisy_recall = round_stats.get('noisy_recall', noisy_recall)
            keep_ratio = round_stats.get('keep_ratio', keep_ratio)
            avg_teacher_similarity = round_stats.get('avg_teacher_similarity', avg_teacher_similarity)
        else:
            epoch_pure = round_stats

        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        accs = round(accs, 3)
        accs_list.append(accs)

        if epoch_pure is not None:
            epoch_pure = round(float(epoch_pure), 3)
            stage_pure_list.append(epoch_pure)

        def _round_or_none(x):
            if x is None:
                return None
            return round(float(x), 3)

        clean_precision = _round_or_none(clean_precision)
        clean_recall = _round_or_none(clean_recall)
        noisy_precision = _round_or_none(noisy_precision)
        noisy_recall = _round_or_none(noisy_recall)
        keep_ratio = _round_or_none(keep_ratio)
        avg_teacher_similarity = _round_or_none(avg_teacher_similarity)

        round_time = time.time() - round_start
        total_time = time.time() - total_train_start

        stage_display = 'warmup' if stage_id == -1 else str(stage_id)

        progress_msg = (
            f"[Round {epoch_index + 1:03d}/{Epoch:03d}] "
            f"Phase={phase} | Stage={stage_display} | Acc={accs:.3f}"
        )

        if epoch_pure is not None:
            progress_msg += f" | StagePure={epoch_pure:.3f}"
        if clean_precision is not None:
            progress_msg += f" | CPrec={clean_precision:.3f}"
        if clean_recall is not None:
            progress_msg += f" | CRec={clean_recall:.3f}"
        if noisy_precision is not None:
            progress_msg += f" | NPrec={noisy_precision:.3f}"
        if noisy_recall is not None:
            progress_msg += f" | NRec={noisy_recall:.3f}"
        if keep_ratio is not None:
            progress_msg += f" | Keep={keep_ratio:.3f}"
        if avg_teacher_similarity is not None:
            progress_msg += f" | TSim={avg_teacher_similarity:.3f}"

        progress_msg += (
            f" | RoundTime={round_time:.2f}s"
            f" | TotalTime={total_time:.2f}s"
            f" | Method={model.args.model}"
        )

        write_log(log_path, progress_msg)

        with open(txt_path, 'a', encoding='utf-8') as f:
            stage_pure_str = "None" if epoch_pure is None else str(epoch_pure)
            clean_precision_str = "None" if clean_precision is None else str(clean_precision)
            clean_recall_str = "None" if clean_recall is None else str(clean_recall)
            noisy_precision_str = "None" if noisy_precision is None else str(noisy_precision)
            noisy_recall_str = "None" if noisy_recall is None else str(noisy_recall)
            keep_ratio_str = "None" if keep_ratio is None else str(keep_ratio)
            avg_teacher_similarity_str = "None" if avg_teacher_similarity is None else str(avg_teacher_similarity)

            f.write(
                f"{epoch_index}\t{phase}\t{stage_display}\t{accs}\t{stage_pure_str}\t"
                f"{clean_precision_str}\t{clean_recall_str}\t"
                f"{noisy_precision_str}\t{noisy_recall_str}\t"
                f"{keep_ratio_str}\t{avg_teacher_similarity_str}\t"
                f"{round_time:.4f}\t{total_time:.4f}\n"
            )

        if getattr(args, 'test_time', False) and epoch_index == 0:
            with open(args.dataset + '_time.csv', 'a', encoding='utf-8') as f:
                f.write(args.model + ',' + f"{round_time:.6f}" + '\n')
            return

    total_elapsed = time.time() - total_train_start

    write_log(log_path, "=" * 120)
    write_log(log_path, f"Training finished at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_log(log_path, f"Total elapsed time: {total_elapsed:.2f}s")

    if len(accs_list) > 0:
        write_log(log_path, f"Final Acc: {accs_list[-1]:.3f}")
        write_log(log_path, f"Best Acc : {max(accs_list):.3f}")

    if len(stage_pure_list) > 0:
        write_log(log_path, f"Final StagePure: {stage_pure_list[-1]:.3f}")
        write_log(log_path, f"Best StagePure : {max(stage_pure_list):.3f}")

    write_log(log_path, "=" * 120)

    if args.csv_log:
        csv_writer.write_acc(accs_list)
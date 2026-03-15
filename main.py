import torch.multiprocessing
import setproctitle
import datetime
import socket
import torch
import uuid
import sys
import os

torch.multiprocessing.set_sharing_strategy('file_system')
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import Priv_NAMES as DATASET_NAMES
from utils.args import add_management_args
from utils.conf import set_random_seed
from datasets import get_prive_dataset
from utils.best_args import best_args
from argparse import ArgumentParser
from models import get_all_models
from utils.training import train
from models import get_model


def parse_args():
    parser = ArgumentParser(description='Federated Learning with Label Skew', allow_abbrev=False)

    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    parser.add_argument('--communication_epoch', type=int, default=100, help='The Communication Epoch in Federated Learning')
    parser.add_argument('--local_epoch', type=int, default=10, help='The Local Epoch for each Participant')
    parser.add_argument('--local_batch_size', type=int, default=64)
    parser.add_argument('--mu', type=float, default=0.01, help='Proximal coefficient for FedProx')
    parser.add_argument('--parti_num', type=int, default=10, help='The Number for Participants')
    parser.add_argument('--rdn_std', type=float, default=0.01, help='Gradient noise std for FedRDN')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')
    parser.add_argument(
        '--model',
        type=str,
        default='fedopt',
        help='Model name.',
        choices=get_all_models()
    )
    parser.add_argument('--structure', type=str, default='homogeneity')
    parser.add_argument(
        '--dataset',
        type=str,
        default='fl_cifar10',
        choices=DATASET_NAMES
    )
    parser.add_argument('--pri_aug', type=str, default='weak', help='Private data augmentation')

    # 兼容旧 best_args 逻辑，暂时保留
    parser.add_argument('--beta', type=float, default=0.01, help='Legacy beta argument for compatibility')

    parser.add_argument('--online_ratio', type=float, default=1, help='The ratio for online clients')

    parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd')
    parser.add_argument('--local_lr', type=float, default=0.1, help='The learning rate for local updating')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")

    parser.add_argument('--learning_decay', type=bool, default=False, help='Learning rate decay')
    parser.add_argument('--averaing', type=str, default='weight', help='Averaging strategy')

    parser.add_argument('--test_time', action='store_true')
    parser.add_argument('--t', type=float, default=0.35)

    # -----------------------------
    # partition / noise protocol 参数
    # -----------------------------
    parser.add_argument(
        '--partition_mode',
        type=str,
        default='dirichlet',
        choices=['iid', 'dirichlet'],
        help='Client data partition mode'
    )
    parser.add_argument(
        '--dir_alpha',
        type=float,
        default=0.3,
        help='Dirichlet concentration for label distribution skew'
    )
    parser.add_argument(
        '--noise_mode',
        type=str,
        default='uniform',
        choices=['uniform', 'heterogeneous'],
        help='Noise mode across clients'
    )
    parser.add_argument(
        '--noise_rate',
        type=float,
        default=0.3,
        help='Uniform noise rate used when noise_mode=uniform'
    )

    # -----------------------------
    # denoise / noise 参数
    # -----------------------------
    parser.add_argument(
        '--noise_type',
        type=str,
        default='symmetric',
        choices=['symmetric', 'asymmetric', 'pairflip'],
        help='Type of label noise'
    )
    parser.add_argument('--noise_max', type=float, default=0.30, help='Max noise rate for clients (used in heterogeneous mode)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight of local loss in denoise scoring')
    parser.add_argument('--drop_rate', type=float, default=0.15, help='Fixed ratio of samples to drop per batch')
    parser.add_argument(
        '--denoise_strategy',
        type=str,
        default='least_sim',
        choices=['most_sim', 'least_sim', 'random', 'median', 'mix'],
        help='Strategy to select scoring models'
    )

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 安全读取 best_args
    # best_args 可能没有新数据集（如 fl_svhn / fl_mnist）的配置
    # ------------------------------------------------------------------
    dataset_best_args = best_args.get(args.dataset, {})

    if args.model in dataset_best_args:
        best = dataset_best_args[args.model]
    else:
        best = {}

    # 安全读取 beta 对应配置（兼容旧结构）
    if best:
        if args.beta in best:
            best = best[args.beta]
        elif 0.5 in best:
            best = best[0.5]
        else:
            best = {}

    # 覆盖默认参数
    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    # 数据集对应学习率
    if args.dataset == 'fl_cifar10':
        args.local_lr = 0.01
    elif args.dataset == 'fl_cifar100':
        args.local_lr = 0.1
    elif args.dataset == 'fl_svhn':
        args.local_lr = 0.01
    elif args.dataset == 'fl_mnist':
        args.local_lr = 0.01
    elif args.dataset == 'fl_tinyimagenet':
        args.local_lr = 0.01
    else:
        args.local_lr = 0.01

    # 数据集对应通信轮数
    if args.dataset in ['fl_cifar10', 'fl_cifar100', 'fl_svhn', 'fl_mnist']:
        args.communication_epoch = 100
    elif args.dataset == 'fl_tinyimagenet':
        args.communication_epoch = 100
    else:
        args.communication_epoch = 50

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, model_name=args.model)

    model = get_model(backbones_list, args, priv_dataset.get_transform())
    args.arch = model.nets_list[0].name

    noise_value = args.noise_rate if args.noise_mode == 'uniform' else args.noise_max

    print('{}_{}_{}_pm-{}_alpha-{}_nm-{}_nr-{}_{}_{}_{}'.format(
        args.model,
        args.parti_num,
        args.dataset,
        args.partition_mode,
        args.dir_alpha,
        args.noise_mode,
        noise_value,
        args.online_ratio,
        args.communication_epoch,
        args.local_epoch
    ))

    if args.test_time:
        setproctitle.setproctitle('test speed')
    else:
        setproctitle.setproctitle('{}_{}_pm-{}_alpha-{}_nm-{}_nr-{}'.format(
            args.model,
            args.dataset,
            args.partition_mode,
            args.dir_alpha,
            args.noise_mode,
            noise_value
        ))

    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
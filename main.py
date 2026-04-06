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
    parser.add_argument('--rdn_eps', type=float, default=1e-6, help='Numerical stability epsilon for FedRDN')
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

    # legacy / compatibility
    parser.add_argument('--beta', type=float, default=0.01, help='Legacy beta argument for compatibility')

    parser.add_argument('--online_ratio', type=float, default=1.0, help='The ratio for online clients')

    parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd')
    parser.add_argument('--local_lr', type=float, default=0.1, help='The learning rate for local updating')
    parser.add_argument('--reg', type=float, default=1e-5, help='L2 regularization strength')

    parser.add_argument('--learning_decay', action='store_true', help='Learning rate decay')
    parser.add_argument('--averaing', type=str, default='weight', help='Averaging strategy')

    parser.add_argument('--test_time', action='store_true')
    parser.add_argument('--t', type=float, default=0.35)

    # -----------------------------
    # partition / noise protocol
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
    parser.add_argument(
        '--noise_type',
        type=str,
        default='symmetric',
        choices=['symmetric', 'asymmetric', 'pairflip'],
        help='Type of label noise'
    )
    parser.add_argument('--noise_max', type=float, default=0.30, help='Max noise rate for clients (used in heterogeneous mode)')

    # -----------------------------
    # shared denoise-style params
    # -----------------------------
    parser.add_argument('--drop_rate', type=float, default=0.15, help='Default fixed ratio of samples to drop per stage')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Legacy FedDenoise argument, kept only for compatibility'
    )
    parser.add_argument(
        '--denoise_strategy',
        type=str,
        default='least_sim',
        choices=['most_sim', 'least_sim', 'random', 'global'],
        help='Strategy to select evaluator models for old FedDenoise'
    )

    # -----------------------------
    # old FedDenoise V2 params
    # -----------------------------
    parser.add_argument(
        '--refresh_gap',
        type=int,
        default=10,
        help='Rounds between evaluator refreshes'
    )
    parser.add_argument(
        '--evaluator_schedule',
        type=str,
        default='8,6,4,2',
        help='Comma-separated evaluator counts per refresh stage, e.g. 8,6,4,2 or 8,6,5,3'
    )
    parser.add_argument(
        '--score_agg',
        type=str,
        default='weighted_mean',
        choices=['weighted_mean', 'mean'],
        help='How to aggregate per-evaluator per-sample losses'
    )

    # -----------------------------
    # FedDenoise V3 params
    # -----------------------------
    parser.add_argument(
        '--warmup_round',
        type=int,
        default=10,
        help='Number of warmup communication rounds before teacher selection'
    )
    parser.add_argument(
        '--stage_round',
        type=int,
        default=50,
        help='Number of communication rounds for each clean-subset FL stage'
    )
    parser.add_argument(
        '--teacher_schedule',
        type=str,
        default='4,3,2,1',
        help='Comma-separated teacher counts for each stage, e.g. 4,3,2,1'
    )
    parser.add_argument(
        '--teacher_select_strategy',
        type=str,
        default='least_sim',
        choices=['least_sim', 'most_sim', 'random', 'all'],
        help='Teacher selection strategy for FedDenoise V3'
    )
    parser.add_argument(
        '--teacher_similarity',
        type=str,
        default='backbone_cosine',
        choices=['backbone_cosine', 'full_model_cosine'],
        help='Similarity metric used for teacher matching'
    )
    parser.add_argument(
        '--teacher_score_mode',
        type=str,
        default='teacher_mean',
        choices=['teacher_mean'],
        help='How teacher losses are aggregated into the sample score'
    )
    parser.add_argument(
        '--warmup_mode',
        type=str,
        default='backbone_only',
        choices=['backbone_only', 'full_model', 'no_comm'],
        help='Warmup communication mode'
    )
    parser.add_argument(
        '--exclude_self_teacher',
        action='store_true',
        help='Exclude same client-id model from teacher candidates'
    )
    parser.add_argument(
        '--drop_rate_schedule',
        type=str,
        default='',
        help='Optional comma-separated drop rates for each stage, e.g. 0.3,0.2,0.1,0.1'
    )

    # -----------------------------
    # FedCDA params
    # -----------------------------
    parser.add_argument('--cda_history_size', type=int, default=3, help='FedCDA history size K')
    parser.add_argument('--cda_batch_num', type=int, default=3, help='FedCDA batch number B')
    parser.add_argument('--cda_warmup_round', type=int, default=50, help='FedCDA warmup rounds')
    parser.add_argument('--cda_L', type=float, default=1.0, help='FedCDA L coefficient')

    # -----------------------------
    # FedGLoSS params
    # -----------------------------
    parser.add_argument('--rho', type=float, default=0.05, help='FedGLoSS perturbation radius')
    parser.add_argument('--server_lr', type=float, default=1.0, help='FedGLoSS server learning rate')

    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_args()

    # -----------------------------
    # parse teacher schedule
    # -----------------------------
    args.teacher_schedule_list = [
        int(x.strip()) for x in args.teacher_schedule.split(',') if x.strip() != ''
    ]
    if len(args.teacher_schedule_list) == 0:
        raise ValueError('teacher_schedule must contain at least one integer.')

    # -----------------------------
    # parse drop rate schedule (optional)
    # -----------------------------
    if args.drop_rate_schedule.strip() == '':
        args.drop_rate_schedule_list = None
    else:
        args.drop_rate_schedule_list = [
            float(x.strip()) for x in args.drop_rate_schedule.split(',') if x.strip() != ''
        ]
        if len(args.drop_rate_schedule_list) == 0:
            raise ValueError('drop_rate_schedule must contain at least one float if provided.')

    # v3 default behavior
    if args.model == 'feddenoise_v3' and not args.exclude_self_teacher:
        args.exclude_self_teacher = True

    # -----------------------------
    # safe read best_args
    # -----------------------------
    dataset_best_args = best_args.get(args.dataset, {})

    if args.model in dataset_best_args:
        best = dataset_best_args[args.model]
    else:
        best = {}

    if best:
        if args.beta in best:
            best = best[args.beta]
        elif 0.5 in best:
            best = best[0.5]
        else:
            best = {}

    for key, value in best.items():
        setattr(args, key, value)

    if args.seed is not None:
        set_random_seed(args.seed)

    # -----------------------------
    # dataset-specific local lr
    # -----------------------------
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

    # -----------------------------
    # communication epochs
    # -----------------------------
    if args.model == 'feddenoise_v3':
        args.communication_epoch = (
            args.warmup_round + args.stage_round * len(args.teacher_schedule_list)
        )
    else:
        if args.dataset in ['fl_cifar10', 'fl_cifar100', 'fl_svhn', 'fl_mnist']:
            args.communication_epoch = 100
        elif args.dataset == 'fl_tinyimagenet':
            args.communication_epoch = 100
        else:
            args.communication_epoch = 50

    # -----------------------------
    # v3 safety checks
    # -----------------------------
    if args.model == 'feddenoise_v3':
        if args.online_ratio != 1.0:
            raise ValueError('feddenoise_v3 currently requires --online_ratio 1.0')
        if args.warmup_round <= 0:
            raise ValueError('warmup_round must be positive for feddenoise_v3')
        if args.stage_round <= 0:
            raise ValueError('stage_round must be positive for feddenoise_v3')

        if args.drop_rate_schedule_list is not None:
            if len(args.drop_rate_schedule_list) != len(args.teacher_schedule_list):
                raise ValueError(
                    'drop_rate_schedule length must match teacher_schedule length for feddenoise_v3'
                )
            for dr in args.drop_rate_schedule_list:
                if dr < 0.0 or dr >= 1.0:
                    raise ValueError('Each drop rate in drop_rate_schedule must be in [0, 1).')

        if args.drop_rate < 0.0 or args.drop_rate >= 1.0:
            raise ValueError('drop_rate must be in [0, 1).')

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

    print(
        '[Launch] model={} | dataset={} | clients={} | rounds={} | local_epoch={} | '
        'batch_size={} | noise_mode={} | noise_value={}'.format(
            args.model,
            args.dataset,
            args.parti_num,
            args.communication_epoch,
            args.local_epoch,
            args.local_batch_size,
            args.noise_mode,
            noise_value
        )
    )

    if args.test_time:
        setproctitle.setproctitle('test speed')
    else:
        setproctitle.setproctitle(
            '{}_{}_pm-{}_alpha-{}_nm-{}_nr-{}'.format(
                args.model,
                args.dataset,
                args.partition_mode,
                args.dir_alpha,
                args.noise_mode,
                noise_value
            )
        )

    train(model, priv_dataset, args)


if __name__ == '__main__':
    main()
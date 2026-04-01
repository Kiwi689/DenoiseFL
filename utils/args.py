from argparse import ArgumentParser
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    """

    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--communication_epoch', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=10)
    parser.add_argument('--local_batch_size', type=int, default=64)

    # method-related
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--rdn_std', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=get_all_models()
    )
    parser.add_argument('--structure', type=str, default='homogeneity')

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=DATASET_NAMES
    )

    parser.add_argument('--pri_aug', type=str, default='weak')
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--online_ratio', type=float, default=1.0)

    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--local_lr', type=float, default=0.01)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--learning_decay', action='store_true')

    # keep original project spelling for compatibility
    parser.add_argument(
        '--averaing',
        type=str,
        default='weight',
        choices=['weight', 'uniform']
    )

    parser.add_argument('--test_time', action='store_true')
    parser.add_argument('--t', type=float, default=0.5)

    # partition / noise
    parser.add_argument(
        '--partition_mode',
        type=str,
        default='dirichlet',
        choices=['iid', 'dirichlet']
    )
    parser.add_argument('--dir_alpha', type=float, default=0.1)

    parser.add_argument(
        '--noise_mode',
        type=str,
        default='uniform',
        choices=['uniform', 'heterogeneous']
    )
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument(
        '--noise_type',
        type=str,
        default='symmetric',
        choices=['symmetric', 'asymmetric', 'pairflip']
    )
    parser.add_argument('--noise_max', type=float, default=0.0)

    # FedDenoise
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Reserved for compatibility with older FedDenoise implementations.'
    )
    parser.add_argument(
        '--denoise_strategy',
        type=str,
        default='most_sim',
        choices=['most_sim', 'least_sim', 'random', 'global']
    )

    # FedDenoise V2
    parser.add_argument(
        '--refresh_gap',
        type=int,
        default=10,
        help='Rounds between evaluator refreshes.'
    )
    parser.add_argument(
        '--evaluator_schedule',
        type=str,
        default='8,6,4,2',
        help='Comma-separated evaluator counts per refresh stage, e.g. 8,6,4,2 or 8,6,5,3.'
    )
    parser.add_argument(
        '--score_agg',
        type=str,
        default='weighted_mean',
        choices=['weighted_mean', 'mean'],
        help='How to aggregate per-evaluator per-sample losses.'
    )

    # participant related
    parser.add_argument('--parti_num', type=int, default=10)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true', help='Enable csv logging')
from argparse import ArgumentParser
from datasets import Priv_NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')

    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')

    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')

    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    # ===== Fed/Local training related =====
    parser.add_argument('--local_epoch', type=int, default=1,
                        help='Number of local training epochs.')
    parser.add_argument('--local_batch_size', type=int, default=64,
                        help='Local batch size for each client.')
    parser.add_argument('--local_lr', type=float, default=0.01,
                        help='Local learning rate for each client.')
    parser.add_argument('--communication_epoch', type=int, default=100,
                        help='Number of communication rounds.')
    parser.add_argument('--parti_num', type=int, default=10,
                        help='Number of participants/clients.')
    parser.add_argument('--online_ratio', type=float, default=1.0,
                        help='Ratio of online clients per round.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device id.')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='Weight decay used in local optimizer.')
    parser.add_argument('--learning_decay', action='store_true',
                        help='Enable learning rate decay across communication rounds.')

    # ===== Aggregation =====
    # 注意：这里保留工程里现有拼写 averaing，避免别处访问 self.args.averaing 出错
    parser.add_argument('--averaing', type=str, default='weight',
                        choices=['weight', 'uniform'],
                        help='Aggregation type. Keep the original project spelling for compatibility.')

    # ===== Partition / noise related =====
    parser.add_argument('--partition_mode', type=str, default='dirichlet',
                        help='Partition mode, e.g. dirichlet or iid.')
    parser.add_argument('--dir_alpha', type=float, default=0.1,
                        help='Dirichlet alpha for non-IID partition.')
    parser.add_argument('--noise_mode', type=str, default='uniform',
                        help='Noise mode, e.g. uniform or heterogeneous.')
    parser.add_argument('--noise_rate', type=float, default=0.0,
                        help='Noise rate.')
    parser.add_argument('--noise_type', type=str, default='symmetric',
                        help='Noise type, e.g. symmetric or asymmetric.')
    parser.add_argument('--noise_max', type=float, default=0.0,
                        help='Maximum noise rate for heterogeneous noise setting.')

    # ===== FedDenoise related =====
    parser.add_argument('--drop_rate', type=float, default=0.2,
                        help='Fraction of high-score samples to drop in FedDenoise.')
    parser.add_argument('--denoise_strategy', type=str, default='most_sim',
                        choices=['least_sim', 'most_sim', 'random', 'global'],
                        help='Evaluator selection strategy for FedDenoise.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Reserved for compatibility with older FedDenoise implementations.')

    # ===== Other method-related reserved args =====
    parser.add_argument('--mu', type=float, default=0.0,
                        help='FedProx regularization coefficient.')
    parser.add_argument('--rdn_std', type=float, default=0.0,
                        help='FedRDN std.')
    parser.add_argument('--cda_history_size', type=int, default=0,
                        help='FedCDA history size.')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='FedGLoSS beta.')

    # ===== Misc =====
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--structure', type=str, default='default',
                        help='Model/network structure name.')
    parser.add_argument('--result_root', type=str, default='results',
                        help='Root directory for experiment results.')
    parser.add_argument('--test_time', action='store_true',
                        help='Only test the time cost of one round.')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
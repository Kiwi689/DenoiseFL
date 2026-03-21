import random
import torch
import os
import numpy as np

def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def data_path() -> str:
    path = '/home/hdf/kiwi/data'
    os.makedirs(path, exist_ok=True)
    return path



def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'

def checkpoint_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './checkpoint/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

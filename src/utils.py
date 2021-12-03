import numpy as np
import torch
import random


def set_random_seed(seed, is_gpu=False):
    """
    Set random seeds for reproducability
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if not (min_seed_value <= seed <= max_seed_value):
        raise ValueError("{} is not in bounds, numpy accepts from {} to {}".format(seed, min_seed_value, max_seed_value))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and is_gpu:
        torch.cuda.manual_seed_all(seed)


def get_device(is_gpu=True, gpu_number=0):
    """
    Set the backend for model training
    """
    gpu_count = torch.cuda.device_count()
    if gpu_count < gpu_number:
        raise ValueError("number of cuda devices: '{}'".format(gpu_count))

    else:
        if torch.cuda.is_available() and is_gpu:
            device = torch.device("cuda:{}".format(gpu_number))
        else:
            device = torch.device("cpu")

    return device
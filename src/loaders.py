import pathlib

import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

from src.datasets.eeg_epilepsy import get_eeg_epilepsy


def get_dataset_splits(name, data_dir, valid_prop, test_prop, seed):
    data_dir = pathlib.Path(data_dir)
    
    if name == "eeg_epilepsy":
        return get_eeg_epilepsy(data_dir, valid_prop, test_prop, seed)
    
    else:
        raise ValueError("dataset not implemented: '{}'".format(name))


def get_dataloaders(name, data_dir, valid_prop=0.10, test_prop=0.10, batch_size=16, 
                    num_workers=0, seed=1234, device=torch.device("cpu")):

    datasets = get_dataset_splits(name, data_dir, valid_prop, test_prop, seed=seed)

    train_dataset = datasets["train"]
    sample_weights = train_dataset.sample_weights
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    pin_memory = True if device.type == "cuda" else False

    train = DataLoader(dataset=train_dataset, 
                       batch_size=batch_size, 
                       shuffle=False,
                       drop_last=True, 
                       sampler=sampler, 
                       pin_memory=pin_memory, 
                       num_workers=num_workers) 

    valid = DataLoader(dataset=datasets["valid"], 
                       batch_size=batch_size, 
                       shuffle=False, 
                       drop_last=True, 
                       pin_memory=pin_memory, 
                       num_workers=num_workers)

    test = DataLoader(dataset=datasets["test"], 
                      batch_size=1, 
                      shuffle=False, 
                      pin_memory=pin_memory, 
                      num_workers=num_workers) 
    
    return {"train": train, "valid": valid, "test": test}
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.transforms import Compose, FlipTime, Shift, FlipPolarity, GuassianNoise
from src.datasets.utils import calculate_sample_weights


def get_eeg_epilepsy(data_dir, valid_prop=0.10, test_prop=0.10, seed=1234):

    data = pd.read_csv(data_dir / "eeg_epilepsy/dataset.csv")
    x, y = data.drop(columns=["Unnamed: 0", "y"]), data["y"]

    x, x_test, y, y_test = train_test_split(x, y, test_size=test_prop, shuffle=True, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_prop, shuffle=True, random_state=seed)

    train_sample_weights = calculate_sample_weights(y_train)
    
    reverse = FlipTime(p=0.5)
    shift = Shift(p=0.5)
    flip = FlipPolarity(p=0.5)
    noise = GuassianNoise(min_amplitude=0.01, max_amplitude=1.0, p=0.5)
    transforms = Compose([reverse, flip, shift, noise])

    datasets = {}
    for stage, x, y in zip(["train", "valid", "test"], [x_train, x_valid, x_test], [y_train, y_valid, y_test]):
        
        dataset = EEGEpilepsyDataset(x, y, transforms=transforms if stage=="train" else None)
        
        if stage == "train":
            dataset.sample_weights = train_sample_weights
        
        datasets[stage] = dataset

    return datasets


class EEGEpilepsyDataset(Dataset):
 
    def __init__(self, data, label, transforms=None):
        self.data = data.values
        self.label = label.values - 1
        self.transforms = transforms
        self.num_classes = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y  = self.data[idx], self.label[idx]

        if self.transforms:
            x = self.transforms(x, sample_rate=None)
    
        x = torch.from_numpy(x).unsqueeze(0).float()
        y = torch.tensor(y).long()
           
        return x, y
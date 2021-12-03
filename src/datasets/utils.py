import numpy as np
import torch


def calculate_sample_weights(y):

    classes, counts = np.unique(y, return_counts=True)
    class_weights = dict(zip(classes, sum(counts) / counts))
    sample_weights = torch.DoubleTensor([class_weights[cls] for cls in y])

    return sample_weights
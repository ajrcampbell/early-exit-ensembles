import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(name, ensemble, **kwargs):

    if name == "cross_entropy":
        
        if ensemble == "early_exit":
            return ExitWeightedCrossEntropyLoss(**kwargs)

        else:
            return nn.CrossEntropyLoss(**kwargs)

    else:
        raise ValueError("loss not implemented: '{}'".format(name))
      
  
class ExitWeightedCrossEntropyLoss:

    def __init__(self, alpha):
        self.alpha=torch.tensor(alpha)
     
    def __call__(self, logits, labels, gamma):

        batch_size, num_exits, _ = logits.shape
  
        loss = 0.0
        for ex in range(num_exits):
            exit_logits = logits[:, ex, :] 
            loss += self.alpha[ex] * gamma[ex] * F.cross_entropy(exit_logits, labels)

        return loss
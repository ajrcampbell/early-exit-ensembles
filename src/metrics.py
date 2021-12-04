import torch
import torch.nn.functional as F
import torchmetrics.functional as tm
from torch.distributions import Categorical


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


def F1(logits, labels, ensemble_weights, average="weighted"):


    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    pred_labels = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale).argmax(-1)

    f1 = tm.f1(pred_labels, labels, num_classes=num_classes, average=average)

    return f1


def precision(logits, labels, ensemble_weights, average="weighted"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    pred_labels = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale).argmax(-1)

    pr = tm.precision(pred_labels, labels, num_classes=num_classes, average=average)

    return pr


def recall(logits, labels, ensemble_weights, average="weighted"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum() 

    pred_labels = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale).argmax(-1)

    rc = tm.recall(pred_labels, labels, num_classes=num_classes, average=average)

    return rc


def negative_loglikelihood(logits, labels, ensemble_weights, reduction="mean"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    nll = -Categorical(probs=probs).log_prob(labels)

    if reduction == "mean":
        nll = nll.mean()

    return nll


def brier_score(logits, labels, ensemble_weights, reduction="mean"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    labels_one_hot = F.one_hot(labels, num_classes=num_classes)

    bs = ((probs - labels_one_hot)**2).sum(dim=-1)

    if reduction == "mean":
        bs = bs.mean()

    return bs


def predictive_entropy(logits, labels, ensemble_weights, reduction="mean"):

    _, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum()

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    et = Categorical(probs=probs).entropy()

    if reduction == "mean":
        et = et.mean()

    return et
    

def predictive_confidence(logits, labels, ensemble_weights, reduction="mean"):

    num_samples, _, _ = logits.shape
    scale = ensemble_weights.sum() 

    probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)

    pc = probs[torch.arange(num_samples), labels]

    if reduction == "mean":
        pc = pc.mean()

    return pc


def expected_calibration_error(logits, labels, ensemble_weights, n_bins=15):

    num_samples, num_exits, num_classes = logits.shape
    scale = ensemble_weights.sum() 

    pred_probs = logits.softmax(dim=-1).mul(ensemble_weights).sum(dim=-2).div(scale)
    pred_labels = pred_probs.argmax(-1)

    pred_probs = pred_probs[torch.arange(num_samples), pred_labels]

    correct = pred_labels.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    conf_bin = torch.zeros_like(bin_boundaries)
    acc_bin = torch.zeros_like(bin_boundaries)
    prop_bin = torch.zeros_like(bin_boundaries)
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):

        in_bin = pred_probs.gt(bin_lower.item()) * pred_probs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            # probability of making a correct prediction given a probability bin
            acc_bin[i] = correct[in_bin].float().mean()
            # average predicted probabily given a probability bin.
            conf_bin[i] = pred_probs[in_bin].mean()
            # probability of observing a probability bin
            prop_bin[i] = prop_in_bin
    
    ece = ((acc_bin - conf_bin).abs() * prop_bin).sum()

    return ece


def calculate_metrics(model, logits, labels, ensemble_weights):

    metrics = dict(f1=F1(logits, labels, ensemble_weights, average="weighted").numpy(),
                   precision=precision(logits, labels, ensemble_weights, average="weighted").numpy(),
                   recall=recall(logits, labels, ensemble_weights, average="weighted").numpy(),
                   negative_loglikelihood=negative_loglikelihood(logits, labels, ensemble_weights, reduction="mean").numpy(),
                   brier_score=brier_score(logits, labels, ensemble_weights, reduction="mean").numpy(),
                   predictive_entropy=predictive_entropy(logits, labels, ensemble_weights, reduction="mean").numpy(),
                   expected_calibration_error=expected_calibration_error(logits, labels, ensemble_weights, n_bins=15).numpy(),
                   params=count_parameters(model))

    return metrics
import torch.optim.lr_scheduler as scl


def get_scheduler(name, optimiser, **kwargs):

    if name == "reduce_on_plateau":
        return scl.ReduceLROnPlateau(optimiser, **kwargs)

    elif name == "multi_step":
        return scl.MultiStepLR(optimiser, **kwargs)
    
    elif name is None:
        return None
    
    else:
        raise NotImplementedError("scheduler not implemented: '{}'".format(name))
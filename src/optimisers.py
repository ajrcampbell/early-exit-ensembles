import torch.optim as opt


def get_optimiser(name, model, **kwargs):

    if name == "adam":

        if "early_exit" in model.name:

            weight_decay = kwargs.pop("weight_decay")
            params = [{"params": [param for name, param in model.named_parameters() if "exit_block" not in name], "weight_decay": weight_decay}]
            
            for block_idx, exit_block in enumerate(model.exit_blocks):
                params += [{"params": exit_block.parameters(), "weight_decay": (block_idx + 1) * weight_decay}]

            return opt.Adam(params, **kwargs)

        else:

            return opt.Adam(model.parameters(), **kwargs)

    else:
        raise NotImplementedError("optimiser not implemented: '{}'".format(name))
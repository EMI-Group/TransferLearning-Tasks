import torch
from torch import nn


def group_weight(weight_group, module, norm_layer, lr, no_decay_lr=None):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    lr = lr if no_decay_lr is None else no_decay_lr
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group
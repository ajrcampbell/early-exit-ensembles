import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_res_net_18(ensemble, **kwargs):

    if ensemble is None:
        return ResNet18(**kwargs)

    elif ensemble == "early_exit":
        return ResNet18EarlyExit(**kwargs)

    elif ensemble == "mc_dropout":
        return ResNet18MCDrop(**kwargs)

    elif ensemble == "deep":
        return ResNet18(**kwargs)

    elif ensemble == "depth":
        return ResNet18Depth(**kwargs)

    else:
        NotImplementedError("ensemble not implemented: '{}'".format(ensemble))


def init_weights(model):

    for module in model.modules():

        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    name = "res_net_18"

    def __init__(self, out_channels, seed=None):
        super().__init__()

        self.out_channels = out_channels
        self.seed = seed

        self.hidden_sizes = [64, 128, 256, 512]
        self.layers = [2, 2, 2, 2]
        self.strides = [1, 2, 2, 2]
        self.inplanes = self.hidden_sizes[0]

        in_block = [nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)]
        in_block += [nn.BatchNorm1d(self.inplanes)]
        in_block += [nn.ReLU(inplace=True)]
        in_block += [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)]
        self.in_block = nn.Sequential(*in_block)

        blocks = []
        for h, l, s in zip(self.hidden_sizes, self.layers, self.strides):
            blocks += [self._make_layer(h, l, s)]
        self.blocks = nn.Sequential(*blocks)

        out_block = [nn.AdaptiveAvgPool1d(1)]
        out_block += [nn.Flatten(1)]
        out_block += [nn.Linear(self.hidden_sizes[-1], self.out_channels)]
        self.out_block = nn.Sequential(*out_block)

        if self.seed is not None:
            torch.manual_seed(seed)

        self.apply(init_weights)

    def _make_layer(self, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(_conv1x1(self.inplanes, planes, stride), nn.BatchNorm1d(planes))

        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes

        for _ in range(1, blocks):
            layers += [BasicBlock(self.inplanes, planes)]
        
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)

        return x


class ExitBlock(nn.Module):

    def __init__(self, in_channels, hidden_sizes, out_channels):
        super().__init__()

        layers = [nn.AdaptiveAvgPool1d(1)]
        layers += [nn.Flatten(1)]
        layers += [nn.Linear(in_channels, hidden_sizes)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_sizes, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


class ResNet18EarlyExit(ResNet18):
    
    name = "res_net_18_early_exit"

    def __init__(self, *args, exit_after=-1, complexity_factor=1.2, **kwargs):
        self.exit_after = exit_after
        self.complexity_factor = complexity_factor

        super().__init__(*args, **kwargs)

        to_exit = [2, 8, 15, 24, 31, 40, 47, 56]
        hidden_sizes = len(self.hidden_sizes)

        num_hidden = len(self.hidden_sizes)
        exit_hidden_sizes = [int(((self.complexity_factor ** 0.5) ** (num_hidden - idx)) * self.hidden_sizes[-1]) for idx in range(num_hidden)]
        exit_hidden_sizes = [h for pair in zip(exit_hidden_sizes, exit_hidden_sizes) for h in pair]

        if self.exit_after == -1:
            self.exit_after = range(len(to_exit))

        num_exits = len(to_exit)

        if (len(self.exit_after) > num_exits) or not set(self.exit_after).issubset(list(range(num_exits))):
            raise ValueError("valid exit points: {}".format(", ".join(str(n) for n in range(num_exits))))

        self.exit_hidden_sizes = np.array(exit_hidden_sizes)[self.exit_after]

        blocks = []
        for idx, module in enumerate(self.blocks.modules()):
            if idx in to_exit:
                blocks += [module]
        self.blocks = nn.ModuleList(blocks)

        idx = 0
        exit_blocks = []
        for block_idx, block in enumerate(self.blocks):
            if block_idx in self.exit_after:
                in_channels = block.conv1.out_channels
                exit_blocks += [ExitBlock(in_channels, self.exit_hidden_sizes[idx], self.out_channels)]
                idx += 1
        self.exit_blocks = nn.ModuleList(exit_blocks)

        self.apply(init_weights)

    def forward(self, x):

        out = self.in_block(x)

        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks += [out]

        out_exits = []
        for exit_after, exit_block in zip(self.exit_after, self.exit_blocks):
            out = exit_block(out_blocks[exit_after])
            out_exits += [out]

        out = self.out_block(out_blocks[-1])
        out = torch.stack(out_exits + [out], dim=1)

        return out


class MCDropout(nn.Dropout):

    def forward(self, x):
        return F.dropout(x, self.p, True, self.inplace)


class ResNet18MCDrop(ResNet18EarlyExit):
    
    name = "res_net_18_mc_drop"

    def __init__(self, *args, drop_after=-1, drop_prob=0.2, **kwargs):
        self.drop_after = drop_after
        self.drop_prob = drop_prob

        super().__init__(*args, exit_after=drop_after, **kwargs)

        self.drop_after = self.exit_after

        self.__delattr__("exit_after")
        self.__delattr__("exit_blocks")

        for block_idx in self.drop_after:            
            self.blocks[block_idx].add_module("dropout", MCDropout(self.drop_prob))

    def forward(self, x):

        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)

        return x


class ResNet18Depth(ResNet18):

    name = "res_net_18_depth"

    def __init__(self, *args, max_depth=1, **kwargs):
        self.max_depth = max_depth
        
        super().__init__(*args, **kwargs)

        num_blocks = len(self.hidden_sizes)

        if self.max_depth == -1:
            self.max_depth = len(self.hidden_sizes)

        elif (max_depth > num_blocks) or (max_depth < 1):
            raise ValueError("valid depths: {}".format(", ".join(str(n) for n in range(1, num_blocks + 1))))

        self.blocks = self.blocks[:self.max_depth]

        out_block = [nn.AdaptiveAvgPool1d(1)]
        out_block += [nn.Flatten(1)]
        out_block += [nn.Linear(self.hidden_sizes[self.max_depth - 1], self.out_channels)]
        self.out_block = nn.Sequential(*out_block)
from __future__ import print_function, division, absolute_import
import torch.nn as nn

__all__ = ['Xception', 'xception39']
from modules.bn import InPlaceABNSync as BatchNorm2d


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation, groups=groups, bias=has_bias)
        self.bn = BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConvBnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_cbr(x)
        return x


class Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_out_channels, has_proj, stride, dilation=1):
        super(Block, self).__init__()
        self.has_proj = has_proj

        if has_proj:
            self.proj = SeparableConvBnRelu(in_channels, mid_out_channels * self.expansion, 3, stride, 1)
        self.residual_branch = nn.Sequential(
            SeparableConvBnRelu(in_channels, mid_out_channels, 3, stride, dilation, dilation),
            SeparableConvBnRelu(mid_out_channels, mid_out_channels, 3, 1, 1),
            SeparableConvBnRelu(mid_out_channels, mid_out_channels * self.expansion, 3, 1, 1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        if self.has_proj:
            shortcut = self.proj(x)
        residual = self.residual_branch(x)
        output = self.relu(shortcut + residual)
        return output


class Xception(nn.Module):
    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()
        self.in_channels = 8
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1, has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, layers[0], channels[0], stride=2)
        self.layer2 = self._make_layer(block, norm_layer, layers[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, norm_layer, layers[2], channels[2], stride=2)
        self.channels = [self.in_channels, channels[0] * block.expansion, channels[1] * block.expansion, channels[2] * block.expansion]

    def _make_layer(self, block, norm_layer, blocks, mid_out_channels, stride=1):
        layers = []
        has_proj = True if stride > 1 else False
        layers.append(block(self.in_channels, mid_out_channels, has_proj, stride=stride))
        self.in_channels = mid_out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, mid_out_channels, has_proj=False, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        x = self.conv1(x)
        x = self.maxpool(x)
        feats.append(x)         # 4
        x = self.layer1(x)      # 8
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)         # 16
        x = self.layer3(x)
        feats.append(x)         # 32

        return feats


def xception39(pretrained_model=None, **kwargs):
    model = Xception(Block, [4, 8, 4], [16, 32, 64], **kwargs)
    return model


if __name__ == "__main__":
    xception39()

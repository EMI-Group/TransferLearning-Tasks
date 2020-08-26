import os
from .DARTS.networks import NetworkImageNet
from .DARTS import genotypes
from .NASNet.network import NASNetAMobile
from .Mnasnet import MNASNet
from .MobileNetV2 import MobileNetV2
from .ShuffleNetV2 import ShuffleNetV2
from .ResNet import ResNet, BasicBlock, Bottleneck
import torch 


def pairnas(weight_path=None):
    genotype = eval("genotypes.%s" % 'PairNAS_CIFAR10')
    base_net = NetworkImageNet(46, 1000, 14, False, genotype)
    base_net.load_state_dict(torch.load('pretrained_models/PairNAS_CIFAR10_ImageNet_weight.pth'))
    return base_net


def darts(weight_path=None):
    genotype = eval("genotypes.%s" % 'DARTS')
    base_net = NetworkImageNet(48, 1000, 14, False, genotype)
    return base_net


def nasnet(weight_path=None):
    base_net = NASNetAMobile()
    return base_net


def mnasnet(weight_path=None, **kwargs):    # 1.0
    model = MNASNet(1.0, **kwargs)
    return model


def mobilenetv2(weight_path=None):
    model = MobileNetV2()
    return model


def shufflenetv2(weight_path=None):
    model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024])   # 1.0
    return model


def resnet18(weight_path=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(weight_path=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(weight_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
    
    
def resnet101(weight_path=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict

from .backbones import pairnas, darts, nasnet, mnasnet, mobilenetv2, shufflenetv2, resnet18

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class PoseNet(nn.Module):
    def __init__(self, cfg, is_train=True):
        super(PoseNet, self).__init__()
        model_name = cfg.MODEL.NAME.split('_')[1]
        self.backbone = eval('{}'.format(model_name))(cfg.MODEL.PRETRAINED)
        if is_train:
            self.backbone.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
        self.backbone.fc = self.backbone.classifier = self.backbone.last_linear = None
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(extra.NUM_DECONV_LAYERS, extra.NUM_DECONV_FILTERS, extra.NUM_DECONV_KERNELS,)
        self.final_layer = nn.Conv2d(in_channels=extra.NUM_DECONV_FILTERS[-1], out_channels=cfg.MODEL.NUM_JOINTS,
                                     kernel_size=extra.FINAL_CONV_KERNEL, stride=1, padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
                                     )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        self.inplanes = self.backbone.channels[-1]
        for i in range(num_layers):     # upsample 2^(num_layers) = 2^3 = x8
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2,
                                             padding=padding, output_padding=output_padding, bias=self.deconv_with_bias
                                             ))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, list):
            x = x[-1]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return [x]

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(cfg, is_train):
    model = PoseNet(cfg, is_train)
    if is_train:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

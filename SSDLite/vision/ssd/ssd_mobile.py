from torch import nn
import torch
from torch.nn import Conv2d, ModuleList

from vision.ssd.trainer import SSDLite
from vision.ssd.predictor import Predictor
from vision.ssd.ops import InvertedResidual, SeperableConv2d
import vision.ssd.config.ssdlite_config as config
import vision.ssd.config.pretrained_paths as paths

from .backbones import pairnas, darts, nasnet, mnasnet, mobilenetv2, shufflenetv2, resnet18


def create_ssd_lite(num_classes, arch, is_test=False):
    base_net = eval(arch)()
    extras = nn.ModuleList([
        InvertedResidual(base_net.channels[-1], 512, 2, 0.2),
        InvertedResidual(512, 256, 2, 0.25),
        InvertedResidual(256, 256, 2, 0.5),
        InvertedResidual(256, 64, 2, 0.25)
    ])
    if not is_test:
        base_net.load_state_dict(torch.load('pretrained_models/{}'.format(paths.pretrained_paths[arch])))
    base_net.fc = base_net.classifier = base_net.last_linear = None
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.channels[-2], out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=base_net.channels[-1], out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.channels[-2], out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=base_net.channels[-1], out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSDLite(num_classes, base_net, extras, classification_headers, regression_headers, is_test=is_test, config=config)


# for test
def create_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean, config.image_std, nms_method=nms_method,
                          iou_threshold=config.iou_threshold, candidate_size=candidate_size, sigma=sigma, device=device)
    return predictor
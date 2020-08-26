from torch import nn
import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d

from ..ssd_mobile import SSDLite
from ..predictor import Predictor
from ..config import pdarts_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d."""
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def create_pdarts_ssd_lite(num_classes, is_test=False):
    from .networks import NetworkImageNet as Network
    from ..DARTS import genotypes
    genotype = eval("genotypes.%s" % 'PDARTS')
    base_net = Network(48, 1000, 14, False, genotype)
    extras = nn.ModuleList([
        InvertedResidual(768, 512, 2, 0.2),
        InvertedResidual(512, 256, 2, 0.25),
        InvertedResidual(256, 256, 2, 0.5),
        InvertedResidual(256, 64, 2, 0.25)
    ])
    #base_net = nn.DataParallel(base_net)
    #base_net.load_state_dict(torch.load(config.weight_path)['state_dict'])
    #base_net = base_net.module
    #base_net.auxiliary_head = None
    #torch.save(base_net.state_dict(), "PDARTS_CIFAR10_ImageNet_weight.pth")
    base_net.load_state_dict(torch.load(config.weight_path))
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=384, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=768, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=384, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=768, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSDLite(num_classes, base_net, extras, classification_headers, regression_headers, is_test=is_test, config=config)


# for test
def create_pdarts_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean, config.image_std, nms_method=nms_method,
                          iou_threshold=config.iou_threshold, candidate_size=candidate_size, sigma=sigma, device=device)
    return predictor


if __name__ == '__main__':
    net = create_pdarts_ssd_lite(21)
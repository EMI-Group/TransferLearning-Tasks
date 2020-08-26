# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.core.config import config, update_config, update_dir
from lib.core.loss import JointsMSELoss
from lib.core.function import validate
from lib.utils.utils import create_logger
from lib.utils.multadds_params_count import comp_multadds, count_parameters_in_MB

from lib import dataset
from lib import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    # It can be mobilenetv2, shufflenetv2, mnasnet, resnet18, darts, nasnet, pairnas
    parser.add_argument('--net', help='experiment configure file name', default='pairnas', type=str)
    parser.add_argument('--cfg', default='experiments/coco_256x192_d256x3_adam_lr1e-3.ymal')
    parser.add_argument('--dataset_path', default='/raid/huangsh/datasets/MSCOCO2017/')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--flip', help='use flip test', default=True)
    parser.add_argument('--post_process', help='use post process', action='store_true')
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)
    # training
    parser.add_argument('--model_file', help='model state file', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int, default=16)
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--use-detect-bbox', help='use detect bbox', action='store_true')
    parser.add_argument('--shift-heatmap', help='shift heatmap', action='store_true')
    parser.add_argument('--coco-bbox-file', help='coco detection bbox file', type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    config.DATASET.ROOT = args.dataset_path
    config.MODEL.NAME = 'pose_{}'.format(args.net)
    cfg_name = os.path.basename(args.cfg)
    dataset_name = cfg_name.split('_')[0]
    config.TEST.MODEL_FILE = 'output/{}/pose_{}/{}/model_best.pth.tar'.format(dataset_name, args.net, cfg_name.split('.yaml')[0])
    if dataset_name == 'coco':
        config.TEST.COCO_BBOX_FILE = '{}/{}'.format(args.dataset_path, config.TEST.COCO_BBOX_FILE)
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip:
        config.TEST.FLIP_TEST = args.flip
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.posenet.get_pose_net')(config, is_train=False)
    model.eval()
    params = count_parameters_in_MB(model)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=(3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)
    model.train()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        print(config.TEST.MODEL_FILE)
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    model = model.cuda()
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + config.DATASET.DATASET)(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                                              transforms.Compose([transforms.ToTensor(), normalize, ])
                                                              )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.TEST.BATCH_SIZE,
                                               shuffle=False, num_workers=config.WORKERS, pin_memory=True
                                               )
    # evaluate on validation set
    validate(config, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()

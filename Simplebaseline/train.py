from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from lib.core.config import config, update_config, update_dir, get_model_name
from lib.core.loss import JointsMSELoss
from lib.core.function import train, validate
from lib.utils.utils import get_optimizer, save_checkpoint, create_logger
from lib.utils.multadds_params_count import comp_multadds, count_parameters_in_MB

from lib.models import posenet
from lib import dataset
from lib import models

pretrained_paths = {
    'pairnas': "PairNAS_CIFAR10_ImageNet_weight.pth",
    'darts': 'DARTS_CIFAR10_ImageNet_weight.pth',
    'nasnet': 'nasnetamobile-7e03cead.pth',
    'mnasnet': 'mnasnet1.0_top1_73.512-f206786ef8.pth',
    'mobilenetv2': 'mobilenet_v2-b0353104.pth',
    'shufflenetv2': 'shufflenetv2_x1-5666bf0f80.pth',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # It can be mobilenetv2, shufflenetv2, mnasnet, darts, pairnas, nasnet
    parser.add_argument('--net', default='pairnas', type=str)
    parser.add_argument('--dataset_path', default='/raid/huangsh/datasets/MSCOCO2017/')
    parser.add_argument('--cfg', default='experiments/coco_256x192_d256x3_adam_lr1e-3.ymal')
    parser.add_argument('--model', default='posenet')
    parser.add_argument('--gpu', type=str, default='0')
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)
    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--workers', help='num of dataloader workers', type=int, default=16)
    args = parser.parse_args()
    return args


def reset_config(config, args):
    config.DATASET.ROOT = args.dataset_path
    config.MODEL.NAME = '{}_{}'.format(args.model, args.net)
    if 'coco' in args.cfg:
        config.TEST.COCO_BBOX_FILE = '{}/{}'.format(args.dataset_path, config.TEST.COCO_BBOX_FILE)
    config.MODEL.PRETRAINED = 'pretrained_models/{}'.format(pretrained_paths[args.net])
    config.GPUS = args.gpu
    config.WORKERS = args.workers


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('{}.get_pose_net'.format(args.model))(config, is_train=True)
    model.eval()
    params = count_parameters_in_MB(model)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(model, input_size=(3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)
    model.train()
    model = model.cuda()

    # copy model file
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    optimizer = get_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(config, config.DATASET.ROOT, config.DATASET.TRAIN_SET, True,
                                                            transforms.Compose([transforms.ToTensor(), normalize,])
                                                            )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(config, config.DATASET.ROOT, config.DATASET.TEST_SET, False,
                                                            transforms.Compose([transforms.ToTensor(), normalize,])
                                                            )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN.BATCH_SIZE,
                                               shuffle=config.TRAIN.SHUFFLE, num_workers=config.WORKERS, pin_memory=True
                                               )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.TEST.BATCH_SIZE,
                                               shuffle=False, num_workers=config.WORKERS, pin_memory=True
                                               )

    best_perf = 0.0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch, final_output_dir, tb_log_dir)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model, criterion, final_output_dir, tb_log_dir)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({'epoch': epoch + 1, 'model': get_model_name(config), 'state_dict': model.state_dict(),
                         'perf': perf_indicator, 'optimizer': optimizer.state_dict(),}, best_model, final_output_dir
                        )

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)

    lr_scheduler.step()


if __name__ == '__main__':
    main()

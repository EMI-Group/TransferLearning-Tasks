import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.trainer import MatchPrior
from vision.nn.multibox_loss import MultiboxLoss
from vision.utils.multadds_params_count import comp_multadds, count_parameters_in_MB
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd_mobile import create_ssd_lite

from utils.utils import create_logger

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# It can be mobilenetv2, shufflenetv2, mnasnet, darts, pairnas, nasnet
parser.add_argument('--net', type=str, default="pairnas")
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--dataset_path', type=str, default="/raid/huangsh/datasets/PASCAL_VOC/VOCdevkit/")
parser.add_argument('--checkpoint_folder', default='output/', help='Directory for saving checkpoint models')

parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int, help='the number epochs')
# Params for SGD and scheduler
parser.add_argument('--scheduler', default="cosine", type=str, help="Scheduler for SGD. It can one of multi_step and cosine")
parser.add_argument('--milestones', default="80,100", type=str, help="milestones for MultiStepLR")  # Params for Multi-step Scheduler
parser.add_argument('--num_epochs', default=200, type=int, help='the number epochs')
parser.add_argument('--t_max', default=200, type=float, help='T_max value for Cosine Annealing Scheduler.')  # Params for Cosine Annealing
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float, help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float, help='initial learning rate for the layers not in base net and prediction heads.')

# Train params
parser.add_argument('--debug_steps', default=100, type=int, help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use CUDA to train model')

args = parser.parse_args()

args.datasets = ['VOC2007', "VOC2012"]
args.validation_dataset = 'VOC2007'

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
args.validation_epochs = args.num_epochs // 20
args.t_max = args.num_epochs
if args.scheduler == 'multi_step':
    steps = [8 * args.num_epochs // 12, 10 * args.num_epochs // 12]
    args.milestones = ''
    for step in steps:
        args.milestones += '{},'.format(step)
args.checkpoint_folder = 'outputs/voc-320-{}-cosine-e{}-lr{:.3f}'.format(args.net, args.num_epochs, args.lr)
if not os.path.isdir(args.checkpoint_folder):
    os.makedirs(args.checkpoint_folder)
logger = create_logger(args.checkpoint_folder, is_train=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logger.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()
    logger.info(args)
    import vision.ssd.config.ssdlite_config as config
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    logger.info("Prepare training datasets.")
    datasets = []
    for dataset_subset in args.datasets:
        img_dir = '{}/{}'.format(args.dataset_path, dataset_subset)
        dataset = VOCDataset(img_dir, transform=train_transform, target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "voc-images-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
    logger.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    logger.info("Prepare Validation datasets.")
    img_dir = '{}/{}'.format(args.dataset_path, args.validation_dataset)
    val_dataset = VOCDataset(img_dir, transform=test_transform, target_transform=target_transform, is_test=True)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)
    logger.info("validation dataset size: {}".format(len(val_dataset)))
    logger.info("Build network.")
    net = create_ssd_lite(num_classes, arch=args.net, is_test=False)

    net.eval()
    params = count_parameters_in_MB(net)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(net, input_size=(3, 320, 320))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)

    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(net.extras.parameters()), 'lr': extra_layers_lr},
        {'params': itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())}
    ]
    timer.start("Load Model")
    logger.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    net.to(DEVICE)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, " + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi_step':
        logger.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logger.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logger.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logger.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        if (epoch+1) % args.validation_epochs == 0 or (epoch+1) == args.num_epochs:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logger.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}.pth")
            net.save(model_path)
            logger.info(f"Saved model {model_path}")
        scheduler.step()

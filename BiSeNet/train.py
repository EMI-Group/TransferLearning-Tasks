import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse

from utils.logger import setup_logger
from utils.multadds_params_count import comp_multadds, count_parameters_in_MB
from opt_utils.loss import OhemCELoss
from opt_utils.optimizer import Optimizer
from data_utils.cityscapes import CityScapes
from utils.init_func import group_weight
import numpy as np
import torch.backends.cudnn as cudnn
from BiSeNet import BiSeNet


def parse_args():
    parse = argparse.ArgumentParser()
    # It can be mobilenetv2, shufflenetv2, mnasnet, darts, pairnas, nasnet
    parse.add_argument('--net', type=str, default='pairnas')
    parse.add_argument('--dataset_path', type=str, default='/raid/huangsh/datasets/cityscapes/')
    parse.add_argument('--save_iter', type=int, default=10000)
    parse.add_argument('--max_iter', type=int, default=80000)
    parse.add_argument('--bs', type=int, default=8)
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1, )
    parse.add_argument('--nccl_ip', default='0', type=str)
    parse.add_argument('--seed', type=int, default=0, help='random seed')
    return parse.parse_args()


def train():
    args = parse_args()
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    # optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2     # 2.5e-2
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    respth = './output/{}'.format(args.net)
    if not os.path.isdir(respth):
        os.makedirs(respth)
    logger = logging.getLogger()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:3327{}'.format(args.nccl_ip), world_size=torch.cuda.device_count(), rank=args.local_rank)
    setup_logger(respth)

    # dataset
    n_classes = 19
    cropsize_wh = [1024, 1024]     # [w, h]
    ds = CityScapes(args.dataset_path, cropsize_wh=cropsize_wh, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    # model
    ignore_idx = 255
    net = BiSeNet(n_classes=n_classes, net_name=args.net, )
    net.eval()
    params = count_parameters_in_MB(net)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(net, input_size=(3, cropsize_wh[1], cropsize_wh[0]))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)
    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank, ], output_device=args.local_rank, find_unused_parameters=True)
    score_thres = 0.7
    n_min = args.bs * cropsize_wh[0] * cropsize_wh[1] // 16     # 16 maybe need to change !!!
    criteria = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    optim = Optimizer(model=net.module, lr0=lr_start, momentum=momentum, wd=weight_decay,
                      warmup_steps=warmup_steps, warmup_start_lr=warmup_start_lr, max_iter=args.max_iter, power=power)

    # train loop
    msg_iter = 50
    loss_avg = []
    loss_p = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(args.max_iter):
        try:
            img, gt = next(diter)
            if not img.size()[0] == args.bs:
                raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            img, gt = next(diter)
        img = img.cuda()
        gt = gt.cuda()
        gt = torch.squeeze(gt, 1)

        optim.zero_grad()
        feats = net(img, mode='train')
        loss = 0
        for i, feat in enumerate(feats):
            if i == 0:
                lossp = criteria(feat, gt)
                loss += lossp
            else:
                loss += criteria(feat, gt)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        loss_p.append(lossp.item())
        # print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss_p = sum(loss_p) / len(loss_p)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((args.max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(['it: {it}/{max_it}', 'lr: {lr:4f}', 'loss: {loss:.4f}', 'loss_p: {loss_p:.4f}', 'eta: {eta}', 'time: {time:.4f}',
                             ]).format(it=it+1, max_it=args.max_iter, lr=lr, loss=loss_avg, loss_p=loss_p, time=t_intv, eta=eta)
            logger.info(msg)
            loss_avg = []
            loss_p = []
            st = ed

        if (it+1) % args.save_iter == 0:
            save_pth = osp.join(respth, 'model_iter{}.pth'.format(it+1))
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save(state, save_pth)

    # dump the final model
    save_pth = osp.join(respth, 'model_iterfinal.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    train()

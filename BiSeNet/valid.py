import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import argparse
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

from utils.logger import setup_logger
from utils.multadds_params_count import comp_multadds, count_parameters_in_MB
from data_utils.cityscapes import CityScapes
from BiSeNet import BiSeNet


class MscEval(object):
    def __init__(self, model, dataloader, scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75], n_classes=19, lb_ignore=255, cropsize=1024, flip=True):
        self.scales = scales
        self.n_classes = n_classes
        self.lb_ignore = lb_ignore
        self.flip = flip
        self.cropsize = cropsize
        self.dl = dataloader    # dataloader
        self.net = model

    def pad_tensor(self, inten, size):
        N, C, H, W = inten.size()
        outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten.requires_grad = False
        margin_h, margin_w = size[0] - H, size[1] - W
        hst, hed = margin_h // 2, margin_h // 2 + H
        wst, wed = margin_w // 2, margin_w // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]

    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop, mode='eval')[0]
            prob = F.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims=(3,))
                out = self.net(crop, mode='eval')[0]
                out = torch.flip(out, dims=(3,))
                prob += F.softmax(out, 1)
            prob = torch.exp(prob)
        return prob

    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5 / 6.
        N, C, H, W = im.size()
        long_size, short_size = (H, W) if H > W else (W, H)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if H < W:
                    im, indices = self.pad_tensor(im, (cropsize, W))
                else:
                    im, indices = self.pad_tensor(im, (H, cropsize))
            N, C, H, W = im.size()
            n_x = math.ceil((W - cropsize) / stride) + 1
            n_y = math.ceil((H - cropsize) / stride) + 1
            prob = torch.zeros(N, self.n_classes, H, W).cuda()
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(H, stride * iy + cropsize), min(W, stride * ix + cropsize)
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob

    def scale_crop_eval(self, img, scale):
        N, C, H, W = img.size()
        new_hw = [int(H * scale), int(W * scale)]
        img = F.interpolate(img, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(img)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.lb_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength=n_classes ** 2)
        hist = hist.reshape((n_classes, n_classes))
        return hist

    def evaluate(self):
        # evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype=np.float32)
        dloader = tqdm(self.dl)
        if dist.is_initialized() and not dist.get_rank() == 0:
            dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, H, W = label.shape
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            imgs = imgs.cuda()
            for sc in self.scales:
                prob = self.scale_crop_eval(imgs, sc)
                probs += prob.detach().cpu()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            # cv2.imwrite('{}/{}'.format(), preds)
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU


def evaluate(save_path=None, args=None, scales=None, flip=None, net_name=None, dspath=None):
    logger = logging.getLogger()
    logger.info('\n')
    logger.info('====' * 20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')

    if save_path is None:
        import sys
        print("     ### You need to provide a valid save path of model ###      ")
        sys.exit(1)
    n_classes = 19
    logger.info(save_path)
    logger.info("       ### scales: {} and {} ###      ".format(scales, 'flip' if flip else ''))
    net = BiSeNet(n_classes=n_classes, net_name=net_name, is_train=False)
    net.load_state_dict(torch.load(save_path, map_location='cuda:0'))
    net.cuda()
    net.eval()
    params = count_parameters_in_MB(net)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(net, input_size=(3, 1024, 1024))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)
    # dataset
    dsval = CityScapes(dspath, mode='val', cropsize_wh=[1024, 1024])
    dl = DataLoader(dsval, batch_size=args.bs, shuffle=False, num_workers=2, drop_last=False)
    # evaluator
    logger.info('compute the mIOU')
    evaluator = MscEval(net, dl, scales=scales, flip=flip)
    # eval
    mIOU = evaluator.evaluate()
    logger.info('mIOU is: {:.6f}'.format(mIOU))


def parse_args():
    parse = argparse.ArgumentParser()
    # It can be mobilenetv2, shufflenetv2, mnasnet, resnet18, darts, pairnas, nasnet
    parse.add_argument('--net', type=str, default='pairnas')
    parse.add_argument('--iter', type=str, default='final')
    parse.add_argument('--dataset_path', type=str, default='/raid/huangsh/datasets/cityscapes/')
    parse.add_argument('--gpu', type=str, default='5')
    parse.add_argument('--bs', type=int, default=5)
    parse.add_argument('--ms', type=str, default='no')
    return parse.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_path = './output/{}'.format(args.net)
    setup_logger(main_path)
    save_path = '{}/model_iter{}.pth'.format(main_path, args.iter)
    if args.ms == 'yes':
        scales = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
        flip = True
    else:
        scales = [1]
        flip = False
    evaluate(save_path=save_path, args=args, scales=scales, flip=flip, net_name=args.net, dspath=args.dataset_path)

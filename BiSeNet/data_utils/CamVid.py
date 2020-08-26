import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
from data_utils.transform import *
import cv2


class TransformationVal(object):
    def __init__(self, size_wh=[1536, 768]):
        self.size_wh = size_wh

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        w, h = self.size_wh
        im = im.resize((w, h), Image.BILINEAR)
        lb = lb.resize((w, h), Image.NEAREST)
        return dict(im=im, lb=lb)


class CamVid(Dataset):
    def __init__(self, rootpth='/raid/huangsh/datasets/CamVid/', cropsize_wh=(960, 720), mode='train'):
        super(CamVid, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        if self.mode == 'train':
            self.file_txt = '{}/trainval.txt'.format(rootpth)
        else:
            self.file_txt = '{}/{}.txt'.format(rootpth, mode)
        self.dir = rootpth
        file_names, img_names, ann_names = [], [], []
        with open(self.file_txt) as f:
            files = f.readlines()
        for item in files:
            item = item.strip()
            item = item.split(' ')
            img_names.append(item[0])
            ann_names.append(item[1])
            file_names.append(item[0].split('/')[-1].split('.png')[0])
        # parse img directory
        self.imgs = {}
        impths = ["{}/{}".format(self.dir, el) for el in img_names]
        #print(impths)
        self.file_names = []
        self.file_names.extend(file_names)
        self.len = len(self.file_names)
        self.imgs.update(dict(zip(file_names, impths)))

        # parse gt directory
        self.labels = {}
        lbpths = ["{}/{}".format(self.dir, el) for el in ann_names]
        self.labels.update(dict(zip(file_names, lbpths)))

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        self.trans_train = Compose([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), HorizontalFlip(),
                                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)), RandomCrop(cropsize_wh)]
                                   )
        self.trans_val = TransformationVal(cropsize_wh)

    def __getitem__(self, idx):
        fn = self.file_names[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = Image.open(impth)
        label = Image.open(lbpth)
        if self.trans_val is None or self.mode == 'train':
            if self.mode == 'train':
                im_lb = dict(im=img, lb=label)
                im_lb = self.trans_train(im_lb)
                img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
        else:
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_val(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]

        return img, label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    ds = CamVid('./data/', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))


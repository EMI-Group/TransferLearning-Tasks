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


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize_wh=(640, 480), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255

        with open('./data_utils/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        self.map_lb = {el['trainId']: el['id'] for el in labels_info}

        # parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        im_names = os.listdir(impth)
        names = [el.replace('_leftImg8bit.png', '') for el in im_names]
        impths = [osp.join(impth, el) for el in im_names]
        imgnames.extend(names)
        self.imnames = imgnames
        self.len = len(self.imnames)
        self.imgs.update(dict(zip(names, impths)))

        # parse gt directory
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        lbnames = os.listdir(gtpth)
        lbnames = [el for el in lbnames if 'labelIds' in el]
        names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
        lbpths = [osp.join(gtpth, el) for el in lbnames]
        gtnames.extend(names)
        self.labels.update(dict(zip(names, lbpths)))

        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        # pre-processing
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        self.trans_train = Compose([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), HorizontalFlip(),
                                    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)), RandomCrop(cropsize_wh)]
                                   )
        if cropsize_wh != [1024, 1024]:
            print("     ### Without using blabla ###     ")
            self.trans_val = TransformationVal(cropsize_wh)
        else:
            self.trans_val = None

    def __getitem__(self, idx):
        fn = self.imnames[idx]
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
            label = self.convert_labels(label)
        else:
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_val(im_lb)
            img, label = im_lb['im'], im_lb['lb']
            img = self.to_tensor(img)
            label = np.array(label).astype(np.int64)[np.newaxis, :]
            label = self.convert_labels(label)

        return img, label

    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes('./data/', n_classes=19, mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))


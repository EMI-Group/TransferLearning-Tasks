import torch
import argparse
import pathlib
import numpy as np
import logging
import sys
import os

from vision.datasets.voc_dataset import VOCDataset
from vision.utils.misc import str2bool, Timer
from utils.utils import create_logger
from vision.ssd.ssd_mobile import create_ssd_lite, create_ssd_lite_predictor
from vision.utils.multadds_params_count import comp_multadds, count_parameters_in_MB
from eval_metrics import compute_average_precision_per_class, group_annotation_by_class


parser = argparse.ArgumentParser(description="SSDLite Evaluation.")
# It can be mobilenetv2, shufflenetv2, mnasnet, resnet18, darts, csonas, nasnet
parser.add_argument('--net', type=str, default="darts")
parser.add_argument("--eval_epoch", type=str, default="199")
parser.add_argument('--gpu', type=int, default='0')
parser.add_argument('--dataset_path', type=str, default='/raid/huangsh/datasets/PASCAL_VOC/VOCdevkit/VOC2007')
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


if __name__ == '__main__':
    timer = Timer()
    args.eval_epoch = 199
    dataset = VOCDataset(args.dataset_path, is_test=True)
    checkpoint_dir = 'output/voc-320-{}-cosine-e200-lr0.010'.format(args.net)
    args.eval_dir = "{}/eval_results".format(checkpoint_dir)
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    args.model = '{}/{}-Epoch-{}.pth'.format(checkpoint_dir, args.net, args.eval_epoch)
    args.label_file = './voc-model-labels.txt'
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    if not os.path.isdir(args.eval_dir):
        os.makedirs(args.eval_dir)
    logger = create_logger(args.eval_dir, is_train=False)

    timer.start("Load Model")
    net = create_ssd_lite(len(class_names), arch=args.net, is_test=True)
    net.eval()
    params = count_parameters_in_MB(net)
    logger.info("Params = %.2fMB" % params)
    mult_adds = comp_multadds(net, input_size=(3, 320, 320))
    logger.info("Mult-Adds = %.2fMB" % mult_adds)
    net.load(args.model)
    net = net.to(DEVICE)
    predictor = create_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    logger.info(f'It took {timer.end("Load Model")} seconds to load the model.')

    results = []
    timer.start("Detection")
    for i in range(len(dataset)):
        timer.start("Load Image")
        image = dataset.get_image(i)
        load_t = timer.end("Load Image")
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        predict_t = timer.end("Predict")
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        boxes = boxes + 1.0     # matlab's indexes start from 1 for pascal voc
        results.append(torch.cat([indexes.reshape(-1, 1), labels.reshape(-1, 1).float(), probs.reshape(-1, 1), boxes], dim=1))
    logger.info(args.model)
    logger.info("     ### Detecting {} imgs with using {:.3f} ###     ".format(len(dataset), timer.end("Detection")))
    results = torch.cat(results)
    results = results
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]      # all
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                img_id = dataset.ids[int(sub[i, 0])]
                print(str(img_id) + " " + " ".join([str(v) for v in prob_box]), file=f)

    aps = []
    logger.info("\n\nAverage Precision Per-class:")
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(true_case_stat[class_index], all_gb_boxes[class_index], all_difficult_cases[class_index],
                                                 prediction_path, args.iou_threshold, args.use_2007_metric
                                                 )
        aps.append(ap)
        logger.info(f"{class_name}: {ap}")
    logger.info(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")






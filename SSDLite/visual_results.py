import sys
import cv2
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
# It can be mobilenetv2, shufflenetv2, mnasnet, resnet18, darts, pairnas, nasnet
parser.add_argument('--net', type=str, default="pairnas")
parser.add_argument('--dataset_path', type=str, default="/raid/huangsh/datasets/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages/")
parser.add_argument('--output_dir', type=str, default='det_imgs')
parser.add_argument('--ths', type=float, default=0.5, help='threshold')
args = parser.parse_args()

output_dir = '{}/voc_{}'.format(args.output_dir, args.net)
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
else:
    for file in os.listdir(output_dir):
        os.remove('{}/{}'.format(output_dir, file))
threshold = args.ths

det_files = './output/voc-320-{}-cosine-e200-lr0.010/eval_results/'.format(args.net)
rs = []
for file in sorted(os.listdir(det_files)):
    print(file)
    r = pd.read_csv('{}/{}'.format(det_files, file), delimiter=" ", names=["ImageID", "Prob", "x1", "y1", "x2", "y2"])
    r['x1'] = r['x1'].astype(int)
    r['y1'] = r['y1'].astype(int)
    r['x2'] = r['x2'].astype(int)
    r['y2'] = r['y2'].astype(int)
    rs.append(r)

colors = [(179, 0, 0), (220, 128, 64), (255, 255, 51), (49, 163, 250), (0, 109, 45), (240, 2, 127), (217, 80, 14),
          (254, 153, 41), (44, 127, 184), (0, 0, 190), (169, 209, 142), (90, 255, 0), (0, 176, 240), (252, 176, 243),
          (255, 0, 255), (0, 170, 170), (255, 0, 0), (0, 255, 255), (153, 153, 153), (250, 170, 30), (220, 220, 0)
          ]

for i, r in enumerate(rs):  # different classes
    for img_id, g in r.groupby('ImageID'):
        img_id = '{}'.format(img_id)
        img_name = '{}.jpg'.format(img_id.zfill(6))
        if os.path.isfile('{}/{}'.format(output_dir, img_name)):
            file = os.path.join(output_dir, img_name)
        else:
            file = os.path.join(args.dataset_path, img_name)
        has_img = False
        for row in g.itertuples():
            if row.Prob < threshold:
                continue
            elif not has_img:
                image = cv2.imread(file)
                has_img = True
            cv2.rectangle(image, (row.x1, row.y1), (row.x2, row.y2), colors[i], 2)
            label = f"{row.Prob:.2f}"
            cv2.putText(image, label, (row.x1 + 8, row.y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)  # line type
        if has_img:
            cv2.imwrite(os.path.join(output_dir, img_name), image)

print(f"Task Done. Processed {r.shape[0]} bounding boxes.")
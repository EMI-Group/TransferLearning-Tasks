# Simple Baselines for Human Pose Estimation and Tracking

## Introduction
This is an official pytorch implementation of [*Simple Baselines for Human Pose Estimation and Tracking*](https://arxiv.org/abs/1804.06208). This work provides baseline methods that are surprisingly simple and effective, thus helpful for inspiring and evaluating new ideas for the field. You can reproduce our results using this repo. All models are provided for research purpose.    </br>

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   pip install pycocotools
   ```
3. Download pytorch imagenet pretrained models from [ImageNet model zoo](https://drive.google.com/drive/folders/1mbaYnvpOxLZRbeXCrXwlul657cXEveT0?usp=sharing).
4. Download coco pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1Syb1GttHMWhPQRSnXxXcNNSP4RSFnp9_?usp=sharing). Please download them under ${POSE_ROOT}/output, and make them look like this:

   ```
   ${POSE_ROOT}
    `-- output
        `-- coco
            |-- posenet_pairnas
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar
            |-- posenet_darts
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar
            |-- posenet_nasnet
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar
            |-- posenet_mnasnet
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar
            |-- posenet_mobilenetv2
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar
            |-- posenet_shufflenetv2
            |   |-- coco_256x192_d256x3_adam_lr1e-3
            |   |   |-- model_best.pth.tar

   ```

4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pretrained_models
   ├── README.md
   ├── train.py
   ├── valid.py
   ├── visual_results.py
   ```

### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1Syb1GttHMWhPQRSnXxXcNNSP4RSFnp9_?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${DATASET_ROOT}
|-- MSCOCO2017
  |-- |-- annotations
      |   |-- person_keypoints_train2017.json
      |   `-- person_keypoints_val2017.json
      |-- person_detection_results
      |   |-- COCO_val2017_detections_AP_H_56_person.json
  `-- images
      |-- train2017
      |   |-- 000000000009.jpg
      |   |-- 000000000025.jpg
      |   |-- 000000000030.jpg
      |   |-- ...
      `-- val2017
          |-- 000000000139.jpg
          |-- 000000000285.jpg
          |-- 000000000632.jpg
          |-- ...
```

### Valid on COCO val2017 using pretrained models

```
python valid.py --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml --flip-test --net pairnas --dataset_path "your dataset path"
```

### Training on COCO train2017

```
python train.py --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml --net pairnas --gpu 0 --dataset_path "your dataset path" 
```

### Citation
If you use this code or models in your research, please cite with:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

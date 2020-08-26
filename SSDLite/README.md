# Single Shot MultiBox Detector

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The implementation is heavily influenced by the projects [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd).
The design goal is modularity and extensibility.

Currently, it has MobileNetV2, ShuffleNetV2, NASNet, Mnasnet, DARTS, PairNAS based on SSD-Lite implementations.

## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0
4. Caffe2
5. Pandas
6. Boto3 if you want to train models on the Google OpenImages Dataset.

## Data preparation
**For PASCAL VOC data**, please download from [YOLO dataset download](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). Download and extract them under {DATASET_ROOT}/MSCOCO2017, and make them look like this:
```
${DATASET_ROOT}
|-- VOCdevkit
  |-- VOC2007
      |-- ImageSets
      |-- Annotations
      |-- JPEGImages
      |   |-- 000001.jpg
      |   |-- 000002.jpg
      |   |-- 000003.jpg
      |   |-- ...
  `-- VOC2012
      |-- ImageSets
      |-- Annotations
      |-- JPEGImages
      |   |-- 000001.jpg
      |   |-- 000002.jpg
      |   |-- 000003.jpg
      |   |-- ...
```


## Evaluation
1. Download PASCAL VOC pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/13_wJ6nC7my1KKouMkQMqyr9r1ZnLnukP?usp=sharing). Please download them under ${SSDLite_ROOT}/output, and make them look like this:

   ```
   ${SSDLite_ROOT}
    |-- output
        |-- voc-320-pairnas-cosine-e200-lr0.010
        |   |-- pairnas-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
        |-- voc-320-darts-cosine-e200-lr0.010
        |   |-- darts-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
        |-- voc-320-nasnet-cosine-e200-lr0.010
        |   |-- nasnet-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
        |-- voc-320-mnasnet-cosine-e200-lr0.010
        |   |-- mnasnet-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
        |-- voc-320-mobilenetv2-cosine-e200-lr0.010
        |   |-- mobilenetv2-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
        |-- voc-320-shufflenetv2-cosine-e200-lr0.010
        |   |-- shufflenetv2-Epoch-199.pth
        |   |-- voc-images-model-labels.txt
   ```

```bash
python valid.py --net pairnas --gpu 0 --eval_epoch 199 --dataset_path "your dataset path"
```

## Training
1. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo), and put them under ${SSDLite_ROOT}/pretrained_models.
2. The code to re-produce the model:

```bash
python train.py --net pairnas --gpu 0 --dataset_path "your dataset path"
```

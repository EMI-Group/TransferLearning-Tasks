# BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

This repo implements [BiSeNet](https://arxiv.org/pdf/1808.00897.pdf). The implementation is heavily influenced by the projects [BiSeNet](https://github.com/CoinCheung/BiSeNet).
The design goal is modularity and extensibility.

Currently, it has MobileNetV2, ShuffleNetV2, NASNet, Mnasnet, DARTS, PairNAS based on BiSeNet implementations.

## Dependencies
1. Python 3.6+
2. Pytorch 1.0

## Data preparation
**For Cityscapes data**, please download from [Cityscapes dataset download](https://www.cityscapes-dataset.com/downloads/). Download and extract them under {DATASET_ROOT}/Cityscapes, and make them look like this:
```
${DATASET_ROOT}
|-- Cityscapes
  |-- gtFine
      |-- train
      |   |-- aachen_000000_000019_leftImg8bit.png
      |   |-- aachen_000001_000019_leftImg8bit.png
      ....
      |   |-- zurich_000121_000019_leftImg8bit.png
      |-- val
  `-- LeftImg8bit
      |-- train
      |   |-- aachen_000000_000019_gtFine_color.png
      |   |-- aachen_000001_000019_gtFine_color.png
      ....
      |   |-- zurich_000121_000019_gtFine_color.png
      |-- val
```


## Evaluation
1. Download Cityscapes pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1Jr2JjXYFG2LOg49rWdcx5Us0UWFWLCwX?usp=sharing). Please download them under ${SSDLite_ROOT}/output, and make them look like this:

   ```
   ${BiSeNet_ROOT}
    |-- output
        |-- pairnas
        |   |-- model_iterfinal.pth
        |-- darts
        |   |-- model_iterfinal.pth
        |-- nasnet
        |   |-- model_iterfinal.pth
        |-- mnasnet
        |   |-- model_iterfinal.pth
        |-- mobilenetv2
        |   |-- model_iterfinal.pth
        |-- shufflenetv2
        |   |-- model_iterfinal.pth
   ```

```bash
python valid.py --net pairnas --gpu 0 --bs 5 --ms no (if used "yes") --dataset_path "your dataset path"
```

## Training
1. Download pytorch imagenet pretrained models from [pytorch model zoo](https://pytorch.org/docs/stable/model_zoo.html#module-torch.utils.model_zoo), and put them under ${SSDLite_ROOT}/pretrained_models.
2. The code to re-produce the model:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --net pairnas --gpu 0 --dataset_path "your dataset path"
```

## Requirements
* torch
* torchvision

## Datasets
* Download the ImageNet dataset from [http://www.image-net.org](http://www.image-net.org). 
* Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 

### Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model, run:
```
bash scripts/main_pretrain.sh
```

### Evaluation of linear classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```
bash scripts/main_lincls.sh
```

### Models

Our pre-trained ResNet-50 models can be downloaded from [ResNet-50](https://github.com/anonymouszyx/JCL/releases/download/v1/checkpoint_0199.pth.tar).

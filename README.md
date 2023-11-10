# SupContrast: Supervised Contrastive Learning

## Supervised Contrastive Learning with cifar10

**Resnet50, batchsize: 400, epochs: 300**

Tensorboard:
```
tensorboard --logdir=./save/SupCon/cifar10_tensorboard/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_400_temp_0.1_trial_0_cosine_warm
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset cifar10 --method SupCon --batch_size 400 --learning_rate 0.5 --temp 0.1 --cosine --epochs 300
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_linear.py --batch_size 256 --learning_rate 5 --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_400_temp_0.1_trial_0_cosine_warm/ckpt_epoch_300.pth --epochs 100
```

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, epochs: 300**

Tensorboard:
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_26_temp_0.1_trial_0_cosine
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_linear.py --batch_size 26 --batch_size_val 26 --epochs 100 --learning_rate 1 --model resnet18 --dataset path --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --size 300 --num_classes 10 --ckpt ./save/SupCon/path_models/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_26_temp_0.1_trial_0_cosine/ckpt_epoch_300.pth > epoch_300_classifier.out
```

Problem this was trained with --batch_size 16 and --size 32 (default for RandomCrop)

Tensorboard:
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_16_temp_0.1_trial_0_cosine_sizeCrop_32
```

**Resnet34, batchsize: 30, epochs: 500**

Tensorboard:
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet34_lr_0.5_decay_0.0001_bsz_30_temp_0.1_trial_0_cosine
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1,2 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet34 --epochs 500 --size 300 --batch_size 30 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=1,2 nohup python main_linear.py --batch_size 30 --batch_size_val 30 --epochs 100 --learning_rate 1 --model resnet34 --dataset path --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --size 300 --num_classes 10 --ckpt ./save/SupCon/path_models/SupCon_path_resnet34_lr_0.5_decay_0.0001_bsz_30_temp_0.1_trial_0_cosine/ckpt_epoch_500.pth > epoch_500_classifier.out
```


<p align="center">
  <img src="figures/teaser.png" width="700">
</p>

This repo covers an reference implementation for the following papers in PyTorch, using CIFAR as an illustrative example:  
(1) Supervised Contrastive Learning. [Paper](https://arxiv.org/abs/2004.11362)  
(2) A Simple Framework for Contrastive Learning of Visual Representations. [Paper](https://arxiv.org/abs/2002.05709)  

## Update

ImageNet model (small batch size with the trick of the momentum encoder) is released [here](https://www.dropbox.com/s/l4a69ececk4spdt/supcon.pth?dl=0). It achieved > 79% top-1 accuracy.

## Loss Function
The loss function [`SupConLoss`](https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11) in `losses.py` takes `features` (L2 normalized) and `labels` as input, and return the loss. If `labels` is `None` or not passed to the it, it degenerates to SimCLR.

Usage:
```python
from losses import SupConLoss

# define loss with a temperature `temp`
criterion = SupConLoss(temperature=temp)

# features: [bsz, n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# SupContrast
loss = criterion(features, labels)
# or SimCLR
loss = criterion(features)
...
```

## Comparison
Results on CIFAR-10:
|          |Arch | Setting | Loss | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  SupCrossEntropy | ResNet50 | Supervised   | Cross Entropy |  95.0  |
|  SupContrast     | ResNet50 | Supervised   | Contrastive   |  96.0  | 
|  SimCLR          | ResNet50 | Unsupervised | Contrastive   |  93.6  |

Results on CIFAR-100:
|          |Arch | Setting | Loss | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  SupCrossEntropy | ResNet50 | Supervised   | Cross Entropy |  75.3 |
|  SupContrast     | ResNet50 | Supervised   | Contrastive   |  76.5 | 
|  SimCLR          | ResNet50 | Unsupervised | Contrastive   |  70.7 |

Results on ImageNet (Stay tuned):
|          |Arch | Setting | Loss | Accuracy(%) |
|----------|:----:|:---:|:---:|:---:|
|  SupCrossEntropy | ResNet50 | Supervised   | Cross Entropy |  -  |
|  SupContrast     | ResNet50 | Supervised   | Contrastive   |  79.1 (MoCo trick)  | 
|  SimCLR          | ResNet50 | Unsupervised | Contrastive   |  -  |

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs, and/or switch to CIFAR100 by `--dataset cifar100`.  
**(1) Standard Cross-Entropy**
```
python main_ce.py --batch_size 1024 \
  --learning_rate 0.8 \
  --cosine --syncBN \
```
**(2) Supervised Contrastive Learning**  
Pretraining stage:
```
python main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine
```

<s>You can also specify `--syncBN` but I found it not crucial for SupContrast (`syncBN` 95.9% v.s. `BN` 96.0%). </s>

WARN: Currently, `--syncBN` has no effect since the code is using `DataParallel` instead of `DistributedDataParaleel`

Linear evaluation stage:
```
python main_linear.py --batch_size 512 \
  --learning_rate 5 \
  --ckpt /path/to/model.pth
```
**(3) SimCLR**  
Pretraining stage:
```
python main_supcon.py --batch_size 1024 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine --syncBN \
  --method SimCLR
```
The `--method SimCLR` flag simply stops `labels` from being passed to `SupConLoss` criterion.
Linear evaluation stage:
```
python main_linear.py --batch_size 512 \
  --learning_rate 1 \
  --ckpt /path/to/model.pth
```

On custom dataset:
```
python main_supcon.py --batch_size 1024 \
  --learning_rate 0.5  \ 
  --temp 0.1 --cosine \
  --dataset path \
  --data_folder ./path \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2675, 0.2565, 0.2761)" \
  --method SimCLR
```

The `--data_folder` must be of form ./path/label/xxx.png folowing https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder convension.

and 
## t-SNE Visualization

**(1) Standard Cross-Entropy**
<p align="center">
  <img src="figures/SupCE.jpg" width="400">
</p>

**(2) Supervised Contrastive Learning**
<p align="center">
  <img src="figures/SupContrast.jpg" width="800">
</p>

**(3) SimCLR**
<p align="center">
  <img src="figures/SimCLR.jpg" width="800">
</p>

## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```

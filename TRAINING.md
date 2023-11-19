# Training

## Supervised Contrastive Learning with cifar10

**Resnet50, batchsize: 400, epochs: 300**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset cifar10 --method SupCon --batch_size 400 --learning_rate 0.5 --temp 0.1 --cosine --epochs 300 > supCon.out
```

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, epochs: 300**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

**Resnet18, batchsize: 26, epochs: 300 (2nd try)**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag try2 > supCon_try2.out &
```
# Training

## Supervised Contrastive Learning with cifar10

**Resnet50, batchsize: 400, epochs: 300**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset cifar10 --method SupCon --batch_size 400 --learning_rate 0.5 --temp 0.1 --cosine --epochs 300 > supCon.out &
```

## Hybride Contrastive Learning with cifar10

**Resnet50, batchsize: 400, epochs: 300**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset cifar10 --method SupConHybrid --batch_size 400 --learning_rate 0.5 --temp 0.1 --cosine --epochs 300 > SupConHybrid.out &
```

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, epochs: 300**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

**Resnet18, batchsize: 26, epochs: 300 (2nd try)**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag try2 > supCon_try2.out &
```

## Cross Entropy Learning with animals10_300x300

**Resnet18, batchsize: 26, leaning rate: 0.125, epochs: 500**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag baseline > supCE_baseline.out &
```

# Adjusting Learning Rate

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, leaning rate: 0.25**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.25 --temp 0.1 --model resnet18 --epochs 10 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust1 > supCon_lrAdjust1.out &
```

**Resnet18, batchsize: 26, leaning rate: 0.125**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.125 --temp 0.1 --model resnet18 --epochs 10 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust2 > supCon_lrAdjust2.out &
```

## Cross Entropy Learning with animals10_300x300

**Resnet18, batchsize: 26, leaning rate: 0.25**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.25 --model resnet18 --epochs 10 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust1 > supCE_lrAdjust1.out &
```

**Resnet18, batchsize: 26, leaning rate: 0.125**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --model resnet18 --epochs 10 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust2 > supCE_lrAdjust2.out &
```

# Adjusting Batch Size

**Resnet18, batchsize: 52, leaning rate: 0.125**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --model resnet18 --epochs 10 --size 300 --batch_size 52 --batch_size_val 52 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag bszAdjust1 > supCE_bszAdjust1.out &
```
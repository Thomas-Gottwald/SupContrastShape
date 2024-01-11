# Animals10

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

**Resnet18, batchsize: 26, learning rate 0.125, epochs: 300 (3nd try)**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1 --data_folder ./datasets/animals10_diff/-1/train/ --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag try3 > supCon_try3.out &
```

## Cross Entropy Learning with animals10_300x300

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag baseline > supCE_baseline.out &
```

## Hybride Contrastive Learning with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, no Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupConHybrid --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag noAug > supConHybrid_noAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, color Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupConHybrid --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag colorAug > supConHybrid_colorAug.out &
```

## Supervised Contrastive Learning with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, no Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag noAug > supCon_noAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag colorAugSameShapeAug > supCon_colorAugSameShapeAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, use all same Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip sameColorJitter sameGrayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag allSameAug > supCon_allSameAug.out &
```

## Supervised Contrastive Learning with Factor with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, Factor 5, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --related_factor 5.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag factor5cAugSameSAug > supCon_factor5cAugSameSAug.out &
```

## Cross Entropy Learning with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500 original images with all augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_-1 --data_folder ./datasets/animals10_diff/-1/train/ --test_folder ./datasets/animals10_diff/-1/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag origAllAug > supCE_origAllAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_4000 --data_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/4000/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3869, 0.3732, 0.3088)" --std "(0.3273, 0.3186, 0.3039)" --tag 4000 > supCE_4000.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500 diffused 4000 with all augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_4000 --data_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/4000/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3869, 0.3732, 0.3088)" --std "(0.3273, 0.3186, 0.3039)" --tag 4000AllAug > supCE_4000AllAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_8000 --data_folder ./datasets/animals10_diff/8000/train/ --test_folder ./datasets/animals10_diff/8000/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3912, 0.3774, 0.3121)" --std "(0.3290, 0.3202, 0.3050)" --tag 8000 > supCE_8000.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500, use diffused images as augmentation**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/-1/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag diffAug > supCE_diffAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500, use diffused images as augmentation and also all kinds of augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/-1/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag diffAugAllAug > supCE_diffAugAllAug.out &
```

## No Training Baseline

"Training":
```
CUDA_VISIBLE_DEVICES=0 nohup python main_ce.py --dataset untrained --data_folder ./datasets/animals10_diff/-1/train/ --test_folder ./datasets/animals10_diff/-1/test/ --num_classes 10 --learning_rate 0 --model resnet18 --epochs 0 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag noTraining > supCE_noTraining.out
```

# Training on smaller Datasets

## Cross Entropy Learning with animals10_small

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 1000**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_-1_small --data_folder ./datasets/animals10_diff/-1/train_small/ --test_folder ./datasets/animals10_diff/-1/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 1000 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3708, 0.3579, 0.2948)" --std "(0.3253, 0.3171, 0.3019)" --tag origSmall > supCE_origSmall.out &
```

## Cross Entropy Learning with animals10_diff_small

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 1000**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_4000_small --data_folder ./datasets/animals10_diff/4000/train_small/ --test_folder ./datasets/animals10_diff/4000/test/ --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 1000 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3749, 0.3618, 0.2974)" --std "(0.3233, 0.3148, 0.2992)" --tag 4000Small > supCE_4000Small.out &
```

# Adjusting Learning Rate

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, learning rate: 0.25**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.25 --temp 0.1 --model resnet18 --epochs 10 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust1 > supCon_lrAdjust1.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.125 --temp 0.1 --model resnet18 --epochs 10 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust2 > supCon_lrAdjust2.out &
```

## Cross Entropy Learning with animals10_300x300

**Resnet18, batchsize: 26, learning rate: 0.25**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.25 --model resnet18 --epochs 10 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust1 > supCE_lrAdjust1.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --model resnet18 --epochs 10 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag lrAdjust2 > supCE_lrAdjust2.out &
```

# Adjusting Batch Size

**Resnet18, batchsize: 52, learning rate: 0.125**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10 --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --num_classes 10 --learning_rate 0.125 --model resnet18 --epochs 10 --size 300 --batch_size 52 --batch_size_val 52 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag bszAdjust1 > supCE_bszAdjust1.out &
```

# City classification

## Cross Entropy Learning with city_classification_original

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset city_classification_original --data_folder ./datasets/city_classification/Original/train/ --test_folder ./datasets/city_classification/Original/val/ --num_classes 11 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.1667, 0.1889, 0.1641)" --std "(0.1941, 0.2075, 0.1908)" --tag cityBaseline > supCE_cityBaseline.out &
```

## Cross Entropy Learning with city_classification_diff

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset city_classification_diff --data_folder ./datasets/city_classification/EEDv2_5792_as_Original5/train/ --test_folder ./datasets/city_classification/EEDv2_5792_as_Original5/val/ --num_classes 11 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.1471, 0.1704, 0.1445)" --std "(0.1896, 0.2021, 0.1862)" --tag cityDiff > supCE_cityDiff.out &
```
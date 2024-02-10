# Animals10

## Hybride Contrastive Learning with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, color Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupConHybrid --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag colorAug > supConHybrid_colorAug.out &
```

## Supervised Contrastive Learning with animals10_diff

### animals10_diff_4000

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag colorAugSameShapeAug > supCon_colorAugSameShapeAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.02, epochs: 100, fine tuning from supCon_colorAugSameShapeAug with related factor 5**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.02 --temp 0.1 --cosine --model resnet18 --epochs 100 --size 300 --batch_size 26 --method SupCon --related_factor 5.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --ckpt ./save/SupCon/animals10_diff_-1+4000/SupCon_animals10_diff_-1+4000_resnet18_lr_0.125_decay_0.0001_bsz_26_temp_0.1_trial_0_colorAugSameShapeAug_cosine/models/last.pth --save_freq 25 --tag fineTuneCAsameSAFactor5 > supCon_fineTuneCAsameSAFactor5.out &
```

<!-- -------------------------------------------------------------------------------------------------------------------------------------- -->
**Resnet18, batchsize: 26, learning rate: 0.02, epochs: 100, fine tuning from supCon_colorAugSameShapeAug with related factor 100**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.02 --temp 0.1 --cosine --model resnet18 --epochs 100 --size 300 --batch_size 26 --method SupCon --related_factor 100.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --ckpt ./save/SupCon/animals10_diff_-1+4000/SupCon_animals10_diff_-1+4000_resnet18_lr_0.125_decay_0.0001_bsz_26_temp_0.1_trial_0_colorAugSameShapeAug_cosine/models/last.pth --save_freq 25 --tag fineTuneCAsameSAFactor100 > supCon_fineTuneCAsameSAFactor100.out &
```

**Resnet18, batchsize: 26, learning rate: 0.02, epochs: 100, fine tuning from supCon_colorAugSameShapeAug with related factor 100 and no factor normalization**

For this training is it necessary to replace
```python
mean_log_prob_pos = (mask * mask_factor * log_prob).sum(1) / (mask * mask_factor).sum(1)
```
with
```python
mean_log_prob_pos = (mask * mask_factor * log_prob).sum(1) / mask.sum(1)
```
in line 109 of losses.py

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.02 --temp 0.1 --cosine --model resnet18 --epochs 100 --size 300 --batch_size 26 --method SupCon --related_factor 100.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --ckpt ./save/SupCon/animals10_diff_-1+4000/SupCon_animals10_diff_-1+4000_resnet18_lr_0.125_decay_0.0001_bsz_26_temp_0.1_trial_0_colorAugSameShapeAug_cosine/models/last.pth --save_freq 25 --tag fineTuneCAsameSAFactor100NoNormalize > supCon_fineTuneCAsameSAFactor100NoNormalize.out &
```
<!-- -------------------------------------------------------------------------------------------------------------------------------------- -->

### animals10_diff_8000

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+8000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/8000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag colorAugSameShapeAug > supCon_colorAugSameShapeAug.out &
```

## Supervised Contrastive Learning with Factor with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, Factor 5, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --related_factor 5.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag factor5cAugSameSAug > supCon_factor5cAugSameSAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 300, Factor 20, colorJitter, grayscale, sameResizedCrop and sameHorizontalFlip Augmentations**

Contrastive training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --related_factor 20.0 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag factor20cAugSameSAug > supCon_factor20cAugSameSAug.out &
```

## Cross Entropy Learning with animals10_diff

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500 original images with all augmentations**

### animals10_diff_-1

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_-1 --data_folder ./datasets/animals10_diff/-1/train/ --test_folder ./datasets/animals10_diff/-1/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag origAllAug > supCE_origAllAug.out &
```

### animals10_diff_4000

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500 diffused 4000 with all augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_4000 --data_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/4000/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3869, 0.3732, 0.3088)" --std "(0.3273, 0.3186, 0.3039)" --tag 4000AllAug > supCE_4000AllAug.out &
```

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500, use diffused images as augmentation and also all kinds of augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_ce.py --dataset animals10_diff_-1+4000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/4000/train/ --test_folder ./datasets/animals10_diff/-1/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag diffAugAllAug > supCE_diffAugAllAug.out &
```

### animals10_diff_8000

**Resnet18, batchsize: 26, learning rate: 0.125, epochs: 500, use diffused images as augmentation and also all kinds of augmentations**

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_ce.py --dataset animals10_diff_-1+8000 --data_folder ./datasets/animals10_diff/-1/train/ --diff_folder ./datasets/animals10_diff/8000/train/ --test_folder ./datasets/animals10_diff/-1/test/ --aug resizedCrop horizontalFlip colorJitter grayscale --num_classes 10 --learning_rate 0.125 --cosine --model resnet18 --epochs 500 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3816, 0.3683, 0.3052)" --std "(0.3281, 0.3198, 0.3055)" --tag diffAugAllAug > supCE_diffAugAllAug.out &
```

## No Training Baseline

"Training":
```
CUDA_VISIBLE_DEVICES=0 nohup python main_ce.py --dataset untrained --data_folder ./datasets/animals10_diff/-1/train/ --test_folder ./datasets/animals10_diff/-1/test/ --num_classes 10 --learning_rate 0 --model resnet18 --epochs 0 --size 300 --batch_size 26 --batch_size_val 26 --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag noTraining > supCE_noTraining.out
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

<!-- -------------------------------------------------------------------------------------------------------------------------------------- -->
## Supervised Contrastive Learning with city_classification_original

Training:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset city_classification_original --data_folder ./datasets/city_classification/Original/train/ --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.1667, 0.1889, 0.1641)" --std "(0.1941, 0.2075, 0.1908)" --tag cityBaseline > supCon_cityBaseline.out &
```

## Supervised Contrastive Learning with city_classification_diff

Training:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset city_classification_original+diff --data_folder ./datasets/city_classification/Original/train/ --diff_folder ./datasets/city_classification/EEDv2_5792_as_Original5/train/ --aug sameResizedCrop sameHorizontalFlip colorJitter grayscale --learning_rate 0.125 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.1667, 0.1889, 0.1641)" --std "(0.1941, 0.2075, 0.1908)" --tag cityCAugSameSAug > supCon_cityCAugSameSAug.out &
```
<!-- -------------------------------------------------------------------------------------------------------------------------------------- -->
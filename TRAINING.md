# Training

## Supervised Contrastive Learning with cifar10

**Resnet50, batchsize: 400, epochs: 300**

Tensorboard (Pretraining):
```
tensorboard --logdir=./save/SupCon/cifar10_tensorboard/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_400_temp_0.1_trial_0_cosine_warm
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset cifar10 --method SupCon --batch_size 400 --learning_rate 0.5 --temp 0.1 --cosine --epochs 300
```

Tensorboard (Classifier):
```
tensorboard --logdir=./save/classifier/cifar10_tensorboard/cifar10_resnet50_lr_5.0_decay_0_bsz_256
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_linear.py --batch_size 256 --learning_rate 5 --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_400_temp_0.1_trial_0_cosine_warm/last.pth --epochs 100  > last_classifier.out
```

## Supervised Contrastive Learning with animals10_300x300

**Resnet18, batchsize: 26, epochs: 300**

Tensorboard (Pretraining):
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_26_temp_0.1_trial_0_cosine
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_linear.py --batch_size 26 --batch_size_val 26 --epochs 100 --learning_rate 5 --model resnet18 --dataset path --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --size 300 --num_classes 10 --ckpt ./save/SupCon/path_models/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_26_temp_0.1_trial_0_cosine/last.pth > last_classifier.out
```

**Resnet18, batchsize: 26, epochs: 300 (2nd try)**

Tensorboard (Pretraining):
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_26_temp_0.1_trial_0_try2_cosine
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=2 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet18 --epochs 300 --size 300 --batch_size 26 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --tag try2 > supCon_try2.out &
```

Problem this was trained with --batch_size 16 and --size 32 (default for RandomCrop)

Tensorboard (of the training with RandomCrop size 32):
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_16_temp_0.1_trial_0_cosine_sizeCrop_32
```

**Resnet34, batchsize: 30, epochs: 500**

Problem this training takes to long (estimated time one month)

Tensorboard (Pretraining):
```
tensorboard --logdir=./save/SupCon/path_tensorboard/SupCon_path_resnet34_lr_0.5_decay_0.0001_bsz_30_temp_0.1_trial_0_cosine
```

Pretraining stage:
```
CUDA_VISIBLE_DEVICES=1,2 nohup python main_supcon.py --dataset path --data_folder ./datasets/animals10_300x300/train/ --learning_rate 0.5 --temp 0.1 --cosine --model resnet34 --epochs 500 --size 300 --batch_size 30 --method SupCon --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" > supCon.out &
```

Classifier training stage:
```
CUDA_VISIBLE_DEVICES=1,2 nohup python main_linear.py --batch_size 30 --batch_size_val 30 --epochs 100 --learning_rate 5 --model resnet34 --dataset path --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --size 300 --num_classes 10 --ckpt ./save/SupCon/path_models/SupCon_path_resnet34_lr_0.5_decay_0.0001_bsz_30_temp_0.1_trial_0_cosine/last.pth > last_classifier.out
```

----------------------

### Test out Classifier

Classifier training stage (animals10_300x300 for the resnet18 that was trained with --size 32):
```
tensorboard --logdir=./save/classifier/path_tensorboard/path_resnet18_lr_1.0_decay_0_bsz_26
```
```
CUDA_VISIBLE_DEVICES=2 nohup python main_linear.py --batch_size 26 --batch_size_val 26 --epochs 5 --learning_rate 1 --model resnet18 --dataset path --mean "(0.3837, 0.3704, 0.3072)" --std "(0.3268, 0.3187, 0.3051)" --data_folder ./datasets/animals10_300x300/train/ --test_folder ./datasets/animals10_300x300/test/ --size 300 --num_classes 10 --ckpt ./save/SupCon/path_models/SupCon_path_resnet18_lr_0.5_decay_0.0001_bsz_16_temp_0.1_trial_0_cosine_sizeCrop_32/last.pth > last_classifier.out
```

Classifier training stage (use pre compute embedding of resnet18 that was trained with --size 32):
```
tensorboard --logdir=./save/classifier/path_tensorboard/path_resnet18_lr_5.0_decay_0_bsz_26_pre_comp_feat
```
```
CUDA_VISIBLE_DEVICES=2 nohup python main_linear.py --batch_size 26 --batch_size_val 26 --epochs 100 --learning_rate 5 --model resnet18 --dataset path --data_folder ./save/embeddings/animals10/resnet18/embedding_train --test_folder ./save/embeddings/animals10/resnet18/embedding_test --num_classes 10 --pre_comp_feat > precomp_classifier.out
```

Classifier training stage on CPU (use pre compute embedding of resnet18 that was trained with --size 32):
```
tensorboard --logdir=./save/classifier/path_tensorboard/path_resnet18_lr_1.0_decay_0_bsz_26_pre_comp_feat
```
```
CUDA_VISIBLE_DEVICES="" nohup python main_linear.py --batch_size 26 --batch_size_val 26 --epochs 10 --learning_rate 1 --model resnet18 --dataset path --data_folder ./save/embeddings/animals10/resnet18/embedding_train --test_folder ./save/embeddings/animals10/resnet18/embedding_test --num_classes 10 --pre_comp_feat > precomp_classifier.out
```
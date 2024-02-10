import sys
import os
dir_file = os.path.dirname(os.path.abspath(__file__))
dir_parent = os.path.dirname(dir_file)
sys.path.append(dir_parent)

import pickle
import glob
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from matplotlib.collections import PatchCollection

from networks.resnet_big import SupCEResNet, SupConResNet, LinearClassifier, model_dict
from util.util_diff import DiffLoader, DiffTransform
from util.util_diff import SameTwoRandomResizedCrop, SameTwoColorJitter, SameTwoApply
from util.util_diff import ShufflePatches, ShuffleInnerPatches
from util.util_logging import compute_accuracies_form_cm, save_confusion_matrix, open_csv_file
from util.util_pre_com_feat import featureEmbeddingDataset


def try_eval(val):
    try:
        return eval(val)
    except:
        return val


# Parameters
def get_root_dataset(dataset):
    if dataset == "animals10" or dataset == "animals10_300x300":
        # animals10_300x300
        root_train = "./datasets/animals10_300x300/train/"
        root_test = "./datasets/animals10_300x300/test/"
    elif dataset == "animals10_diff_-1":
        root_train = "./datasets/animals10_diff/-1/train/"
        root_test = "./datasets/animals10_diff/-1/test/"
    elif dataset == "animals10_diff_-1_small":
        root_train = "./datasets/animals10_diff/-1/train_small/"
        root_test = "./datasets/animals10_diff/-1/test/"
    elif dataset == "animals10_diff_4000":
        root_train = "./datasets/animals10_diff/4000/train/"
        root_test = "./datasets/animals10_diff/4000/test/"
    elif dataset == "animals10_diff_4000_small":
        root_train = "./datasets/animals10_diff/4000/train_small/"
        root_test = "./datasets/animals10_diff/4000/test/"
    elif dataset == "animals10_diff_8000":
        root_train = "./datasets/animals10_diff/8000/train/"
        root_test = "./datasets/animals10_diff/8000/test/"
    elif dataset == "city_classification_original":
        root_train = "./datasets/city_classification/Original/train/"
        root_test = "./datasets/city_classification/Original/val/"
    elif dataset == "city_classification_diff":
        root_train = "./datasets/city_classification/EEDv2_5792_as_Original5/train/"
        root_test = "./datasets/city_classification/EEDv2_5792_as_Original5/val/"
    elif dataset == "shape_texture_conflict_animals10_two":
        root_train = None
        root_test = "./datasets/adaIN/shape_texture_conflict_animals10_two/"
    elif dataset == "shape_texture_conflict_animals10_many":
        root_train = None
        root_test = "./datasets/adaIN/shape_texture_conflict_animals10_many/"
    elif dataset == "stylized_animals10":
        root_train = None
        root_test = "./datasets/adaIN/stylized_animals10/test/"
    elif dataset in ["animals10_diff_-1PatchSize30",  "animals10_diff_-1PatchSize30CJitter",
                    "animals10_diff_-1InnerPatchSize30",  "animals10_diff_-1InnerPatchSize30CJitter",
                     "animals10_diff_-1PixelShuffled"]:
        root_train = None
        root_test = "./datasets/animals10_diff/-1/test/"
    else:
        root_train = None
        root_test = None

    return root_train, root_test

def get_dataset_augmentations(dataset):
    aug_dict = None

    if dataset == "animals10_diff_-1PatchSize30":
        aug_dict = {"aug": ["shufflePatches"], "shufflePatches": 30}
    elif dataset == "animals10_diff_-1PatchSize30CJitter":
        aug_dict = {"aug": ["colorJitter", "shufflePatches"], "shufflePatches": 30, "colorJitter": [1.0, 0.4, 0.4, 0.4, 0.4]}
    elif dataset == "animals10_diff_-1InnerPatchSize30":
        aug_dict = {"aug": ["shuffleInnerPatches"], "shufflePatches": 30}
    elif dataset == "animals10_diff_-1InnerPatchSize30CJitter":
        aug_dict = {"aug": ["colorJitter", "shuffleInnerPatches"], "shufflePatches": 30, "colorJitter": [1.0, 0.4, 0.4, 0.4, 0.4]}
    elif dataset == "animals10_diff_-1PixelShuffled":
        aug_dict = {"aug": ["shufflePatches"], "shufflePatches": 1}

    return aug_dict


def get_classes(dataset):
    _, root_test = get_root_dataset(dataset)
    if root_test:
        classes = [x[:-1].replace(root_test, '') for x in glob.glob(os.path.join(root_test, "*/"))]

        return classes
    else:
        return None


def get_paths_to_embeddings_and_run_md(root_model, dataset_1=None, dataset_2=None):# TODO remove!
    split_model = root_model.split('/')

    epoch = split_model[-1].replace(".pth", '').split('_')[-1]

    path_save = os.path.join(*split_model[:-2])
    path_run_md = os.path.join(path_save, "run.md")

    if dataset_1:
        path_val_md_1 = os.path.join(path_save, f"val_{dataset_1}.md")
        path_embeddings_1 = os.path.join(path_save, f"val_{epoch}", f"{dataset_1}", "embeddings")

        if dataset_2:
            path_val_md_2 = os.path.join(path_save, f"val_{dataset_2}.md")
            path_embeddings_2 = os.path.join(path_save, f"val_{epoch}", f"{dataset_2}", "embeddings")

            if dataset_1 < dataset_2:
                path_comb_md = os.path.join(*split_model[:-2], f"comb_{dataset_1}_{dataset_2}.md")
                path_comb = os.path.join(*split_model[:-2], f"comb_{dataset_1}_{dataset_2}")
            else:
                path_comb_md = os.path.join(*split_model[:-2], f"comb_{dataset_2}_{dataset_1}.md")
                path_comb = os.path.join(*split_model[:-2], f"comb_{dataset_2}_{dataset_1}")

            return path_save, path_run_md, path_val_md_1, path_val_md_2, path_comb_md, path_comb, path_embeddings_1, path_embeddings_2, epoch
        else:
            return path_save, path_run_md, path_val_md_1, path_embeddings_1, epoch
    else:
        return path_save, path_run_md, epoch
    

def read_parameters_from_run_md(path_run_md):# TODO remove!
    with open(path_run_md, 'r') as f:
        lines = f.readlines()

    params = dict()

    line_indices = [7, 13, 19]
    for idx in line_indices:
        params_names = lines[idx].split(' | ')
        params_names[0] = params_names[0].replace('| ', '')
        params_names[-1] = params_names[-1].replace(' |\n', '')
        params_names = [name.replace(' ', '_') for name in params_names]

        params_vals = lines[idx+2].split('|')[1:-1]
        params_vals = [try_eval(val) for val in params_vals]

        for i,n in enumerate(params_names):
            params[n] = params_vals[i]

    return params


def get_paths_from_model_checkpoint(root_model, dataset_1=None, dataset_2=None):
    split_model = root_model.split('/')

    epoch = split_model[-1].replace(".pth", '').split('_')[-1]

    path_folder = os.path.join(*split_model[:-2])

    if dataset_1:
        path_embeddings_1 = os.path.join(path_folder, f"val_{epoch}", f"{dataset_1}", "embeddings")

        if dataset_2:
            path_embeddings_2 = os.path.join(path_folder, f"val_{epoch}", f"{dataset_2}", "embeddings")

            return path_folder, path_embeddings_1, path_embeddings_2, epoch
        else:
            return path_folder, path_embeddings_1, epoch
    else:
        return path_folder, epoch


def get_path_classifier(root_model:str, dataset_classifier:str, params:dict):
    split_model = root_model.split('/')
    epoch = split_model[-1].replace(".pth", '').split('_')[-1]
    if dataset_classifier == "":
        path_classifier = glob.glob(os.path.join(*split_model[:-2], f"val_{epoch}", "classifier", f"{params['dataset']}*"))
    else:
        path_classifier = glob.glob(os.path.join(*split_model[:-2], f"val_{epoch}", f"{dataset_classifier}", "classifier", f"{params['dataset']}*"))

    if len(path_classifier) > 0:
        return path_classifier[0]
    else:
        return None
    

def check_exclude_with_params(params, exclude_params_dict=dict(), keep_params_dict=dict()):
    for p in exclude_params_dict:
        if p in params and params[p] in exclude_params_dict[p]:
            return False
    
    for p in keep_params_dict:
        if p not in params:
            return False
        elif p == "aug":
            if set(keep_params_dict[p]).intersection(params[p]) != set(keep_params_dict[p]):
                return False
        elif params[p] != keep_params_dict[p]:
            return False

    return True


def collect_models_dict(save_root="./save/", epoch="last", method="", dataset="", dataset_classifier="", exclude_params_dict=dict(), keep_params_dict=dict()):
    method_folder = "*"
    model_folder = "*"
    if method:
        method_folder = "SupCE" if method in ["CE", "SupCE"] else "SupCon"
        model_folder = "SupCE_*" if method == "CE" else f"{method}_*"
    dataset_folder = dataset if dataset else "*"
    ckpt_file = "last.pth" if epoch=="last" else f"ckpt_epoch_{epoch}.pth"
    model_roots = glob.glob(os.path.join(save_root, method_folder, dataset_folder, model_folder, "models", ckpt_file))

    repeated_names = dict()
    models_dict = dict()
    for rm in model_roots:
        path_folder, _ = get_paths_from_model_checkpoint(rm)
        params = open_csv_file(os.path.join(path_folder, "params.csv"))

        if check_exclude_with_params(params, exclude_params_dict, keep_params_dict):
            # verify that a classifier exists for dataset_classifier
            if dataset_classifier != "ignore":
                if dataset_classifier != "" or "method" in params:
                    path_classifier = get_path_classifier(rm, dataset_classifier, params)
                    if path_classifier is None or not os.path.isfile(os.path.join(path_classifier, "models", "last.pth")):
                        continue

            # create model name consisting of the used method + dataset + augmentations
            if len(params['aug']) == 0:
                aug_description = "noAug"
            elif len(params['aug']) == 1:
                aug_description = params['aug']
            else:
                if set(['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale']) == set(params['aug']):
                    aug_description = "allAug"
                elif set(['sameResizedCrop', 'sameHorizontalFlip', 'sameColorJitter', 'sameGrayscale']) == set(params['aug']):
                    aug_description = "allSameAug"
                else:
                    aug_description = ""
                    if "colorJitter" in params['aug'] and "grayscale" in params['aug']:
                        aug_description += "cAug"
                    elif "sameColorJitter" in params['aug'] and "sameGrayscale" in params['aug']:
                        aug_description += "SameCAug"
                    if "resizedCrop" in params['aug'] and "horizontalFlip" in params['aug']:
                        aug_description += "sAug"
                    elif "sameResizedCrop" in params['aug'] and "sameHorizontalFlip" in params['aug']:
                        aug_description += "SameSAug"
                    if aug_description == "":
                        aug_description = "_".join(params['aug'])
            
            model_name = (params["method"] if "method" in params else "CE")\
                    + (f"_{params['related_factor']}" if "related_factor" in params and params['related_factor'] != 1.0 else "")\
                    + f"_{params['dataset']}_{aug_description}"
            

            if model_name in models_dict:
                if model_name in repeated_names:
                    repeated_names[model_name] += 1
                else:
                    repeated_names[model_name] = 1
                model_name += f"_{repeated_names[model_name]}"
            models_dict[model_name] = [rm, dataset_classifier]
        
    for mn in repeated_names:
        for i in range(repeated_names[mn]+1):
            m_key = mn + (f"_{i}" if i > 0 else "")
            rm, dc = models_dict[m_key]
            del models_dict[m_key]
            path_folder, _ = get_paths_from_model_checkpoint(rm)
            params = open_csv_file(os.path.join(path_folder, "params.csv"))

            new_name = mn + f"_{params['tag']}" 
            models_dict[new_name] = [rm, dc]

    df_modelNames = pd.DataFrame(models_dict.keys(), columns=["model_name"]).sort_values("model_name").reset_index(drop=True)

    return models_dict, df_modelNames


# Dataloader and Model
def set_data_transforms(params, aug=[],
                        resizedCrop=[0.2, 1.0, 3/4, 4/3], horizontalFlip=0.5, colorJitter=[0.8, 0.4, 0.4, 0.4, 0.4], grayscale=0.2, shufflePatches=30):
    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])

    same_transform_list = []
    same_transform_list = []
    if 'sameResizedCrop' in aug:
        scaleMin, scaleMax, ratioMin, ratioMax = resizedCrop
        same_transform_list.append(SameTwoRandomResizedCrop(size=params['size'], scale=(scaleMin, scaleMax), ratio=(ratioMin, ratioMax)))
    if 'sameHorizontalFlip' in aug:
        same_transform_list.append(transforms.RandomApply([
            SameTwoApply(transforms.RandomHorizontalFlip(p=1.0))
        ], p=horizontalFlip))
    if 'sameColorJitter' in aug:
        pJitter, brightness, contrast, saturation, hue = colorJitter
        same_transform_list.append(transforms.RandomApply([
            SameTwoColorJitter(brightness, contrast, saturation, hue)
        ], p=pJitter))
    if 'sameGrayscale' in aug:
        same_transform_list.append(transforms.RandomApply([
            SameTwoApply(transforms.RandomGrayscale(p=1.0))
        ], p=grayscale))

    transform_list = []
    if 'resizedCrop' in aug:
        scaleMin, scaleMax, ratioMin, ratioMax = resizedCrop
        transform_list.append(transforms.RandomResizedCrop(size=params['size'], scale=(scaleMin, scaleMax), ratio=(ratioMin, ratioMax)))
    if 'horizontalFlip' in aug:
        transform_list.append(transforms.RandomHorizontalFlip(p=horizontalFlip))
    if 'colorJitter' in aug:
        pJitter, brightness, contrast, saturation, hue = colorJitter
        transform_list.append(transforms.RandomApply([
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        ], p=pJitter))
    if 'grayscale' in aug:
        transform_list.append(transforms.RandomGrayscale(p=grayscale))
    transform_list.extend([
        transforms.ToTensor(),
        normalize
    ])

    if 'shufflePatches' in aug:
        transform_list.append(ShufflePatches(patch_size=shufflePatches))
    elif 'shuffleInnerPatches' in aug:
        transform_list.append(ShuffleInnerPatches(patch_size=shufflePatches))

    if len(same_transform_list) > 0:
        aug_same_transform = transforms.Compose(same_transform_list)
    else:
        aug_same_transform = None

    aug_transform = transforms.Compose(transform_list)

    return aug_transform, aug_same_transform


def set_dataloader(dataset, params, root_train_1, root_test_1, root_train_2=None, root_test_2=None, aug_dict=None):
    if aug_dict:
        aug_transform, aug_same_transform = set_data_transforms(params=params, **aug_dict)
    else:
        aug_transform, aug_same_transform = set_data_transforms(params=params)

    if root_train_2 and root_test_2:
        diff_transform = DiffTransform(aug_transform, aug_same_transform)

        train_dataset = datasets.ImageFolder(root=root_train_1,
                                           loader=DiffLoader(path_orig=root_train_1, path_diff=root_train_2),
                                           transform=diff_transform)
        val_dataset = datasets.ImageFolder(root=root_test_1,
                                           loader=DiffLoader(path_orig=root_test_1, path_diff=root_test_2),
                                           transform=diff_transform)
    else:
        val_transform = aug_transform

        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=root_train_1,
                                             transform=val_transform,
                                             download=True)
            val_dataset = datasets.CIFAR10(root=root_test_1,
                                           train=False,
                                           transform=val_transform)
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=root_train_1,
                                              transform=val_transform,
                                              download=True)
            val_dataset = datasets.CIFAR100(root=root_test_1,
                                            train=False,
                                            transform=val_transform)
        else:
            if root_train_1:
                train_dataset = datasets.ImageFolder(root=root_train_1,
                                                    transform=val_transform)
            else:
                train_dataset = None
            if root_test_1:
                val_dataset = datasets.ImageFolder(root=root_test_1,
                                                transform=val_transform)
            else:
                val_dataset = None
    
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        train_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=params['batch_size'], shuffle=False,
            num_workers=8, pin_memory=True)
    else:
        val_loader = None
        
    return train_loader, val_loader


def set_model(root_model, params, cuda_device):
    if "num_classes" in params:
        model = SupCEResNet(name=params['model'], num_classes=params["num_classes"])
    else:
        model = SupConResNet(name=params['model'])

    ckpt = torch.load(root_model, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict

    model = model.cuda(device=cuda_device)

    model.load_state_dict(state_dict)

    return model


# Compute Embeddings
def compute_embedding(model, data_loader, params, cuda_device):
    _, embedding_size = model_dict[params['model']]

    model.eval()

    embedding = np.array([])
    class_labels = np.array([], dtype=int)
    for images, labels in tqdm(data_loader):
        images = images.cuda(device=cuda_device, non_blocking=True)

        with torch.no_grad():
            features = model.encoder(images)

        embedding = np.append(embedding, features.cpu().numpy())
        class_labels = np.append(class_labels, labels.numpy())

    embedding = embedding.reshape(-1, embedding_size)

    return embedding, class_labels


def compute_diff_embeddings(model, data_loader, params, cuda_device):
    _, embedding_size = model_dict[params['model']]

    model.eval()

    embedding_1 = np.array([])
    embedding_2 = np.array([])
    class_labels = np.array([], dtype=int)
    for images, labels in tqdm(data_loader):
        # embedding 1
        images_1 = images[0].cuda(device=cuda_device, non_blocking=True)

        with torch.no_grad():
            features_1 = model.encoder(images_1)

        embedding_1 = np.append(embedding_1, features_1.cpu().numpy())
        class_labels = np.append(class_labels, labels.numpy())

        # embedding 2
        images_2 = images[1].cuda(device=cuda_device, non_blocking=True)

        with torch.no_grad():
            features_2 = model.encoder(images_2)

        embedding_2 = np.append(embedding_2, features_2.cpu().numpy())

    embedding_1 = embedding_1.reshape(-1, embedding_size)
    embedding_2 = embedding_2.reshape(-1, embedding_size)

    return embedding_1, embedding_2, class_labels


def compute_and_save_embeddings(model, train_loader, val_loader, path_embeddings, params, cuda_device):
    if not os.path.isdir(path_embeddings):
        os.makedirs(path_embeddings)

    # Trainings Data
    if train_loader is not None:
        embedding_train, class_labels_train = compute_embedding(model, train_loader, params, cuda_device)
        images_train = [img[0].replace(train_loader.dataset.root, '') for img in train_loader.dataset.imgs]

        entry = {'data': embedding_train, 'labels': class_labels_train, 'images': images_train}
        with open(os.path.join(path_embeddings, "embedding_train"), 'wb') as f:
            pickle.dump(entry, f, protocol=-1)
    else:
        embedding_train, class_labels_train, images_train = None, None, None

    # Test Data
    if val_loader is not None:
        embedding_test, class_labels_test = compute_embedding(model, val_loader, params, cuda_device)
        images_test = [img[0].replace(val_loader.dataset.root, '') for img in val_loader.dataset.imgs]

        entry = {'data': embedding_test, 'labels': class_labels_test, 'images': images_test}
        with open(os.path.join(path_embeddings, "embedding_test"), 'wb') as f:
            pickle.dump(entry, f, protocol=-1)
    else:
        embedding_test, class_labels_test, images_test = None, None, None

    return embedding_train, class_labels_train, images_train, embedding_test, class_labels_test, images_test


# t-SNE
def save_tSNE_plots(dataset, path_embeddings, params, epoch):
    seaborn.set_theme(style="darkgrid")

    classes = get_classes(dataset)
    if classes:
        cmap = 'tab10' if len(classes)<=10 else 'tab20'
    else:
        cmap = 'tab20'

    # trainings data
    if os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_train")):
        with open(os.path.join(path_embeddings, "embedding_tSNE_train"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_tSNE_train = entry['data']
            labels_train = entry['labels']
        df_train = pd.DataFrame.from_dict({'x': embedding_tSNE_train[:,0], 'y': embedding_tSNE_train[:,1], 'label': labels_train})

        if classes:
            df_train['class'] = df_train['label'].map(lambda l: classes[l])
        else:
            df_train['class'] = df_train['label']

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.tight_layout()
        seaborn.scatterplot(df_train, x='x', y='y', hue='class', palette=cmap, ax=ax)
        ax.collections[0].set_sizes([10])
        ax.set_title(f"Data: {dataset} (Train)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
        ax.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.22))
        fig.savefig(os.path.join(path_embeddings, f"tSNE_epoch_{epoch}_train.png"), bbox_inches="tight")

    # test data
    if os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_test")):
        with open(os.path.join(path_embeddings, "embedding_tSNE_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_tSNE_test = entry['data']
            labels_test = entry['labels']
        df_test = pd.DataFrame.from_dict({'x': embedding_tSNE_test[:,0], 'y': embedding_tSNE_test[:,1], 'label': labels_test})

        if classes:
            df_test['class'] = df_test['label'].map(lambda l: classes[l])
        else:
            df_test['class'] = df_test['label']

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.tight_layout()
        seaborn.scatterplot(df_test, x='x', y='y', hue='class', palette=cmap, ax=ax)
        ax.collections[0].set_sizes([10])
        ax.set_title(f"Data: {dataset} (Test)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
        ax.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.22))
        fig.savefig(os.path.join(path_embeddings, f"tSNE_epoch_{epoch}_test.png"), bbox_inches="tight")


def plot_tSNE(dataset, path_embeddings, params, epoch):
    seaborn.set_theme(style="darkgrid")

    classes = get_classes(dataset)
    if classes:
        cmap = 'tab10' if len(classes)<=10 else 'tab20'
    else:
        cmap = 'tab20'

    if os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_train")) and os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_test")):
        fig, axs = plt.subplots(ncols=2, figsize=(13, 5))
        fig.tight_layout(w_pad=6)
        ax_train, ax_val = axs[0], axs[1]
    else:
        fig, ax = plt.subplots(ncols=1, figsize=(6, 5))
        fig.tight_layout()
        ax_train, ax_val = ax, ax

    # trainings data
    if os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_train")):
        with open(os.path.join(path_embeddings, "embedding_tSNE_train"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_tSNE_train = entry['data']
            labels_train = entry['labels']
        df_train = pd.DataFrame.from_dict({'x': embedding_tSNE_train[:,0], 'y': embedding_tSNE_train[:,1], 'label': labels_train})

        if classes:
            df_train['class'] = df_train['label'].map(lambda l: classes[l])
        else:
            df_train['class'] = df_train['label']

        seaborn.scatterplot(df_train, x='x', y='y', hue='class', palette=cmap, ax=ax_train)
        ax_train.collections[0].set_sizes([10])
        ax_train.set_title(f"Data: {dataset} (Train)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
        ax_train.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.2))

    # test data
    if os.path.isfile(os.path.join(path_embeddings, "embedding_tSNE_test")):
        with open(os.path.join(path_embeddings, "embedding_tSNE_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_tSNE_test = entry['data']
            labels_test = entry['labels']
        df_test = pd.DataFrame.from_dict({'x': embedding_tSNE_test[:,0], 'y': embedding_tSNE_test[:,1], 'label': labels_test})

        if classes:
            df_test['class'] = df_test['label'].map(lambda l: classes[l])
        else:
            df_test['class'] = df_test['label']

        seaborn.scatterplot(df_test, x='x', y='y', hue='class', palette=cmap, ax=ax_val)
        ax_val.collections[0].set_sizes([10])
        ax_val.set_title(f"Data: {dataset} (Test)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
        ax_val.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.2))


# Classifier
def move_classifier_out_file(path_classifier):
    os.makedirs(os.path.join(path_classifier, "out"), exist_ok=True)

    if os.path.isfile("classifier.out"):
        os.replace("classifier.out", os.path.join(path_classifier, "out", "classifier.out"))
    else:
        os.replace("precomp_classifier.out", os.path.join(path_classifier, "out", "precomp_classifier.out"))


def load_classifier_plots(path_classifier):
    train_loss_plot = Image.open(os.path.join(path_classifier, "tensorboard", "train_loss.png"))
    val_top1_plot = Image.open(os.path.join(path_classifier, "tensorboard", "val_top1.png"))

    fig, axs = plt.subplots(ncols=2, figsize=(10,6))
    fig.tight_layout(pad=0)
    axs[0].imshow(np.array(train_loss_plot))
    axs[0].axis('off')
    axs[1].imshow(np.array(val_top1_plot))
    axs[1].axis('off')


# Confusion Matrix
def load_classifier_checkpoint(path_classifier, model_name, num_classes, cuda_device=None):
    classifier = LinearClassifier(name=model_name, num_classes=num_classes)

    checkpoint = "last.pth"
    # if os.path.isfile(os.path.join(path_classifier, "models", "best.pth")):# TODO incorporate best classification checkpoint
    #     checkpoint = "best.pth"
    ckpt = torch.load(os.path.join(path_classifier, "models", checkpoint), map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict

    if cuda_device != None:
        classifier = classifier.cuda(device=cuda_device)

    classifier.load_state_dict(state_dict)

    return classifier


def set_up_classifier(root_model, dataset_classifier, path_embeddings, params, cuda_device):
    if os.path.isfile(os.path.join(path_embeddings, 'embedding_train')):
        train_dataset = featureEmbeddingDataset(root=os.path.join(path_embeddings, 'embedding_train'))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        train_loader = None

    if os.path.isfile(os.path.join(path_embeddings, 'embedding_test')):
        val_dataset = featureEmbeddingDataset(root=os.path.join(path_embeddings, 'embedding_test'))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=16, pin_memory=True)
    else:
        val_loader = None

    if 'method' in params:
        path_classifier = get_path_classifier(root_model=root_model, dataset_classifier=dataset_classifier, params=params)
        classifier = load_classifier_checkpoint(path_classifier=path_classifier, model_name=params["model"],
                                                num_classes=len(set(val_dataset.targets)), cuda_device=cuda_device)
    else:
        model = set_model(root_model, params, cuda_device)
        classifier = model.fc

    return classifier, train_loader, val_loader


def get_predictions(classifier, data_loader, cuda_device):
    classifier.eval()

    true_classes = np.array([], dtype=int)
    pred_classes = np.array([], dtype=int)

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.float().cuda(device=cuda_device)

            output = classifier(features)
            _, pred = output.topk(1, 1, True, True)

            true_classes = np.append(true_classes, labels.numpy())
            pred_classes = np.append(pred_classes, pred.cpu().numpy().reshape(-1))

    return pd.DataFrame.from_dict({"true_class": true_classes, "pred_class": pred_classes})


def get_confusion_matrix(df_pred, true="true_class", pred="pred_class"):
    # confusion matrix
    C = confusion_matrix(df_pred[true], df_pred[pred])

    return C


def compute_accuracies(C):
    return compute_accuracies_form_cm(C)


def compute_and_save_confusion_matrix(root_model, dataset_classifier, path_embeddings, params, epoch, cuda_device):
    classifier, train_loader, val_loader = set_up_classifier(root_model, dataset_classifier, path_embeddings, params, cuda_device)

    dataset = path_embeddings.split('/')[-2].replace("val_", '').replace(f"_{epoch}", '')
    if re.fullmatch(f"val_.+_{epoch}", dataset):
        dataset = dataset.replace("val_", '')
        dataset = "_".join(dataset.split('_')[:-1])
    classes = get_classes(dataset)

    os.makedirs(os.path.join(*path_embeddings.split('/')[:-1], "cm"), exist_ok=True)

    # trainings data
    if train_loader is not None:
        df_pred_train = get_predictions(classifier, train_loader, cuda_device)

        C_train = get_confusion_matrix(df_pred_train)
        acc_train, acc_b_train = compute_accuracies(C_train)

        save_confusion_matrix(C_train, classes, title=f"Confusion Matrix {dataset} (Train) (epoch: {epoch})",
                            path=os.path.join(*path_embeddings.split('/')[:-1], "cm", f"cm_train_epoch_{epoch}.png"))
    else:
        C_train, acc_train, acc_b_train = None, None, None

    # validation data
    if val_loader is not None:
        df_pred_val = get_predictions(classifier, val_loader, cuda_device)

        C_val = get_confusion_matrix(df_pred_val)
        acc_val, acc_b_val = compute_accuracies(C_val)

        save_confusion_matrix(C_val, classes, title=f"Confusion Matrix {dataset} (Test) (epoch: {epoch})",
                            path=os.path.join(*path_embeddings.split('/')[:-1], "cm", f"cm_val_epoch_{epoch}.png"))
    else:
        C_val, acc_val, acc_b_val = None, None, None

    return C_train, acc_train, acc_b_train, C_val, acc_val, acc_b_val
    

# Distances between two Embeddings
def compute_mean_distance(embedding_1, embedding_2, class_labels):
    # related distances
    distances_embeddings = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_1), torch.tensor(embedding_2))
    mean_distance = distances_embeddings.mean()

    # in class distances
    dist_mean_class_list = []
    for l in tqdm(range(len(set(class_labels)))):
        class_indices = np.where(class_labels == l)[0]

        embedding_class_1 = embedding_1[class_indices]
        embedding_class_2 = embedding_2[class_indices]

        mean_dist_class = 0.0
        for s in range(len(class_indices)):
            distances_embeddings_class = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_class_1), torch.tensor(embedding_class_2).roll(s, dims=0))
            mean_dist_class += distances_embeddings_class.mean()
        mean_dist_class /= len(class_indices)

        dist_mean_class_list.append(mean_dist_class.item())

    mean_distance_classes = np.mean(dist_mean_class_list)

    # all versus all distances
    mean_distances_all = 0.0

    for s in tqdm(range(len(class_labels))):
        distances_embeddings_roll = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_1), torch.tensor(embedding_2).roll(s, dims=0))
        mean_distances_all += distances_embeddings_roll.mean()

    mean_distances_all /= len(class_labels)

    return mean_distance.item(), mean_distance_classes, mean_distances_all.item()


def plot_distances(dataset_1, dataset_2, embedding_train_1, embedding_train_2, class_labels_train, embedding_test_1, embedding_test_2, class_labels_test):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    fig.tight_layout(w_pad=3)

    # Training Data Embeddings
    distances_embeddings_train = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_train_1), torch.tensor(embedding_train_2))
    distances_embeddings_train[distances_embeddings_train < 1e-8] = 0.0
    df_train = pd.DataFrame.from_dict({'dist': distances_embeddings_train, 'label': class_labels_train})
    df_train['mean'] = len(df_train)*[df_train['dist'].mean()]
    df_train['class_mean'] = df_train.groupby('label').mean()['dist'].repeat(df_train.groupby('label').count().values[:,0]).values
    df_train['index'] = df_train.index

    axs[0].scatter(np.arange(len(class_labels_train)), distances_embeddings_train, s=2, c=class_labels_train, cmap='tab10')
    df_train['class_mean'].plot(c='black', label="class mean", ax=axs[0])
    df_train['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df_train['mean'][0]:.4f})", ax=axs[0])
    axs[0].set_ylabel('cosine distance')
    axs[0].legend()
    axs[0].set_title(f"Training Data:\n{dataset_1} vs. {dataset_2}")

    # Test Data Embeddings
    distances_embeddings = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_test_1), torch.tensor(embedding_test_2))
    distances_embeddings[distances_embeddings < 1e-8] = 0.0
    df = pd.DataFrame.from_dict({'dist': distances_embeddings, 'label': class_labels_test})
    df['mean'] = len(df)*[df['dist'].mean()]
    df['class_mean'] = df.groupby('label').mean()['dist'].repeat(df.groupby('label').count().values[:,0]).values
    df['index'] = df.index

    axs[1].scatter(np.arange(len(class_labels_test)), distances_embeddings, s=2, c=class_labels_test, cmap='tab10')
    df['class_mean'].plot(c='black', label="class mean", ax=axs[1])
    df['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean'][0]:.4f})", ax=axs[1])
    axs[1].set_ylabel('cosine distance')
    axs[1].legend()
    axs[1].set_title(f"Test Data:\n{dataset_1} vs. {dataset_2}")


def plot_aug_distances(dataset_1, dataset_2, embedding_1, embedding_2, embedding_aug_1, embedding_aug_2, class_labels, aug):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(11,10))
    fig.tight_layout(h_pad=3, w_pad=3)

    distances_embeddings = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_1), torch.tensor(embedding_2))
    distances_embeddings[distances_embeddings < 1e-8] = 0.0
    distances_embeddings_aug = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_aug_1), torch.tensor(embedding_aug_2))
    distances_embeddings_aug[distances_embeddings_aug < 1e-8] = 0.0
    distances_embeddings_aug_1 = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_1), torch.tensor(embedding_aug_1))
    distances_embeddings_aug_1[distances_embeddings_aug_1 < 1e-8] = 0.0
    distances_embeddings_aug_2 = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_2), torch.tensor(embedding_aug_2))
    distances_embeddings_aug_2[distances_embeddings_aug_2 < 1e-8] = 0.0
    df = pd.DataFrame.from_dict({'dist': distances_embeddings, 'dist_aug': distances_embeddings_aug, 'dist_aug_1': distances_embeddings_aug_1,
                             'dist_aug_2': distances_embeddings_aug_2, 'label': class_labels})
    df = df.join(pd.DataFrame(
        np.repeat(df.groupby('label').mean().values, repeats=df.groupby('label').count().values[:,0], axis=0)
    ).rename(columns={0: 'class_mean', 1: 'class_mean_aug', 2: 'class_mean_aug_1', 3: 'class_mean_aug_2'}))
    df['mean'] = len(df)*[df['dist'].mean()]
    df['mean_aug'] = len(df)*[df['dist_aug'].mean()]
    df['mean_aug_1'] = len(df)*[df['dist_aug_1'].mean()]
    df['mean_aug_2'] = len(df)*[df['dist_aug_2'].mean()]

    axs[0,0].scatter(np.arange(len(class_labels)), distances_embeddings, s=2, c=class_labels, cmap='tab10')
    df['class_mean'].plot(c='black', label="class mean", ax=axs[0,0])
    df['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean'][0]:.4f})", ax=axs[0,0])
    axs[0,0].set_ylabel('cosine distance')
    axs[0,0].set_title(f"{dataset_1} vs. {dataset_2}")
    axs[0,0].legend()

    axs[0,1].scatter(np.arange(len(class_labels)), distances_embeddings_aug, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug'].plot(c='black', label="class mean", ax=axs[0,1])
    df['mean_aug'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug'][0]:.4f})", ax=axs[0,1])
    axs[0,1].set_ylabel('cosine distance')
    axs[0,1].set_title(f"{dataset_1} vs. {dataset_2} both with\n{', '.join(aug)}")
    axs[0,1].legend()

    axs[1,0].scatter(np.arange(len(class_labels)), distances_embeddings_aug_1, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug_1'].plot(c='black', label="class mean", ax=axs[1,0])
    df['mean_aug_1'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug_1'][0]:.4f})", ax=axs[1,0])
    axs[1,0].set_ylabel('cosine distance')
    axs[1,0].set_title(f"{dataset_1} vs. {dataset_1} with\n{', '.join(aug)}")
    axs[1,0].legend()

    axs[1,1].scatter(np.arange(len(class_labels)), distances_embeddings_aug_2, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug_2'].plot(c='black', label="class mean", ax=axs[1,1])
    df['mean_aug_2'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug_2'][0]:.4f})", ax=axs[1,1])
    axs[1,1].set_ylabel('cosine distance')
    axs[1,1].set_title(f"{dataset_2} vs. {dataset_2} with\n{', '.join(aug)}")
    axs[1,1].legend()


def plot_roll_distances(shifts, dataset_1, dataset_2, embedding_1, embedding_2, class_labels):
    distances_embeddings_roll = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_1), torch.tensor(embedding_2).roll(shifts, dims=0))
    df_roll = pd.DataFrame.from_dict({'dist': distances_embeddings_roll, 'label': class_labels})
    df_roll['mean'] = len(df_roll)*[df_roll['dist'].mean()]
    df_roll['class_mean'] = df_roll.groupby('label').mean()['dist'].repeat(df_roll.groupby('label').count().values[:,0]).values
    df_roll['index'] = df_roll.index

    plt.scatter(np.arange(len(class_labels)), distances_embeddings_roll, s=2, c=class_labels, cmap='tab10')
    df_roll['class_mean'].plot(c='black', label="class mean")
    df_roll['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df_roll['mean'][0]:.4f})")
    plt.ylabel('cosine distance')
    plt.title(f"{dataset_1} vs. {dataset_2} with\n shift {shifts}")
    plt.legend()


# Shape Texture Conflict Validation
def compute_miss_classified_dict(root_model, dataset, dataset_classifier, cuda_device):
    _, root_test = get_root_dataset(dataset)

    path_folder, _ = get_paths_from_model_checkpoint(root_model)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))
    path_classifier = None
    if dataset_classifier:
        path_classifier = get_path_classifier(root_model, dataset_classifier, params)
    elif "method" in params:
        path_classifier = get_path_classifier(root_model, "", params)


    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.ImageFolder(root=root_test,
                                       transform=val_transform)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params['batch_size'],
                                                 shuffle=False, num_workers=8, pin_memory=True)
    classes = val_dataset.classes

    model = set_model(root_model, params, cuda_device)
    if path_classifier:
        classifier = load_classifier_checkpoint(path_classifier, params["model"], len(classes), cuda_device)
    else:
        classifier = model.fc

    model.eval()
    classifier.eval()

    miss_classified = None

    for idx, (images, labels) in enumerate(tqdm(val_dataloader)):
        images = images.cuda(device=cuda_device, non_blocking=True)
        bsz = len(labels)

        with torch.no_grad():
            features = model.encoder(images)
            output = classifier(features)
            _, pred = output.topk(1, 1, True, True)

        imgs_batch = np.array(val_dataset.imgs[idx*params['batch_size']:idx*params['batch_size']+bsz])
        miss_classified_batch = imgs_batch[labels != pred.cpu().reshape(-1)]

        if miss_classified is None:
            miss_classified = miss_classified_batch
        else:
            miss_classified = np.append(miss_classified, miss_classified_batch, axis=0)
            
    return pd.DataFrame.from_dict({"image": miss_classified[:,0], "true_class": np.array(miss_classified[:,1], dtype=int)})


def compute_exclude_dict(models_dict, dataset_stConflict, cuda_device):
    root_orig_dict = {"./datasets/adaIN/shape_texture_conflict_animals10_two/": "animals10_diff_-1",
                      "./datasets/adaIN/shape_texture_conflict_animals10_many/": "animals10_diff_-1"}
    dataset_orig = root_orig_dict[dataset_stConflict]
    classes = get_classes(dataset_orig)

    exclude_original_dict = dict()
    for c in classes:
        exclude_original_dict[c] = []

    for m in models_dict:
        root_model, dataset_classifier = models_dict[m]

        df_miss_classified = compute_miss_classified_dict(root_model, dataset_orig, dataset_classifier, cuda_device)

        for l,c in enumerate(classes):
            exclude_original_dict[c].extend([p.split('/')[-1].split('.')[0] for p in df_miss_classified.query(f"true_class == {l}")["image"].values])

    return exclude_original_dict


class shapeTextureConflictDataset(Dataset):

    def __init__(self, root, transform=None, exclude_original_dict=None):
        self.root = root
        self.transform = transform

        self.classes = [x[:-1].replace(root, '') for x in glob.glob(os.path.join(root, "*/"))]
        self.paths = []
        self.targets_shape = []
        self.targets_texture = []

        for ls, shape in enumerate(self.classes):
            for lt, texture in enumerate(self.classes):
                if ls != lt:
                    s_t_paths = np.array(glob.glob(os.path.join(root, f"{shape}", f"{texture}", "*")))

                    if exclude_original_dict:
                        keep_indices = [idx for idx in range(len(s_t_paths))]
                        for e in exclude_original_dict[shape]:
                            for idx, s_t in enumerate(s_t_paths):
                                if re.search(f"/{e}_stylized_", s_t) and idx in keep_indices:
                                    keep_indices.remove(idx)

                        s_t_paths = s_t_paths[keep_indices]

                    self.paths.extend(s_t_paths)
                    self.targets_shape.extend(len(s_t_paths)*[ls])
                    self.targets_texture.extend(len(s_t_paths)*[lt])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        image = datasets.folder.default_loader(self.paths[index])
        target = (self.targets_shape[index], self.targets_texture[index])

        if self.transform:
            image = self.transform(image)

        return image, target


def shape_texture_predictions(model, classifier, conflict_dataloader, cuda_device):
    model.eval()
    classifier.eval()

    shape_classes = []
    texture_classes = []
    pred_classes = []

    for images, labels in tqdm(conflict_dataloader):
        images = images.cuda(device=cuda_device, non_blocking=True)

        with torch.no_grad():
            features = model.encoder(images)
            output = classifier(features)
            _, pred = output.topk(1, 1, True, True)

        shape_classes.extend(labels[0].numpy())
        texture_classes.extend(labels[1].numpy())
        pred_classes.extend(pred.cpu().numpy().reshape(-1))

    df_pred = pd.DataFrame.from_dict({"shape_class": shape_classes, "texture_class": texture_classes, "pred_class": pred_classes})

    return df_pred


def evaluate_shape_texture_conflict(models_dict, dataset_stConflict, cuda_device, exclude_miss_classified=False):
    exclude_original_dict = None
    if exclude_miss_classified:
        print("Get the union over all miss classified images")
        exclude_original_dict = compute_exclude_dict(models_dict, dataset_stConflict, cuda_device)

    print("Get predictions for shape texture cue conflict dataset")
    pred_dict = dict()

    for m in models_dict:
        root_model, dataset_classifier = models_dict[m]

        path_folder, _ = get_paths_from_model_checkpoint(root_model)
        params = open_csv_file(os.path.join(path_folder, "params.csv"))
        path_classifier = None
        if dataset_classifier:
            path_classifier = get_path_classifier(root_model, dataset_classifier, params)
        elif "method" in params:
            path_classifier = get_path_classifier(root_model, "", params)

        normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
        val_transform = transforms.Compose([transforms.Resize(params['size']), transforms.CenterCrop(params['size']), transforms.ToTensor(), normalize])

        conflict_dataset = shapeTextureConflictDataset(dataset_stConflict, val_transform, exclude_original_dict)
        classes = conflict_dataset.classes

        conflict_dataloader = torch.utils.data.DataLoader(conflict_dataset, batch_size=params['batch_size'],
                                                          shuffle=False, num_workers=8, pin_memory=True)
        
        model = set_model(root_model, params, cuda_device)
        if path_classifier:
            classifier = load_classifier_checkpoint(path_classifier, params["model"], len(classes), cuda_device)
        else:
            classifier = model.fc

        df_pred = shape_texture_predictions(model, classifier, conflict_dataloader, cuda_device)

        pred_dict[m] = df_pred

    return pred_dict, classes, exclude_original_dict


def shape_texture_conflict_bias(df_pred, shape="shape_class", texture="texture_class", pred="pred_class"):
    if len(df_pred) > 0:
        acc = len(df_pred.query(f"{shape} == {pred} or {texture} == {pred}")) / len(df_pred)
        acc_shape = len(df_pred.query(f"{shape} == {pred}")) / len(df_pred)
        acc_texture = len(df_pred.query(f"{texture} == {pred}")) / len(df_pred)
    else:
        acc, acc_shape, acc_texture = 0.0, 0.0, 0.0

    if acc != 0.0:
        shape_bias = acc_shape / acc
    else:
        shape_bias = np.nan
    
    return shape_bias, acc, acc_shape, acc_texture


def compte_texture_conflict_metrics(pred_dict, classes, shape="shape_class", texture="texture_class", pred="pred_class"):
    bias_dict = dict()
    class_biasses = dict()

    for m in pred_dict:
        class_bias_dict = dict()
        for l,c in enumerate(classes):
            # number of shape/texture images for the class ("positive")
            total_class_shape = len(pred_dict[m].query(f"{shape}=={l}"))
            total_class_texture = len(pred_dict[m].query(f"{texture}=={l}"))
            total_class = total_class_shape + total_class_texture
            # number of correct shape/texture predictions for the class ("true positive")
            true_class_shape = len(pred_dict[m].query(f"{shape}=={l} and {pred}=={l}"))
            true_class_texture = len(pred_dict[m].query(f"{texture}=={l} and {pred}=={l}"))
            true_class = true_class_shape + true_class_texture

            # compute class shape bias ["true positive shape" / ("true positive shape" + "true positive texture")]
            if true_class_shape + true_class_texture != 0.0:
                c_s_bias = true_class_shape / (true_class_shape + true_class_texture)
            else:
                c_s_bias = np.nan

            # compute class recall ["true positive" / "positive"]
            c_recall_shape = true_class_shape / total_class_shape
            c_recall_texture = true_class_texture / total_class_texture
            c_recall = true_class / total_class
            c_recall_b = (c_recall_shape + c_recall_texture) / 2

            # compute balanced class shape bias ["recall shape" / ("recall shape" + "recall texture")]
            if c_recall_shape + c_recall_texture != 0.0:
                c_s_bias_b = c_recall_shape / (c_recall_shape + c_recall_texture)
            else:
                c_s_bias_b = np.nan

            class_bias_dict[c] = {"shape_bias": c_s_bias, "shape_bias_b": c_s_bias_b,
                                  "recall": c_recall, "recall_b": c_recall_b, "recall_shape": c_recall_shape, "recall_texture": c_recall_texture}
        class_biasses[m] = pd.DataFrame(class_bias_dict)

        # shape bias and accuracies
        shape_bias, acc, acc_shape, acc_texture = shape_texture_conflict_bias(df_pred=pred_dict[m], shape=shape, texture=texture, pred=pred)

        # balanced shape bias and balanced accuracies
        acc_b_shape, acc_b_texture = class_biasses[m].loc[["recall_shape", "recall_texture"]].mean(axis=1).values
        acc_b =  acc_b_shape + acc_b_texture
        if acc_b != 0.0:
            shape_bias_b = acc_b_shape / acc_b
        else:
            shape_bias_b = np.nan

        bias_dict[m] = {"shape_bias": shape_bias, "shape_bias_b": shape_bias_b,
                        "acc": acc, "acc_b": acc_b,
                        "acc_shape": acc_shape, "acc_b_shape": acc_b_shape,
                        "acc_texture": acc_texture, "acc_b_texture": acc_b_texture}
        
    df_bias = pd.DataFrame.from_dict(bias_dict)

    return df_bias, class_biasses


def plot_shape_texture_conflict_bias(class_biasses, df_bias, ax=None, balanced=False):
    tag_sb = "shape_bias"
    title = "Shape Bias"
    xlabel = "shape bias"
    xlabel_2 = "texture bias"
    if balanced:
        tag_sb = "shape_bias_b"
        title = "Balanced Shape Bias"
        xlabel = "balanced shape bias"
        xlabel_2 = "balanced texture bias"

    if not ax:
        fig, ax = plt.subplots()

    model_names = list(class_biasses.keys())
    classes = class_biasses[model_names[0]].columns[::-1]

    yMin = -0.5
    yMax = len(classes) - 0.5

    markers = ['s', 'D', 'o', 'v', 'X', 'p', '*', 'P', 'x', '+']
    marker_sizes = [100-(i*(50//len(class_biasses))) for i in range(len(class_biasses))]
    for i,m in enumerate(class_biasses):
        ax.scatter(x=class_biasses[m].loc[tag_sb].values[::-1], y=classes, s=marker_sizes[i], marker=markers[i],
                   label=f"{m} ({tag_sb}: {df_bias[m][tag_sb]:.2f})")
    ax.vlines(df_bias.loc[tag_sb].values, ymin=-0.5, ymax=9.5, colors=[plt.colormaps["tab10"](i) for i in range(len(model_names))])

    ax.set_xlabel(xlabel)
    ax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))
    ax.set_yticks(ticks=np.arange(len(classes)), labels=classes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(yMin, yMax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.18 + len(model_names)*0.0625))
    ax.set_title(title, y=1.18 + len(model_names)*0.0625)
    secax = ax.secondary_xaxis("top")
    secax.set_xlabel(xlabel_2)
    secax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))
    secax.set_xticklabels(labels=[i/10 for i in range(10,-1,-1)])


def plot_class_recall(class_biasses, df_bias, ax=None, balanced=False):
    text_acc = r"$\mathbf{acc}$"
    tag_acc = "acc"
    tag_recall = "recall"
    title = "Class Recall"
    xlabel = "recall"
    xlabel_2 = "accuracy"
    if balanced:
        text_acc = r"$\mathbf{acc_b}$"
        tag_acc = "acc_b"
        tag_recall = "recall_b"
        title = "Balanced Class Recall"
        xlabel = "balanced recall"
        xlabel_2 = "balanced accuracy"

    if not ax:
        fig, ax = plt.subplots()

    model_names = list(class_biasses.keys())

    index = list((text_acc, *class_biasses[model_names[0]].columns))
    df_class_acc = pd.DataFrame(index=index)
    for m in class_biasses:
        df_class_acc[m] = class_biasses[m].loc[tag_recall]
    df_class_acc.loc[text_acc] = df_bias.loc[tag_acc]

    df_class_acc.plot.barh(ax=ax, width=0.85)
    ax.set_xlabel(xlabel)
    ax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(len(df_class_acc) - 0.3, -0.7)
    ax.legend([f"{m} ({tag_acc}: {100*df_bias[m][tag_acc]:.2f}%)" for m in model_names], 
              loc="upper center", bbox_to_anchor=(0.5,1.18 + len(class_biasses)*0.0625))
    ax.set_title(title, y=1.18 + len(class_biasses)*0.0625)
    secax = ax.secondary_xaxis("top")
    secax.set_xlabel(xlabel_2)
    secax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))

# from https://stackoverflow.com/questions/31908982/multi-color-legend-entry
# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='white'))

        patch = PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch


def plot_class_recall_stacked(class_biasses, df_bias, ax=None, balanced=False):
    text_acc = r"$\mathbf{acc}$"
    tag_acc_shape = "acc_shape"
    tag_acc_texture = "acc_texture"
    text_recall_shape = "recall_shape"
    text_recall_texture = "recall_texture"
    tag_recall_shape = "recall_shape"
    tag_recall_texture = "recall_texture"
    title = "Class Recall"
    xlabel = "balanced recall"
    xlabel_2 = "accuracy"
    if balanced:
        text_acc = r"$\mathbf{acc_b}$"
        tag_acc_shape = "acc_b_shape"
        tag_acc_texture = "acc_b_texture"
        title = "Balanced Class Recall"
        xlabel_2 = "balanced accuracy"

    if not ax:
        fig, ax = plt.subplots()

    model_names = list(class_biasses.keys())
    
    index = list((text_acc, *class_biasses[model_names[0]].columns))
    df_class_acc = pd.DataFrame(index=index)
    for m in class_biasses:
        df_class_acc[f"_{m}_{tag_recall_shape}"] = 0.5 * class_biasses[m].loc[tag_recall_shape]
        df_class_acc[f"_{m}_{tag_recall_texture}"] = 0.5 * class_biasses[m].loc[tag_recall_texture]
    df_class_acc.loc[text_acc, [f"_{m}_{text_recall_shape}" for m in class_biasses]] = df_bias.rename(columns=dict(zip(class_biasses,
                                                                                                           [f"_{m}_{text_recall_shape}" for m in class_biasses]))).loc[tag_acc_shape]
    df_class_acc.loc[text_acc, [f"_{m}_{text_recall_texture}" for m in class_biasses]] = df_bias.rename(columns=dict(zip(class_biasses,
                                                                                                             [f"_{m}_{text_recall_texture}" for m in class_biasses]))).loc[tag_acc_texture]

    cmap = plt.colormaps["tab20"]
    legend_handles = []
    legend_labels = []
    for i,m in enumerate(model_names):
        df_class_acc[[f"_{m}_{text_recall_shape}", f"_{m}_{text_recall_texture}"]].plot.barh(stacked=True, ax=ax, position=-i+(len(model_names)/2),
                                                                       width=0.8/len(model_names), color=[cmap(2*i), cmap(2*i+1)])
        legend_handles.append(MulticolorPatch([cmap(2*i), cmap(2*i+1)]))
        legend_labels.append(f"shape+texture {m}")

    ax.set_xlabel(xlabel)
    ax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(len(df_class_acc) - 0.3, -0.7)
    secax = ax.secondary_xaxis("top")
    secax.set_xlabel(xlabel_2)
    secax.set_xticks(ticks=np.arange(start=0, stop=1.1, step=0.1))
    ax.set_title(title, y=1.18 + len(class_biasses)*0.0625)
    ax.legend(handles=legend_handles, labels=legend_labels, handler_map={MulticolorPatch: MulticolorPatchHandler()},
              loc='upper center', bbox_to_anchor=(0.5,1.18 + len(class_biasses)*0.0625))

import sys
import os
dir_file = os.path.dirname(os.path.abspath(__file__))
dir_parent = os.path.dirname(dir_file)
sys.path.append(dir_parent)

import pickle
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn
from torchvision import transforms, datasets
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from networks.resnet_big import SupCEResNet, SupConResNet, LinearClassifier, model_dict

from util.util_diff import DiffLoader, DiffTransform
from util.util_diff import SameTwoRandomResizedCrop, SameTwoColorJitter, SameTwoApply
from util.util_logging import add_tsne_to_run_md, create_val_md
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
    elif dataset == "animals10_diff_4000":
        root_train = "./datasets/animals10_diff/4000/train/"
        root_test = "./datasets/animals10_diff/4000/test/"
    else:
        root_test = None

    return root_train, root_test


def get_classes(dataset):
    _, root_test = get_root_dataset(dataset)
    if root_test:
        classes = [x[:-1].replace(root_test, '') for x in glob.glob(os.path.join(root_test, "*/"))]

        return classes
    else:
        return None


def get_paths_to_embeddings_and_run_md(root_model, dataset_1, dataset_2=None):
    split_model = root_model.split('/')

    epoch = split_model[-1].replace(".pth", '').split('_')[-1]

    training_dataset = split_model[-4]

    path_save = os.path.join(*split_model[:-2])
    path_run_md = os.path.join(path_save, "run.md")
    if dataset_1 == training_dataset:
        path_val_md_1 = path_run_md
        path_embeddings_1 = os.path.join(path_save, f"val_{epoch}", "embeddings")
    else:
        path_val_md_1 = os.path.join(path_save, f"val_{dataset_1}.md")
        path_embeddings_1 = os.path.join(path_save, f"val_{dataset_1}_{epoch}", "embeddings")

    if dataset_2:
        if dataset_2 == training_dataset:
            path_val_md_2 = path_run_md
            path_embeddings_2 = os.path.join(path_save, f"val_{epoch}", "embeddings")
        else:
            path_val_md_2 = os.path.join(path_save, f"val_{dataset_2}.md")
            path_embeddings_2 = os.path.join(path_save, f"val_{dataset_2}_{epoch}", "embeddings")

        if dataset_1 < dataset_2:
            path_comb_md = os.path.join(*split_model[:-2], f"comb_{dataset_1}_{dataset_2}.md")
            path_comb = os.path.join(*split_model[:-2], f"comb_{dataset_1}_{dataset_2}")
        else:
            path_comb_md = os.path.join(*split_model[:-2], f"comb_{dataset_2}_{dataset_1}.md")
            path_comb = os.path.join(*split_model[:-2], f"comb_{dataset_2}_{dataset_1}")

        return path_save, path_run_md, path_val_md_1, path_val_md_2, path_comb_md, path_comb, path_embeddings_1, path_embeddings_2, epoch
    else:
        return path_save, path_run_md, path_val_md_1, path_embeddings_1, epoch
    

def read_parameters_from_run_md(path_run_md):
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


def get_path_classifier(root_model, dataset, params, epoch):
    split_model = root_model.split('/')
    if dataset == params['dataset']:
        path_classifier = glob.glob(os.path.join(*split_model[:-2], f"val_{epoch}", "classifier", "*/"))
    else:
        path_classifier = glob.glob(os.path.join(*split_model[:-2], f"val_{dataset}_{epoch}", "classifier", "*/"))
    if len(path_classifier) > 0:
        path_classifier = (path_classifier[0])[:-1]

    return path_classifier


# Dataloader and Model
def set_data_transforms(params, aug=[],
                      resizedCrop=[0.2, 1.0, 3/4, 4/3], horizontalFlip=0.5, colorJitter=[0.8, 0.4, 0.4, 0.4, 0.4], grayscale=0.2):
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
            train_dataset = datasets.ImageFolder(root=root_train_1,
                                                 transform=val_transform)
            val_dataset = datasets.ImageFolder(root=root_test_1,
                                               transform=val_transform)
            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=8, pin_memory=True)
        
    return train_loader, val_loader


def set_model(root_model, params, val_loader, cuda_device):
    if params['method'] == "SupCE":
        model = SupCEResNet(name=params['model'], num_classes=len(val_loader.dataset.classes))
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


def compute_and_save_embeddings(model, dataset, train_loader, val_loader, path_val_md, path_embeddings, params, cuda_device):
    if not os.path.isdir(path_embeddings):
        os.makedirs(path_embeddings)

    if params['dataset'] != dataset:
        create_val_md(path_val_md=path_val_md, dataset_val=dataset)

    # Trainings Data
    embedding_train, class_labels_train = compute_embedding(model, train_loader, params, cuda_device)

    entry = {'data': embedding_train, 'labels': class_labels_train}
    with open(os.path.join(path_embeddings, "embedding_train"), 'wb') as f:
        pickle.dump(entry, f, protocol=-1)

    # Test Data
    embedding_test, class_labels_test = compute_embedding(model, val_loader, params, cuda_device)

    entry = {'data': embedding_test, 'labels': class_labels_test}
    with open(os.path.join(path_embeddings, "embedding_test"), 'wb') as f:
        pickle.dump(entry, f, protocol=-1)

    return embedding_train, class_labels_train, embedding_test, class_labels_test


# t-SNE
def save_tSNE_plots(dataset, path_val_md, path_embeddings, params, epoch):
    seaborn.set_theme(style="darkgrid")

    with open(os.path.join(path_embeddings, "embedding_tSNE_train"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_tSNE_train = entry['data']
        labels_train = entry['labels']
    df_train = pd.DataFrame.from_dict({'x': embedding_tSNE_train[:,0], 'y': embedding_tSNE_train[:,1], 'label': labels_train})

    with open(os.path.join(path_embeddings, "embedding_tSNE_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_tSNE_test = entry['data']
        labels_test = entry['labels']
    df_test = pd.DataFrame.from_dict({'x': embedding_tSNE_test[:,0], 'y': embedding_tSNE_test[:,1], 'label': labels_test})

    classes = get_classes(dataset)
    if classes:
        df_train['class'] = df_train['label'].map(lambda l: classes[l])
        df_test['class'] = df_test['label'].map(lambda l: classes[l])
    else:
        df_train['class'] = df_train['label']
        df_test['class'] = df_test['label']

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout()
    seaborn.scatterplot(df_train, x='x', y='y', hue='class', palette='tab10', ax=ax)
    ax.collections[0].set_sizes([10])
    ax.set_title(f"Data: {dataset} (Train)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.22))
    fig.savefig(os.path.join(path_embeddings, f"tSNE_epoch_{epoch}_train.png"), bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout()
    seaborn.scatterplot(df_test, x='x', y='y', hue='class', palette='tab10', ax=ax)
    ax.collections[0].set_sizes([10])
    ax.set_title(f"Data: {dataset} (Test)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
    ax.legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.22))
    fig.savefig(os.path.join(path_embeddings, f"tSNE_epoch_{epoch}_test.png"), bbox_inches="tight")
    
    add_tsne_to_run_md(path=path_val_md, epoch=epoch, dataset_val=(None if params['dataset']==dataset else dataset))


def plot_tSNE(dataset, path_embeddings, params, epoch):
    seaborn.set_theme(style="darkgrid")

    with open(os.path.join(path_embeddings, "embedding_tSNE_train"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_tSNE_train = entry['data']
        labels_train = entry['labels']
    df_train = pd.DataFrame.from_dict({'x': embedding_tSNE_train[:,0], 'y': embedding_tSNE_train[:,1], 'label': labels_train})

    with open(os.path.join(path_embeddings, "embedding_tSNE_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_tSNE_test = entry['data']
        labels_test = entry['labels']
    df_test = pd.DataFrame.from_dict({'x': embedding_tSNE_test[:,0], 'y': embedding_tSNE_test[:,1], 'label': labels_test})

    classes = get_classes(dataset)
    if classes:
        df_train['class'] = df_train['label'].map(lambda l: classes[l])
        df_test['class'] = df_test['label'].map(lambda l: classes[l])
    else:
        df_train['class'] = df_train['label']
        df_test['class'] = df_test['label']

    fig, axs = plt.subplots(ncols=2, figsize=(11, 5))
    fig.tight_layout(h_pad=4)

    seaborn.scatterplot(df_train, x='x', y='y', hue='class', palette='tab10', ax=axs[0])
    axs[0].collections[0].set_sizes([10])
    axs[0].set_title(f"Data: {dataset} (Train)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
    axs[0].legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.2))

    seaborn.scatterplot(df_test, x='x', y='y', hue='class', palette='tab10', ax=axs[1])
    axs[1].collections[0].set_sizes([10])
    axs[1].set_title(f"Data: {dataset} (Test)\nModel: {params['model']}, bsz={params['batch_size']} (epoch {epoch})", y=1.2)
    axs[1].legend(loc='upper center', ncols=4, bbox_to_anchor=(0.5,1.2))


# Classifier
def move_classifier_out_file(path_classifier):
    os.makedirs(os.path.join(path_classifier, "out"), exist_ok=True)

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
def set_up_classifier(path_classifier, path_embeddings, params, cuda_device):
    train_dataset = featureEmbeddingDataset(root=os.path.join(path_embeddings, 'embedding_train'))

    val_dataset = featureEmbeddingDataset(root=os.path.join(path_embeddings, 'embedding_test'))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False,
        num_workers=16, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=16, pin_memory=True)

    classifier = LinearClassifier(name=params['model'], num_classes=len(set(val_dataset.targets)))

    ckpt = torch.load(os.path.join(path_classifier, "models", "last.pth"), map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict

    classifier = classifier.cuda(device=cuda_device)

    classifier.load_state_dict(state_dict)

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


def get_confusion_matrix(df_pred):
    # confusion matrix
    C = confusion_matrix(df_pred["true_class"], df_pred["pred_class"])

    c_lens = df_pred.groupby('true_class').count().values.reshape(-1)

    # accuracy
    acc = 0.0
    for i in range(len(c_lens)):
        acc += C[i,i]
    acc *= 100/len(df_pred)
    acc
    # balanced accuracy
    acc_b = 0.0
    for i, n in enumerate(c_lens):
        acc_b += C[i,i] / n
    acc_b *= 100/len(c_lens)
    acc_b

    return C, acc, acc_b


def save_confusion_matrix(C, classes, path, title="Confusion Matrix"):
    seaborn.set_theme(style="ticks")

    disp = ConfusionMatrixDisplay(C, display_labels=classes)
    disp.plot(xticks_rotation=45)
    for labels in disp.text_.ravel():
        labels.set_fontsize(10)
    disp.ax_.set_title(title)
    disp.figure_.tight_layout(pad=0.5)

    plt.savefig(path)


def compute_and_save_confusion_matrix(root_model, dataset, path_embeddings, params, epoch, cuda_device):
    path_classifier = get_path_classifier(root_model, dataset, params, epoch)
    classifier, train_loader, val_loader = set_up_classifier(path_classifier, path_embeddings, params, cuda_device)

    df_pred_train = get_predictions(classifier, train_loader, cuda_device)
    df_pred_val = get_predictions(classifier, val_loader, cuda_device)

    C_train, acc_train, acc_b_train = get_confusion_matrix(df_pred_train)
    C_val, acc_val, acc_b_val = get_confusion_matrix(df_pred_val)

    classes = get_classes(dataset)

    save_confusion_matrix(C_train, classes, title=f"Confusion Matrix {dataset} (Train) (epoch: {epoch})",
                        path=os.path.join(path_classifier, "models", f"cm_train_epoch_{epoch}.png"))
    save_confusion_matrix(C_val, classes, title=f"Confusion Matrix {dataset} (Test) (epoch: {epoch})",
                        path=os.path.join(path_classifier, "models", f"cm_val_epoch_{epoch}.png"))
    
    return C_train, acc_train, acc_b_train, C_val, acc_val, acc_b_val
    

# Distances between two Embeddings
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
    df_train['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df_train['mean'][0]:.3f})", ax=axs[0])
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
    df['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean'][0]:.3f})", ax=axs[1])
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
    df['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean'][0]:.3f})", ax=axs[0,0])
    axs[0,0].set_ylabel('cosine distance')
    axs[0,0].set_title(f"{dataset_1} vs. {dataset_2}")
    axs[0,0].legend()

    axs[0,1].scatter(np.arange(len(class_labels)), distances_embeddings_aug, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug'].plot(c='black', label="class mean", ax=axs[0,1])
    df['mean_aug'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug'][0]:.3f})", ax=axs[0,1])
    axs[0,1].set_ylabel('cosine distance')
    axs[0,1].set_title(f"{dataset_1} vs. {dataset_2} both with\n{', '.join(aug)}")
    axs[0,1].legend()

    axs[1,0].scatter(np.arange(len(class_labels)), distances_embeddings_aug_1, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug_1'].plot(c='black', label="class mean", ax=axs[1,0])
    df['mean_aug_1'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug_1'][0]:.3f})", ax=axs[1,0])
    axs[1,0].set_ylabel('cosine distance')
    axs[1,0].set_title(f"{dataset_1} vs. {dataset_1} with\n{', '.join(aug)}")
    axs[1,0].legend()

    axs[1,1].scatter(np.arange(len(class_labels)), distances_embeddings_aug_2, s=2, c=class_labels, cmap='tab10')
    df['class_mean_aug_2'].plot(c='black', label="class mean", ax=axs[1,1])
    df['mean_aug_2'].plot(c='darkred', linestyle='--', label=f"mean ({df['mean_aug_2'][0]:.3f})", ax=axs[1,1])
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
    df_roll['mean'].plot(c='darkred', linestyle='--', label=f"mean ({df_roll['mean'][0]:.3f})")
    plt.ylabel('cosine distance')
    plt.title(f"{dataset_1} vs. {dataset_2} with\n shift {shifts}")
    plt.legend()


# Combine validation results of two Datasets
def get_comb_results(root_model, dataset_1, dataset_2, path_save, path_comb, path_embeddings_1, path_embeddings_2, params, epoch, cuda_device):
    path_classifier_1 = get_path_classifier(root_model, dataset_1, params, epoch)
    path_classifier_2 = get_path_classifier(root_model, dataset_2, params, epoch)

    folder_epoch_1 = f"val_{epoch}" if params['dataset']==dataset_1 else f"val_{dataset_1}_{epoch}"
    folder_epoch_2 = f"val_{epoch}" if params['dataset']==dataset_2 else f"val_{dataset_2}_{epoch}"

    all_val_folders = [folder_path.split('/')[-1] for folder_path in glob.glob(os.path.join(path_save, f"val_*[!.md]"))]
    # all_comb_folders = [os.path.join(*folder_path.split('/')[-2:]) for folder_path in glob.glob(os.path.join(path_save, f"comb_*[!.md]", "*"))]

    all_epochs = set([val_folder.split('_')[-1] for val_folder in all_val_folders])

    comb_dict = dict()
    for e in all_epochs:
        # check that this epoch checkpoint was evaluated on both datasets
        folder_e_1 = f"val_{e}" if params['dataset']==dataset_1 else f"val_{dataset_1}_{e}"
        folder_e_2 = f"val_{e}" if params['dataset']==dataset_2 else f"val_{dataset_2}_{e}"
        if folder_e_1 in all_val_folders and folder_e_2 in all_val_folders:

            # get the plots of the t-SNE embeddings
            path_embeddings_e_1 = path_embeddings_1.replace(folder_epoch_1, folder_e_2)
            path_embeddings_e_2 = path_embeddings_2.replace(folder_epoch_2, folder_e_2)
            tSNE_plots_1 = glob.glob(os.path.join(path_embeddings_e_1, "*.png"))
            tSNE_plots_2= glob.glob(os.path.join(path_embeddings_e_2, "*.png"))
            if len(tSNE_plots_1) == 2 and len(tSNE_plots_2) == 2:
                comb_dict[f'tSNE{e}'] = (tSNE_plots_1, tSNE_plots_2)

            # get the plots of the confusion matrices
            path_classifier_e_1 = path_classifier_1.replace(folder_epoch_1, folder_e_1).replace(f"_Epoch{epoch}", f"_Epoch{e}")
            path_classifier_e_2 = path_classifier_2.replace(folder_epoch_2, folder_e_2).replace(f"_Epoch{epoch}", f"_Epoch{e}")
            cm_plots_1 = glob.glob(os.path.join(path_classifier_e_1, "models", "*.png"))
            cm_plots_2 = glob.glob(os.path.join(path_classifier_e_2, "models", "*.png"))
            if len(cm_plots_1) == 2 and len(cm_plots_2) == 2:
                seaborn.set_theme(style="ticks")
                classes = get_classes(dataset_1)

                os.makedirs(os.path.join(path_comb, f"{e}"), exist_ok=True)
                cm_comb_plots_1 = [os.path.join(path_comb, f"{e}", f"cm_{dataset_2}_train_epoch_{e}.png"),
                                os.path.join(path_comb, f"{e}", f"cm_{dataset_2}_test_epoch_{e}.png")]
                cm_comb_plots_2 = [os.path.join(path_comb, f"{e}", f"cm_{dataset_1}_train_epoch_{e}.png"),
                                        os.path.join(path_comb, f"{e}", f"cm_{dataset_1}_test_epoch_{e}.png")]
                
                # evaluate classifiers
                # classifier trained on dataset_1 evaluate on dataset_1
                classifier, train_loader, val_loader = set_up_classifier(path_classifier_e_1, path_embeddings_e_1, params, cuda_device)
                df_pred_train = get_predictions(classifier, train_loader, cuda_device)
                df_pred_val = get_predictions(classifier, val_loader, cuda_device)
                C_train, acc_train, acc_b_train = get_confusion_matrix(df_pred_train)
                C_val, acc_val, acc_b_val = get_confusion_matrix(df_pred_val)
                comb_dict[f'acc{e}'] = [[acc_train, acc_b_train, acc_val, acc_b_val]]

                # classifier trained on dataset_1 evaluate on dataset_2
                classifier, train_loader, val_loader = set_up_classifier(path_classifier_e_1, path_embeddings_e_2, params, cuda_device)
                df_pred_train = get_predictions(classifier, train_loader, cuda_device)
                df_pred_val = get_predictions(classifier, val_loader, cuda_device)
                C_train, acc_train, acc_b_train = get_confusion_matrix(df_pred_train)
                C_val, acc_val, acc_b_val = get_confusion_matrix(df_pred_val)
                save_confusion_matrix(C_train, classes, title=f"Confusion Matrix {dataset_2} (Train) (epoch: {e})",
                                    path=cm_comb_plots_1[0])
                save_confusion_matrix(C_val, classes, title=f"Confusion Matrix {dataset_2} (Test) (epoch: {e})",
                                    path=cm_comb_plots_1[1])
                comb_dict[f'acc{e}'].append([acc_train, acc_b_train, acc_val, acc_b_val])

                # classifier trained on dataset_second evaluate on dataset_second
                classifier, train_loader, val_loader = set_up_classifier(path_classifier_e_2, path_embeddings_e_2, params, cuda_device)
                df_pred_train = get_predictions(classifier, train_loader, cuda_device)
                df_pred_val = get_predictions(classifier, val_loader, cuda_device)
                C_train, acc_train, acc_b_train = get_confusion_matrix(df_pred_train)
                C_val, acc_val, acc_b_val = get_confusion_matrix(df_pred_val)
                comb_dict[f'acc{e}'].append([acc_train, acc_b_train, acc_val, acc_b_val])

                # classifier trained on dataset_second evaluate on dataset_val
                classifier, train_loader, val_loader = set_up_classifier(path_classifier_e_2, path_embeddings_e_1, params, cuda_device)
                df_pred_train = get_predictions(classifier, train_loader, cuda_device)
                df_pred_val = get_predictions(classifier, val_loader, cuda_device)
                C_train, acc_train, acc_b_train = get_confusion_matrix(df_pred_train)
                C_val, acc_val, acc_b_val = get_confusion_matrix(df_pred_val)
                save_confusion_matrix(C_train, classes, title=f"Confusion Matrix {dataset_1} (Train) (epoch: {e})",
                                    path=cm_comb_plots_2[0])
                save_confusion_matrix(C_val, classes, title=f"Confusion Matrix {dataset_1} (Test) (epoch: {e})",
                                    path=cm_comb_plots_2[1])
                comb_dict[f'acc{e}'].append([acc_train, acc_b_train, acc_val, acc_b_val])
                
                comb_dict[f'cm{e}'] = (cm_plots_1, cm_comb_plots_1, cm_plots_2, cm_comb_plots_2)


        # # check that a comb folder for both datasets exits
        # comb_folder_e = None
        # if os.path.join(f"comb_{dataset_1}_{dataset_2}", f"{e}") in all_comb_folders:
        #     comb_folder_e = os.path.join(f"comb_{dataset_1}_{dataset_2}", f"{e}")
        # elif os.path.join(f"comb_{dataset_2}_{dataset_1}", f"{e}") in all_comb_folders:
        #     comb_folder_e = os.path.join(f"comb_{dataset_2}_{dataset_1}", f"{e}")
        # if comb_folder_e:
        #     print("comb", e, comb_folder_e)

    return all_epochs, comb_dict
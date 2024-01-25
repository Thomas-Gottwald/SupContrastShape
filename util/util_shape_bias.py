import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
import matplotlib.pyplot as plt
import seaborn
from torchvision import transforms, datasets

import util.util_validation as ut_val
from tqdm import tqdm

from util.util_logging import open_csv_file
from util.util_diff import DiffLoader, DiffTransform
from util.util_diff import SameTwoColorJitter, SameTwoApply


# Cue Conflict Shape Bias
def evaluate_cue_conflict_dataset(root_model, dataset_classifier, root_dataset_conflict, cuda_device, exclude_original_dict=dict()):
    path_folder, _ = ut_val.get_paths_from_model_checkpoint(root_model)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))
    if dataset_classifier is None:
        path_classifier = ut_val.get_path_classifier(root_model, "", params)
    else:
        path_classifier = ut_val.get_path_classifier(root_model, dataset_classifier, params)

    normalize = transforms.Normalize(mean=params["mean"], std=params["std"])
    val_transform = transforms.Compose([transforms.Resize(params["size"]), transforms.CenterCrop(params["size"]), transforms.ToTensor(), normalize])

    conflict_dataset = ut_val.shapeTextureConflictDataset(root_dataset_conflict, val_transform, exclude_original_dict)
    classes = conflict_dataset.classes

    conflict_dataloader = torch.utils.data.DataLoader(conflict_dataset, batch_size=params["batch_size"],
                                                      shuffle=False, num_workers=params["num_workers"], pin_memory=True)

    model = ut_val.set_model(root_model, params, cuda_device)
    if path_classifier:
        classifier = ut_val.load_classifier_checkpoint(path_classifier, params["model"], len(classes), cuda_device)
    else:
        assert "num_classes" in params
        classifier = model.fc
    
    df_pred = ut_val.shape_texture_predictions(model, classifier, conflict_dataloader, cuda_device)

    return df_pred, classes


def evaluate_cue_conflict_dataset_for_many(models_dict, root_dataset_conflict, cuda_device, exclude_original_dict=dict()):
    print("Get predictions for shape texture cue conflict dataset")
    pred_dict = dict()

    for m in models_dict:
        root_model, dataset_classifier = models_dict[m]

        df_pred, classes = evaluate_cue_conflict_dataset(root_model, dataset_classifier, root_dataset_conflict, cuda_device, exclude_original_dict)
        pred_dict[m] = df_pred

    return pred_dict, classes


def compute_cue_conflict_shape_bias_metric(df_pred, classes, model_short_name, shape="shape_class", texture="texture_class", pred="pred_class"):
    class_bias_dict = dict()

    for l,c in enumerate(classes):
        # number of shape/texture images for the class ("positive")
        total_class_shape = len(df_pred.query(f"{shape}=={l}"))
        total_class_texture = len(df_pred.query(f"{texture}=={l}"))
        total_class = total_class_shape + total_class_texture
        # number of correct shape/texture predictions for the class ("true positive")
        true_class_shape = len(df_pred.query(f"{shape}=={l} and {pred}=={l}"))
        true_class_texture = len(df_pred.query(f"{texture}=={l} and {pred}=={l}"))
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
    df_class_bias= pd.DataFrame(class_bias_dict)

    # shape bias and accuracies
    shape_bias, acc, acc_shape, acc_texture = ut_val.shape_texture_conflict_bias(df_pred=df_pred, shape=shape, texture=texture, pred=pred)

    # balanced shape bias and balanced accuracies
    acc_b_shape, acc_b_texture = df_class_bias.loc[["recall_shape", "recall_texture"]].mean(axis=1).values
    acc_b =  acc_b_shape + acc_b_texture
    if acc_b != 0.0:
        shape_bias_b = acc_b_shape / acc_b
    else:
        shape_bias_b = np.nan

    df_bias = pd.DataFrame.from_dict({"shape_bias": shape_bias, "shape_bias_b": shape_bias_b,
                                      "acc": acc, "acc_b": acc_b,
                                      "acc_shape": acc_shape, "acc_b_shape": acc_b_shape,
                                      "acc_texture": acc_texture, "acc_b_texture": acc_b_texture},
                                     orient="index", columns=[model_short_name])

    return df_bias, df_class_bias


def compute_cue_conflict_shape_bias_metric_for_many(pred_dict, classes, shape="shape_class", texture="texture_class", pred="pred_class"):
    bias_list = []
    class_biasses = dict()

    for m in pred_dict:
        df_bias, df_class_bias = compute_cue_conflict_shape_bias_metric(pred_dict[m], classes, m, shape, texture, pred)

        bias_list.append(df_bias)
        class_biasses[m] = df_class_bias

    df_biases = pd.concat(bias_list, axis=1)

    return df_biases, class_biasses


def cue_conflict_shape_bias_metric(root_model, dataset_classifier, model_short_name, root_dataset_conflict, cuda_device):
    df_pred, classes = evaluate_cue_conflict_dataset(root_model, dataset_classifier, root_dataset_conflict, cuda_device)
    df_bias, df_class_bias = compute_cue_conflict_shape_bias_metric(df_pred, classes, model_short_name)

    return df_bias, df_class_bias


def cue_conflict_shape_bias_metric_for_many(models_dict, root_dataset_conflict, cuda_device):
    pred_dict, classes = evaluate_cue_conflict_dataset_for_many(models_dict, root_dataset_conflict, cuda_device)
    df_biases, class_biasses = compute_cue_conflict_shape_bias_metric_for_many(pred_dict, classes)

    return df_biases, class_biasses


def save_cue_conflict_shape_bias_plots(root_model, df_bias, class_bias, dataset_cue_conflict, stacked=True):
    path_folder, epoch = ut_val.get_paths_from_model_checkpoint(root_model)

    path_cue_conf_shape_bias = os.path.join(path_folder, f"val_{epoch}", "shapeBiasMetrics", "CueConflict", dataset_cue_conflict)
    os.makedirs(path_cue_conf_shape_bias, exist_ok=True)

    fig, ax = plt.subplots()
    fig.tight_layout()
    ut_val.plot_shape_texture_conflict_bias(class_bias, df_bias, ax=ax)
    fig.savefig(os.path.join(path_cue_conf_shape_bias, "shape_bias.png"), bbox_inches="tight")

    fig, ax = plt.subplots()
    fig.tight_layout()
    if stacked:
        ut_val.plot_class_recall_stacked(class_bias, df_bias, ax=ax)
    else:
        ut_val.plot_class_recall(class_bias, df_bias, ax=ax)
    fig.savefig(os.path.join(path_cue_conf_shape_bias, "classes_recall.png"), bbox_inches="tight")
    

def save_cue_conflict_shape_bias_to_csv_file(root_model, df_bias, df_class_bias, dataset_cue_conflict, save_plots=False):
    path_folder, epoch = ut_val.get_paths_from_model_checkpoint(root_model)

    path_cue_conf_shape_bias = os.path.join(path_folder, f"val_{epoch}", "shapeBiasMetrics", "CueConflict", dataset_cue_conflict)
    os.makedirs(path_cue_conf_shape_bias, exist_ok=True)

    if save_plots:
        class_bias = {df_bias.columns[0]: df_class_bias}
        save_cue_conflict_shape_bias_plots(root_model, df_bias, class_bias, dataset_cue_conflict)

    df_bias.columns = [dataset_cue_conflict]
    df_bias.to_csv(os.path.join(path_cue_conf_shape_bias, "shape_bias.csv"))
    df_class_bias.to_csv(os.path.join(path_cue_conf_shape_bias, "classes_shape_bias.csv"))


def save_cue_conflict_shape_bias_to_csv_file_for_many(models_dict, df_biases, class_biases, dataset_cue_conflict, save_plots=False):
    for m in models_dict:
        root_model, _ = models_dict[m]
        df_bias = df_biases[[m]]

        save_cue_conflict_shape_bias_to_csv_file(root_model, df_bias, class_biases[m], dataset_cue_conflict, save_plots)


# Correlation Coefficient Shape Bias
class ShufflePatches:
    # inspired from https://stackoverflow.com/questions/66962837/shuffle-patches-in-image-batch
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, x):
        # unfold the tensor image
        u = nnf.unfold(x, kernel_size=self.patch_size , stride=self.patch_size , padding=0)
        # shuffle the patches in unfolded form
        pu = u[:,torch.randperm(u.shape[-1])]
        # fold the tensor back in its original form
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        return f
    
    def apply_to_batch(self, x):
        # unfold the tensor image
        u = nnf.unfold(x, kernel_size=self.patch_size , stride=self.patch_size , padding=0)
        # shuffle the patches in unfolded form
        pu = torch.stack([b_[:,torch.randperm(b_.shape[-1])] for b_ in u], dim=0)
        # fold the tensor back in its original form
        f = nnf.fold(pu, x.shape[-2:], kernel_size=self.patch_size, stride=self.patch_size, padding=0)

        return f
    

class TwoDifferentApply:

    def __init__(self, transform_orig=None, transform_diff=None):
        self.transform_orig = transform_orig
        self.transform_diff = transform_diff

    def __call__(self, x):
        x_orig, x_diff = x

        if self.transform_orig:
            x_orig = self.transform_orig(x_orig)
        if self.transform_diff:
            x_diff = self.transform_diff(x_diff)

        return [x_orig, x_diff]
    

def compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, use_colorJitter=False, patch_size=30):
    path_folder, path_embeddings_orig, path_embeddings_diff, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1=dataset_orig, dataset_2=dataset_shape)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    _, root_dataset_orig = ut_val.get_root_dataset(dataset=dataset_orig)

    # compute patch shuffled embeddings of the original dataset to keep texture information but remove shape information
    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
    shufflePatchers_transform_list = []
    if use_colorJitter:
        shufflePatchers_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.4))
    shufflePatchers_transform_list.extend([transforms.ToTensor(), ShufflePatches(patch_size=patch_size), normalize])
    shufflePatchers_transform = transforms.Compose(shufflePatchers_transform_list)

    shufflePatches_dataset = datasets.ImageFolder(root=root_dataset_orig,transform=shufflePatchers_transform)
    shufflePatches_dataloader = torch.utils.data.DataLoader(shufflePatches_dataset, batch_size=params['batch_size'],
                                                            shuffle=False, num_workers=16, pin_memory=True)

    model = ut_val.set_model(root_model, params, cuda_device)

    embedding_texture, _ = ut_val.compute_embedding(model, shufflePatches_dataloader, params, cuda_device)

    # load the pre computed original embeddings
    with open(os.path.join(path_embeddings_orig, "embedding_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_orig = entry['data']
        class_labels = entry['labels']

    if use_colorJitter:
        # compute the color jittered embeddings of the shape dataset
        _, root_dataset_shape = ut_val.get_root_dataset(dataset=dataset_shape)

        shape_transform = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                              transforms.ToTensor(), normalize])
        
        shape_dataset = datasets.ImageFolder(root=root_dataset_shape,transform=shape_transform)
        shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=params['batch_size'],
                                                       shuffle=False, num_workers=16, pin_memory=True)
        
        embedding_shape, _ = ut_val.compute_embedding(model, shape_dataloader, params, cuda_device)
    else:
        # load pre computed embeddings of the shape dataset
        with open(os.path.join(path_embeddings_diff, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_shape = entry['data']

    return embedding_orig, embedding_shape, embedding_texture, class_labels


def compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, use_colorJitter=False, patch_size=30):
    path_folder, path_embeddings_orig, path_embeddings_diff, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1=dataset_orig, dataset_2=dataset_shape)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    _, root_dataset_orig = ut_val.get_root_dataset(dataset=dataset_orig)

    # load the pre computed original embeddings
    with open(os.path.join(path_embeddings_orig, "embedding_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_orig = entry['data']
        class_labels = entry['labels']

    if use_colorJitter:
        # compute the embeddings of the shape dataset and the patch shuffled original dataset with the same color jitter applied to both
        _, root_dataset_shape = ut_val.get_root_dataset(dataset=dataset_shape)

        normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
        colorJitterOrigPatchShuffle_transform = transforms.Compose([SameTwoColorJitter(0.4, 0.4, 0.4, 0.4), SameTwoApply(transforms.ToTensor()),
                                                                   TwoDifferentApply(transform_orig=ShufflePatches(patch_size=patch_size))])
        
        textureShape_dataset = datasets.ImageFolder(root=root_dataset_orig,
                                                    loader=DiffLoader(path_orig=root_dataset_orig, path_diff=root_dataset_shape),
                                                    transform=DiffTransform(normalize, colorJitterOrigPatchShuffle_transform))
        textureShape_dataloader = torch.utils.data.DataLoader(textureShape_dataset, batch_size=params['batch_size'],
                                                                shuffle=False, num_workers=params["num_workers"], pin_memory=True)
        
        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_texture, embedding_shape, _ = ut_val.compute_diff_embeddings(model, textureShape_dataloader, params, cuda_device)
        
    else:
        # load pre computed embeddings of the shape dataset
        with open(os.path.join(path_embeddings_diff, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_shape = entry['data']

        # compute patch shuffled embeddings of the original dataset to keep texture information but remove shape information
        normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
        shufflePatchers_transform = transforms.Compose([transforms.ToTensor(), ShufflePatches(patch_size=patch_size), normalize])

        shufflePatches_dataset = datasets.ImageFolder(root=root_dataset_orig,transform=shufflePatchers_transform)
        shufflePatches_dataloader = torch.utils.data.DataLoader(shufflePatches_dataset, batch_size=params['batch_size'],
                                                                shuffle=False, num_workers=params["num_workers"], pin_memory=True)

        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_texture, _ = ut_val.compute_embedding(model, shufflePatches_dataloader, params, cuda_device)

    return embedding_orig, embedding_shape, embedding_texture, class_labels


def compute_dim_correlation_coefficients(embedding_A, embedding_B):
    A = torch.tensor(embedding_A)
    B = torch.tensor(embedding_B)

    A_dm = A - A.mean(dim=0)
    B_dm = B - B.mean(dim=0)

    correlation = (A_dm * B_dm).sum(dim=0) / ((A_dm * A_dm).sum(dim=0) * (B_dm * B_dm).sum(dim=0)).sqrt()
    correlation = torch.nan_to_num(correlation, nan=0.0)

    return correlation.numpy()


def estimate_dims(correlation):
    correlation_scores = np.mean(np.array(correlation), axis=1)

    scores = np.array(np.concatenate((correlation_scores, [1.0])))

    m = np.max(scores)
    e = np.exp(scores-m)
    softmaxed = e / np.sum(e)

    dim = len(correlation[0])
    dims = [int(s*dim) for s in softmaxed]
    dims[-1] = dim - sum(dims[:-1])

    return dims


def compute_corelation_coefficient_shape_bias_metric(embedding_orig, embedding_shape, embedding_texture, model_short_name, use_colorJitter=False):
    corr_coef_shape = compute_dim_correlation_coefficients(embedding_orig, embedding_shape)
    corr_coef_texture = compute_dim_correlation_coefficients(embedding_orig, embedding_texture)
    if use_colorJitter:
        corr_coef_color = compute_dim_correlation_coefficients(embedding_shape, embedding_texture)

        dims = estimate_dims([corr_coef_shape, corr_coef_texture, corr_coef_color])

        df_dims = pd.DataFrame.from_dict({model_short_name: dims}, orient="index", columns=["shape_dims", "texture_dims", "color_dims", "remaining_dims"])
    else:
        dims = estimate_dims([corr_coef_shape, corr_coef_texture])

        df_dims = pd.DataFrame.from_dict({model_short_name: dims}, orient="index", columns=["shape_dims", "texture_dims", "remaining_dims"])

    return df_dims


def corelation_coefficient_shape_bias_metric(root_model, model_short_name, dataset_orig, dataset_shape, cuda_device, use_colorJitter=False, patch_size=30):
    embedding_orig, embedding_shape, embedding_texture, _ = compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, use_colorJitter, patch_size)

    df_dims = compute_corelation_coefficient_shape_bias_metric(embedding_orig, embedding_shape, embedding_texture, model_short_name, use_colorJitter)

    return df_dims


def corelation_coefficient_shape_bias_metric_for_many(models_dict, dataset_orig, dataset_shape, cuda_device, use_colorJitter=False, patch_size=30):
    dims_list = []
    for m in models_dict:
        root_model, _ = models_dict[m]

        embedding_orig, embedding_shape, embedding_texture, _ = compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, use_colorJitter, patch_size)

        df_dims = compute_corelation_coefficient_shape_bias_metric(embedding_orig, embedding_shape, embedding_texture, m, use_colorJitter)
        dims_list.append(df_dims)

    return pd.concat(dims_list, axis=0)


# Feature Embedding Distances
def load_compute_orig_diff_embeddings(root_model, dataset_orig, dataset_diff, cuda_device, aug_dict=None):
    path_folder, path_embeddings_orig, path_embeddings_diff, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1=dataset_orig, dataset_2=dataset_diff)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    if aug_dict:
        # compute original and diffused embeddings with data augmentations from aug_dict
        root_dataset_train_orig, root_dataset_test_orig = ut_val.get_root_dataset(dataset=dataset_orig)
        root_dataset_train_diff, root_dataset_test_diff = ut_val.get_root_dataset(dataset=dataset_diff)

        _, val_loader = ut_val.set_dataloader("", params, root_dataset_train_orig, root_dataset_test_orig, root_dataset_train_diff, root_dataset_test_diff, aug_dict)

        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_orig, embedding_diff, class_labels = ut_val.compute_diff_embeddings(model, val_loader, params, cuda_device)
    else:
        # load pre computed original and diffused embeddings
        with open(os.path.join(path_embeddings_orig, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_orig = entry['data']
            class_labels = entry['labels']

        with open(os.path.join(path_embeddings_diff, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_diff = entry['data']

    return embedding_orig, embedding_diff, class_labels


def compute_distance_mean_std(embedding_orig, embedding_diff, class_labels, model_short_name, cuda_device=-1):
    # related distances
    distances_embeddings = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_orig), torch.tensor(embedding_diff))
    std_distance, mean_distance = torch.std_mean(distances_embeddings)
    mean_distance = mean_distance.item()
    std_distance = std_distance.item()

    # in class distances
    sum_distances_class = 0.0
    count_distances_class = 0
    for l in tqdm(range(len(set(class_labels)))):
        class_indices = np.where(class_labels == l)[0]

        embedding_class_orig = embedding_orig[class_indices]
        embedding_class_diff = embedding_diff[class_indices]

        for s in range(len(class_indices)):
            tensor_class_orig = torch.tensor(embedding_class_orig)
            tensor_class_diff = torch.tensor(embedding_class_diff)
            if cuda_device != -1:
                tensor_class_orig = tensor_class_orig.cuda(cuda_device)
                tensor_class_diff = tensor_class_diff.cuda(cuda_device)
            distances_embeddings_class = 1-torch.nn.functional.cosine_similarity(tensor_class_orig, tensor_class_diff.roll(s, dims=0))
            sum_distances_class += distances_embeddings_class.sum()
            count_distances_class += len(distances_embeddings_class)

    mean_distance_classes = (sum_distances_class / count_distances_class).item()


    sum_dm_distances_class = 0.0
    for l in tqdm(range(len(set(class_labels)))):
        class_indices = np.where(class_labels == l)[0]

        embedding_class_orig = embedding_orig[class_indices]
        embedding_class_diff = embedding_diff[class_indices]

        for s in range(len(class_indices)):
            tensor_class_orig = torch.tensor(embedding_class_orig)
            tensor_class_diff = torch.tensor(embedding_class_diff)
            if cuda_device != -1:
                tensor_class_orig = tensor_class_orig.cuda(cuda_device)
                tensor_class_diff = tensor_class_diff.cuda(cuda_device)
            distances_embeddings_class = 1-torch.nn.functional.cosine_similarity(tensor_class_orig, tensor_class_diff.roll(s, dims=0))
            sum_dm_distances_class += ((distances_embeddings_class - mean_distance_classes)**2).sum()

    std_distance_class = (sum_dm_distances_class / (count_distances_class-1)).sqrt().item()

    # all versus all distances
    sum_distances_all = 0.0
    count_distances_all = 0
    min_distance = np.inf
    max_distance = -np.inf
    for s in tqdm(range(len(class_labels))):
        tensor_orig = torch.tensor(embedding_orig)
        tensor_diff = torch.tensor(embedding_diff)
        if cuda_device != -1:
            tensor_orig = tensor_orig.cuda(cuda_device)
            tensor_diff = tensor_diff.cuda(cuda_device)
        distances_embeddings_all = 1-torch.nn.functional.cosine_similarity(tensor_orig, tensor_diff.roll(s, dims=0))
        sum_distances_all += distances_embeddings_all.sum()
        count_distances_all += len(distances_embeddings_all)

        min_dist = distances_embeddings_all.min().item()
        max_dist = distances_embeddings_all.max().item()
        if min_dist < min_distance:
            min_distance = min_dist
        if max_dist > max_distance:
            max_distance = max_dist

    mean_distance_all = (sum_distances_all / count_distances_all).item()


    sum_dm_distances_all = 0.0
    for s in tqdm(range(len(class_labels))):
        tensor_orig = torch.tensor(embedding_orig)
        tensor_diff = torch.tensor(embedding_diff)
        if cuda_device != -1:
            tensor_orig = tensor_orig.cuda(cuda_device)
            tensor_diff = tensor_diff.cuda(cuda_device)
        distances_embeddings_all = 1-torch.nn.functional.cosine_similarity(tensor_orig, tensor_diff.roll(s, dims=0))
        sum_dm_distances_all += ((distances_embeddings_all - mean_distance_all)**2).sum()

    std_distance_all = (sum_dm_distances_all / (count_distances_all-1)).sqrt().item()


    df_dist = pd.DataFrame.from_dict({"mean_distance_related": [mean_distance], "std_distance_related": [std_distance],
                                      "mean_distance_classes": [mean_distance_classes], "std_distance_classes": [std_distance_class],
                                      "mean_distance_all_vs_all": [mean_distance_all], "std_distance_all_vs_all": [std_distance_all],
                                      "min_distance": [min_distance], "max_distance": [max_distance]},
                                      orient="index", columns=[model_short_name])

    return df_dist


def plot_distance_histograms(df_dist, embedding_orig, embedding_diff, class_labels, model_short_name, cuda_device=-1, n_bins=100, ax=None, dataset_orig=None, dataset_diff=None, save_plot_path=None):
    min_distance = df_dist.loc["min_distance", model_short_name]
    max_distance = df_dist.loc["max_distance", model_short_name]

    # related distances
    distances_embeddings = 1-torch.nn.functional.cosine_similarity(torch.tensor(embedding_orig), torch.tensor(embedding_diff))
    hist_related, bin_edges = np.histogram(distances_embeddings, bins=n_bins, range=(min_distance, max_distance))
    hist_related = hist_related.astype(float) / np.sum(hist_related)

    # in class distances
    hist_class = np.zeros(n_bins)
    for l in tqdm(range(len(set(class_labels)))):
        class_indices = np.where(class_labels == l)[0]

        embedding_class_orig = embedding_orig[class_indices]
        embedding_class_diff = embedding_diff[class_indices]

        for s in range(len(class_indices)):
            tensor_class_orig = torch.tensor(embedding_class_orig)
            tensor_class_diff = torch.tensor(embedding_class_diff)
            if cuda_device != -1:
                tensor_class_orig = tensor_class_orig.cuda(cuda_device)
                tensor_class_diff = tensor_class_diff.cuda(cuda_device)
            distances_embeddings_class = 1-torch.nn.functional.cosine_similarity(tensor_class_orig, tensor_class_diff.roll(s, dims=0))
            if cuda_device != -1:
                distances_embeddings_class = distances_embeddings_class.cpu()
            hist, _ = np.histogram(distances_embeddings_class, bins=n_bins, range=(min_distance, max_distance))
            hist_class += hist
    hist_class /= np.sum(hist_class)

    # all versus all distances
    hist_all = np.zeros(n_bins)
    for s in tqdm(range(len(class_labels))):
        tensor_orig = torch.tensor(embedding_orig)
        tensor_diff = torch.tensor(embedding_diff)
        if cuda_device != -1:
            tensor_orig = tensor_orig.cuda(cuda_device)
            tensor_diff = tensor_diff.cuda(cuda_device)
        distances_embeddings_all = 1-torch.nn.functional.cosine_similarity(tensor_orig, tensor_diff.roll(s, dims=0))
        if cuda_device != -1:
            distances_embeddings_all = distances_embeddings_all.cpu()
        hist, _ = np.histogram(distances_embeddings_all, bins=n_bins, range=(min_distance, max_distance))
        hist_all += hist
    hist_all /= np.sum(hist_all)

    seaborn.set_style("darkgrid")
    if ax is None:
        fig,ax = plt.subplots()

    ax.bar(bin_edges[:-1], hist_related, width=1/n_bins, label="related")
    ax.bar(bin_edges[:-1], hist_class, width=1/n_bins, label="in class", alpha=0.6)
    ax.bar(bin_edges[:-1], hist_all, width=1/n_bins, label="all vs all", alpha=0.5)
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("relative frequency")
    if dataset_orig and dataset_diff:
        ax.set_title(f"Distances for {model_short_name}\nbetween {dataset_orig} and {dataset_diff}")
    else:
        ax.set_title(f"Distances for {model_short_name}")
    ax.legend()

    if save_plot_path is not None:
        plt.savefig(save_plot_path, bbox_inches="tight")


def compute_distance_to_diff(root_model, model_short_name, dataset_orig, dataset_diff, cuda_device, aug_dict=None, plot_hist=False, save_hist=False, n_bins=100, ax=None):
    
    embedding_orig, embedding_diff, class_labels = load_compute_orig_diff_embeddings(root_model, dataset_orig, dataset_diff, cuda_device, aug_dict)

    df_dist = compute_distance_mean_std(embedding_orig, embedding_diff, class_labels, model_short_name, cuda_device)

    if plot_hist or save_hist:
        save_plot_path = None
        if save_hist:
            _, path_embeddings_1, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_orig)
            save_plot_path = os.path.join(path_embeddings_1, f"distance_hist_between_{dataset_orig}_and_{dataset_diff}.png")
        plot_distance_histograms(df_dist, embedding_orig, embedding_diff, class_labels, model_short_name, cuda_device, n_bins, ax, dataset_orig, dataset_diff, save_plot_path)

    return df_dist


def compute_distance_to_diff_for_many(models_dict, dataset_orig, dataset_diff, cuda_device, aug_dict=None, plot_hist=False, save_hist=False, n_bins=100, ax_dict=None):

    dist_list = []
    for m in models_dict:
        root_model, _ = models_dict[m]

        if ax_dict is None:
            ax = None
        else:
            ax = ax_dict[m]

        df_dist = compute_distance_to_diff(root_model, m, dataset_orig, dataset_diff, cuda_device, aug_dict, plot_hist, save_hist, n_bins, ax)
        dist_list.append(df_dist)

    df_distances = pd.concat(dist_list, axis=1)

    return df_distances


def save_dist_to_csv_file(root_model, df_dist, dataset_1, dataset_2):
    _, path_embeddings_1, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1)

    df_dist.columns = [f"{dataset_1}_dist_to_{dataset_2}"]

    df_dist.to_csv(os.path.join(path_embeddings_1, f"{dataset_1}_dist_to_{dataset_2}.csv"))


def save_dist_to_csv_file_for_many(models_dict, df_distances, dataset_1, dataset_2):
    for m in models_dict:
        root_model, _ = models_dict[m]
        df_dist = df_distances[[m]]

        save_dist_to_csv_file(root_model, df_dist, dataset_1, dataset_2)
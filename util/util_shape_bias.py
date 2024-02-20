import os
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn
from torchvision import transforms, datasets

import util.util_validation as ut_val
from tqdm import tqdm

from networks.resnet_big import model_dict
from util.util_logging import open_csv_file
from util.util import TwoCropTransform
from util.util_diff import SameTwoApply
from util.util_diff import ShufflePatches, TwoDifferentApply


# Cue Conflict Shape Bias
def evaluate_cue_conflict_dataset(root_model, dataset_classifier, dataset_cue_conflict, cuda_device, exclude_original_dict=dict()):
    path_folder, _ = ut_val.get_paths_from_model_checkpoint(root_model)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))
    if dataset_classifier is None:
        path_classifier = ut_val.get_path_classifier(root_model, "", params)
    else:
        path_classifier = ut_val.get_path_classifier(root_model, dataset_classifier, params)

    _, root_dataset_conflict = ut_val.get_root_dataset(dataset_cue_conflict)

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


def evaluate_cue_conflict_dataset_for_many(models_dict, dataset_cue_conflict, cuda_device, exclude_original_dict=dict()):
    print("Get predictions for shape texture cue conflict dataset")
    pred_dict = dict()

    for m in models_dict:
        root_model, dataset_classifier = models_dict[m]

        df_pred, classes = evaluate_cue_conflict_dataset(root_model, dataset_classifier, dataset_cue_conflict, cuda_device, exclude_original_dict)
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


def cue_conflict_shape_bias_metric(root_model, dataset_classifier, model_short_name, dataset_cue_conflict, cuda_device):
    df_pred, classes = evaluate_cue_conflict_dataset(root_model, dataset_classifier, dataset_cue_conflict, cuda_device)
    df_bias, df_class_bias = compute_cue_conflict_shape_bias_metric(df_pred, classes, model_short_name)

    return df_bias, df_class_bias


def cue_conflict_shape_bias_metric_for_many(models_dict, dataset_cue_conflict, cuda_device):
    pred_dict, classes = evaluate_cue_conflict_dataset_for_many(models_dict, dataset_cue_conflict, cuda_device)
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
def compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, separate_color=False, apply_ColorJitter=[], patch_size=30):
    path_folder, path_embeddings_orig, path_embeddings_diff, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1=dataset_orig, dataset_2=dataset_shape)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    _, root_dataset_orig = ut_val.get_root_dataset(dataset=dataset_orig)
    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])

    # load the pre computed original embeddings
    with open(os.path.join(path_embeddings_orig, "embedding_test"), 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        embedding_orig = entry['data']
        class_labels = entry['labels']

        # load the image names (or compute and store them for the future)
        if "images" in entry:
            images_orig = entry["images"]
        else:
            print(f"{dataset_orig}: Image names not found in precomputed embeddings! Recompute them and store them wich image names!")
            root_dataset_orig_train, root_dataset_orig = ut_val.get_root_dataset(dataset=dataset_orig)

            train_loader, val_loader = ut_val.set_dataloader(dataset_orig, params, root_dataset_orig_train, root_dataset_orig)
            model = ut_val.set_model(root_model, params, cuda_device)

            _, _, _, embedding_orig, class_labels, images_orig = ut_val.compute_and_save_embeddings(model, train_loader, val_loader, path_embeddings_orig, params, cuda_device)
        # use image names to sort the feature embeddings and class labels so that they match up with the other embeddings
        embedding_orig = embedding_orig[np.argsort(images_orig)]
        class_labels = class_labels[np.argsort(images_orig)]

    if "shape" in apply_ColorJitter or not os.path.isfile(os.path.join(path_embeddings_diff, "embedding_test")):
        # if no precomputed embeddings exist or color jitter should be applied to the shape data then compute the embeddings
        _, root_dataset_shape = ut_val.get_root_dataset(dataset=dataset_shape)

        if "shape" in apply_ColorJitter:
            shape_transform = transforms.Compose([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), transforms.ToTensor(), normalize])
        else:
            shape_transform = transforms.Compose([transforms.ToTensor(), normalize])

        shape_dataset = datasets.ImageFolder(root=root_dataset_shape, transform=shape_transform)
        shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=params['batch_size'],
                                                        shuffle=False, num_workers=params["num_workers"], pin_memory=True)

        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_shape, _ = ut_val.compute_embedding(model, shape_dataloader, params, cuda_device)
        images_shape = [img[0].replace(shape_dataset.root, '') for img in shape_dataset.imgs]
        # use image names to sort the feature embeddings so that they match up with the other embeddings
        embedding_shape = embedding_shape[np.argsort(images_shape)]
    else:
        # load pre computed embeddings of the shape dataset if they exit
        with open(os.path.join(path_embeddings_diff, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_shape = entry['data']

        # load the image names (or compute and store them for the future)
        if "images" in entry:
            images_shape = entry["images"]
        else:
            print(f"{dataset_shape}: Image names not found in precomputed embeddings! Recompute them and store them wich image names!")
            root_dataset_shape_train, root_dataset_shape = ut_val.get_root_dataset(dataset=dataset_shape)

            train_loader, val_loader = ut_val.set_dataloader(dataset_shape, params, root_dataset_shape_train, root_dataset_shape)
            model = ut_val.set_model(root_model, params, cuda_device)

            _, _, _, embedding_shape, _, images_shape = ut_val.compute_and_save_embeddings(model, train_loader, val_loader, path_embeddings_diff, params, cuda_device)
        # use image names to sort the feature embeddings so that they match up with the other embeddings
        embedding_shape = embedding_shape[np.argsort(images_shape)]

    if separate_color:
        # if the color should be considered as a component independent of texture than compute the "color embedding" from pixel wise shuffled original images
        # at the same time compute the "texture embeddings" (apply color jitter if given)
        if "texture" in apply_ColorJitter:
            patchShuffle_transform = transforms.Compose([TwoDifferentApply(transform_orig=transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)),
                                                         SameTwoApply(transforms.ToTensor()),
                                                         TwoDifferentApply(transform_orig=ShufflePatches(patch_size=patch_size), transform_diff=ShufflePatches(patch_size=1))])
        else:
            patchShuffle_transform = transforms.Compose([SameTwoApply(transforms.ToTensor()),
                                                         TwoDifferentApply(transform_orig=ShufflePatches(patch_size=patch_size), transform_diff=ShufflePatches(patch_size=1))])
        textureColor_dataset = datasets.ImageFolder(root=root_dataset_orig,
                                                    transform=TwoCropTransform(normalize, patchShuffle_transform))
        textureColor_dataloader = torch.utils.data.DataLoader(textureColor_dataset, batch_size=params['batch_size'],
                                                              shuffle=False, num_workers=params["num_workers"], pin_memory=True)
        
        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_texture, embedding_color, _ = ut_val.compute_diff_embeddings(model, textureColor_dataloader, params, cuda_device)
        # use image names to sort the feature embeddings so that they match up with the other embeddings
        images_textureColor_argsort = np.argsort([img[0].replace(textureColor_dataset.root, '') for img in textureColor_dataset.imgs])
        embedding_texture = embedding_texture[images_textureColor_argsort]
        embedding_color = embedding_color [images_textureColor_argsort]


        return embedding_orig, embedding_shape, embedding_texture, embedding_color, class_labels
    else:
        # compute the "texture embedding" as patches shuffled images with given patch_size from the original images (apply color jitter if given)
        if "texture" in apply_ColorJitter:
            texture_transform = transforms.Compose([transforms.ToTensor(), ShufflePatches(patch_size=patch_size), normalize])
        else:
            texture_transform = transforms.Compose([transforms.ToTensor(), ShufflePatches(patch_size=patch_size), normalize])

        texture_dataset = datasets.ImageFolder(root=root_dataset_orig, transform=texture_transform)
        texture_dataloader = torch.utils.data.DataLoader(texture_dataset, batch_size=params['batch_size'],
                                                                shuffle=False, num_workers=params["num_workers"], pin_memory=True)

        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_texture, _ = ut_val.compute_embedding(model, texture_dataloader, params, cuda_device)
        images_texture = [img[0].replace(texture_dataset.root, '') for img in texture_dataset.imgs]
        embedding_texture = embedding_texture[np.argsort(images_texture)]


        return embedding_orig, embedding_shape, embedding_texture, class_labels


def compute_cue_conflict_embeddings(root_model, dataset_cue_conflict, cuda_device):
    path_folder, _ = ut_val.get_paths_from_model_checkpoint(root_model)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    _, root_dataset_conflict = ut_val.get_root_dataset(dataset_cue_conflict)

    # dataloader and model
    normalize = transforms.Normalize(mean=params['mean'], std=params['std'])
    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    conflict_dataset = ut_val.shapeTextureConflictDataset(root_dataset_conflict, val_transform)

    classes = conflict_dataset.classes
    shape_texture_name_list = [path.replace(".jpg", '').split('/')[-3:] for path in conflict_dataset.paths]
    # list of shape and texture image names in the form ("shapeClassLabel/shapeImageName", "textureClassLabel/textureImageName")
    shapeName_textureName_list = [(s+'/'+n.split('_stylized_')[0], t+'/'+n.split('_stylized_')[1]) for s,t,n in shape_texture_name_list]

    conflict_dataloader = torch.utils.data.DataLoader(conflict_dataset, batch_size=params['batch_size'],
                                                      shuffle=False, num_workers=16, pin_memory=True)
    
    model = ut_val.set_model(root_model, params, cuda_device)

    # compute embeddings
    _, embedding_size = model_dict[params['model']]

    model.eval()

    embedding = np.array([])
    shape_labels = np.array([], dtype=int)
    texture_labels = np.array([], dtype=int)
    with torch.no_grad():
        for images, labels in tqdm(conflict_dataloader):
            images = images.cuda(device=cuda_device, non_blocking=True)

            features = model.encoder(images)

            embedding = np.append(embedding, features.cpu().numpy())
            shape_labels = np.append(shape_labels, labels[0].numpy())
            texture_labels = np.append(texture_labels, labels[1].numpy())

    embedding = embedding.reshape(-1, embedding_size)

    return embedding, shape_labels, texture_labels, classes, shapeName_textureName_list


def compute_dim_correlation_coefficients(embedding_A, embedding_B):
    A = torch.tensor(embedding_A)
    B = torch.tensor(embedding_B)

    A_dm = A - A.mean(dim=0)
    B_dm = B - B.mean(dim=0)

    correlation = ((A_dm * B_dm).sum(dim=0) / ((A_dm * A_dm).sum(dim=0) * (B_dm * B_dm).sum(dim=0)).sqrt())

    # if there is a neuron that has no variation in its output the computed correlation coefficient will be NaN
    # set these NaN values to zero because a constant value can be seen to be uncorrelated to any random variable
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


def compute_corelation_coefficient_shape_bias_metric(shape_pair, texture_pair, model_short_name, color_pair=None):
    corr_coef_shape = compute_dim_correlation_coefficients(shape_pair[0], shape_pair[1])
    corr_coef_texture = compute_dim_correlation_coefficients(texture_pair[0], texture_pair[1])
    if color_pair is not None:
        corr_coef_color = compute_dim_correlation_coefficients(color_pair[0], color_pair[1])

        dims = estimate_dims([corr_coef_shape, corr_coef_texture, corr_coef_color])

        df_dims = pd.DataFrame.from_dict({model_short_name: dims}, orient="index", columns=["shape_dims", "texture_dims", "color_dims", "remaining_dims"])
    else:
        dims = estimate_dims([corr_coef_shape, corr_coef_texture])

        df_dims = pd.DataFrame.from_dict({model_short_name: dims}, orient="index", columns=["shape_dims", "texture_dims", "remaining_dims"])

    return df_dims


def corelation_coefficient_shape_bias_metric(root_model, model_short_name, dataset_orig, dataset_shape, cuda_device, separate_color=False, apply_ColorJitter=[], patch_size=30):
    if separate_color:
        embedding_orig, embedding_shape, embedding_texture, embedding_color, _ = compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, separate_color, apply_ColorJitter, patch_size)

        df_dims = compute_corelation_coefficient_shape_bias_metric(shape_pair=(embedding_orig,embedding_shape), texture_pair=(embedding_orig,embedding_texture),
                                                                   color_pair=(embedding_orig,embedding_color), model_short_name=model_short_name)
    else:
        embedding_orig, embedding_shape, embedding_texture, _ = compute_load_orig_shape_texture_embeddings(root_model, dataset_orig, dataset_shape, cuda_device, separate_color, apply_ColorJitter, patch_size)

        df_dims = compute_corelation_coefficient_shape_bias_metric(shape_pair=(embedding_orig,embedding_shape), texture_pair=(embedding_orig,embedding_texture),
                                                                   model_short_name=model_short_name)

    return df_dims


def corelation_coefficient_shape_bias_metric_for_many(models_dict, dataset_orig, dataset_shape, cuda_device, separate_color=False, apply_ColorJitter=[], patch_size=30):
    dims_list = []
    for m in models_dict:
        root_model, _ = models_dict[m]

        df_dims = corelation_coefficient_shape_bias_metric(root_model, m, dataset_orig, dataset_shape, cuda_device, separate_color, apply_ColorJitter, patch_size)
        dims_list.append(df_dims)

    return pd.concat(dims_list, axis=0)


def corelation_coefficient_shape_bias_metric_from_cue_conflict_dataset(root_model, model_short_name, dataset_cue_conflict, cuda_device):
    embedding, _, _, _, shapeName_textureName_list = compute_cue_conflict_embeddings(root_model, dataset_cue_conflict, cuda_device)

    # determent the shape and texture pairs in the cue conflict embedding
    shape_pairs = []
    shape_array = np.array([sN for sN,_ in shapeName_textureName_list])
    for sN in set(shape_array):
        shape_indices = np.where(shape_array == sN)[0]
        shape_pairs.append(shape_indices)

    shape_pair_A = np.concatenate([np.tile(shape_pairs[i], reps=len(shape_pairs[i])-1) for i in range(len(shape_pairs))])
    shape_pair_B = np.concatenate([np.concatenate([np.roll(shape_pairs[i], shift=j) for j in range(1,len(shape_pairs[i]))]) for i in range(len(shape_pairs))])

    texture_pairs = []
    texture_array = np.array([tN for _,tN in shapeName_textureName_list])
    for tN in set(texture_array):
        texture_indices = np.where(texture_array == tN)[0]
        texture_pairs.append(texture_indices)

    texture_pair_A = np.concatenate([np.tile(texture_pairs[i], reps=len(texture_pairs[i])-1) for i in range(len(texture_pairs))])
    texture_pair_B = np.concatenate([np.concatenate([np.roll(texture_pairs[i], shift=j) for j in range(1,len(texture_pairs[i]))]) for i in range(len(texture_pairs))])

    # estimate the shape and texture dimensions
    df_dims = compute_corelation_coefficient_shape_bias_metric(shape_pair=(embedding[shape_pair_A],embedding[shape_pair_B]), texture_pair=(embedding[texture_pair_A],embedding[texture_pair_B]),
                                                               model_short_name=model_short_name)
    
    return df_dims


def corelation_coefficient_shape_bias_metric_from_cue_conflict_dataset_for_many(models_dict, dataset_cue_conflict, cuda_device):
    dims_list = []
    for m in models_dict:
        root_model, _ = models_dict[m]

        df_dims = corelation_coefficient_shape_bias_metric_from_cue_conflict_dataset(root_model, m, dataset_cue_conflict, cuda_device)
        dims_list.append(df_dims)

    return pd.concat(dims_list, axis=0)


def save_correlation_coefficient_shape_bias_to_csv_file(root_model, df_dims, dataset_dict, apply_ColorJitter=[], patch_size=30):
    path_folder, epoch = ut_val.get_paths_from_model_checkpoint(root_model)

    # determent the datasets used
    if "cue_conflict" in dataset_dict:
        assert len(dataset_dict) == 1
        datasets_corr_coef = dataset_dict["cue_conflict"]
    else:
        assert "shape" in dataset_dict and "texture" in dataset_dict
        dataset_shape = dataset_dict["shape"] + ("CJitter" if "shape" in apply_ColorJitter else "") + "Shape"
        dataset_texture = dataset_dict["texture"] + f"PatchSize{patch_size}" + ("CJitter" if "texture" in apply_ColorJitter else "") + "Texture"
        datasets_corr_coef = dataset_shape + "_" + dataset_texture

        if "color" in dataset_dict:
            dataset_color = dataset_dict["color"] + "PixelShuffledColor"
            datasets_corr_coef += "_" + dataset_color

    path_corr_coef_shape_bias = os.path.join(path_folder, f"val_{epoch}", "shapeBiasMetrics", "CorrelationCoefficient", datasets_corr_coef)
    os.makedirs(path_corr_coef_shape_bias, exist_ok=True)

    df_dims.index = [datasets_corr_coef]
    df_dims.to_csv(os.path.join(path_corr_coef_shape_bias, "pred_dims.csv"))


def save_correlation_coefficient_shape_bias_to_csv_file_for_many(models_dict, df_dimensions, dataset_dict, apply_ColorJitter=[], patch_size=30):
    for m in models_dict:
        root_model, _ = models_dict[m]
        df_dims = df_dimensions.loc[[m]]

        save_correlation_coefficient_shape_bias_to_csv_file(root_model, df_dims, dataset_dict, apply_ColorJitter, patch_size)


# Feature Embedding Distances
def load_compute_orig_diff_embeddings(root_model, dataset_orig, dataset_diff, cuda_device, aug_dict=None):
    path_folder, path_embeddings_orig, path_embeddings_diff, _ = ut_val.get_paths_from_model_checkpoint(root_model, dataset_1=dataset_orig, dataset_2=dataset_diff)
    params = open_csv_file(os.path.join(path_folder, "params.csv"))

    if aug_dict:
        # compute original and diffused embeddings with data augmentations from aug_dict
        root_dataset_train_orig, root_dataset_test_orig = ut_val.get_root_dataset(dataset=dataset_orig)
        root_dataset_train_diff, root_dataset_test_diff = ut_val.get_root_dataset(dataset=dataset_diff)

        # original and diffused images get loaded pair wise and therefor is this match also in the computed feature embeddings
        _, val_loader = ut_val.set_dataloader("", params, root_dataset_train_orig, root_dataset_test_orig, root_dataset_train_diff, root_dataset_test_diff, aug_dict)

        model = ut_val.set_model(root_model, params, cuda_device)

        embedding_orig, embedding_diff, class_labels = ut_val.compute_diff_embeddings(model, val_loader, params, cuda_device)
    else:
        # load pre computed original and diffused embeddings
        with open(os.path.join(path_embeddings_orig, "embedding_test"), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            embedding_orig = entry['data']
            class_labels = entry['labels']

            # load the image names (or compute and store them for the future)
            if "images" in entry:
                images_orig = entry["images"]
            else:
                print(f"{dataset_orig}: Image names not found in precomputed embeddings! Recompute them and store them wich image names!")
                root_dataset_train_orig, root_dataset_test_orig = ut_val.get_root_dataset(dataset=dataset_orig)

                train_loader, val_loader = ut_val.set_dataloader(dataset_orig, params, root_dataset_train_orig, root_dataset_test_orig)
                model = ut_val.set_model(root_model, params, cuda_device)

                _, _, _, embedding_orig, class_labels, images_orig = ut_val.compute_and_save_embeddings(model, train_loader, val_loader, path_embeddings_orig, params, cuda_device)
            # use image names to sort the feature embeddings and class labels so that they match up with the other embeddings
            embedding_orig = embedding_orig[np.argsort(images_orig)]
            class_labels = class_labels[np.argsort(images_orig)]

        if os.path.isfile(os.path.join(path_embeddings_diff, "embedding_test")):
            with open(os.path.join(path_embeddings_diff, "embedding_test"), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                embedding_diff = entry['data']

                # load the image names (or compute and store them for the future)
                if "images" in entry:
                    images_diff = entry["images"]
                else:
                    print(f"{dataset_diff}: Image names not found in precomputed embeddings! Recompute them and store them wich image names!")
                    root_dataset_train_diff, root_dataset_test_diff = ut_val.get_root_dataset(dataset=dataset_diff)

                    train_loader, val_loader = ut_val.set_dataloader(dataset_diff, params, root_dataset_train_diff, root_dataset_test_diff)
                    model = ut_val.set_model(root_model, params, cuda_device)

                    _, _, _, embedding_diff, _, images_diff = ut_val.compute_and_save_embeddings(model, train_loader, val_loader, path_embeddings_diff, params, cuda_device)
        else:
            root_dataset_train_diff, root_dataset_test_diff = ut_val.get_root_dataset(dataset=dataset_diff)

            _, val_loader = ut_val.set_dataloader(dataset_diff, params, root_dataset_train_diff, root_dataset_test_diff)
            model = ut_val.set_model(root_model, params, cuda_device)

            embedding_diff, _ = ut_val.compute_embedding(model, val_loader, params, cuda_device)
            images_diff = [img[0].replace(val_loader.dataset.root, '') for img in val_loader.dataset.imgs]
        # use image names to sort the feature embeddings so that they match up with the other embeddings
        embedding_diff = embedding_diff[np.argsort(images_diff)]

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
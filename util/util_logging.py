import os
import glob
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import ConfusionMatrixDisplay
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

seaborn.set_theme(style="darkgrid")


def try_eval(val):
    try:
        return eval(val)
    except:
        return val


# Convert teonsorboard logs into plots
def open_tensorboard(path):
    """
    Gets the scalar data from a tensorboard log in a dataFrame

    Parameters
    ---------
    path: str
        path to the tenorboard log file (event...) in the form <path>/tensorboard/event...
    """
    tb_path = glob.glob(os.path.join(path, "tensorboard", 'event*'))
    if len(tb_path) < 1:
        print(f"No tesorboard-log-file found at {os.path.join(path, 'tensorboard')}!")
        return
    tb_path = tb_path[0]

    event_acc = EventAccumulator(tb_path)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']

    log_dict = dict()
    for tag in tags:
        scalar_list = event_acc.Scalars(tag)
        if not 'steps' in log_dict:
            steps = [scalar.step for scalar in scalar_list]
            log_dict['step'] = steps
            start_time = scalar_list[0].wall_time
            rel_time = [(scalar.wall_time-start_time)/60 for scalar in scalar_list]
            log_dict['rel. time [min]'] = rel_time
        values = [scalar.value for scalar in scalar_list]
        log_dict[tag] = values

    return pd.DataFrame.from_dict(log_dict)


def create_training_plots(path):
    """
    Creates plots of learning_rate and loss from a tensorboard log

    Parameters
    ---------
    path: str
        path to the tenorboard log file (even*) in the form <path>/tensorboard/event*.
        The same path is used for saving the plots.
    """
    df_log = open_tensorboard(path)

    df_log.plot.line(x='step', y='learning_rate', title="Learning rate scheduling"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "learning_rate.png"))

    df_log.plot.line(x='step', y='loss', title="Loss"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "loss.png"))


def create_classifier_training_plots(path_class):
    """
    Creates plots of train_loss and val_acc from a tensorboard log

    Parameters
    ---------
    path: str
        path to the tenorboard log file (even*) in the form <path>/tensorboard/event*.
        The same path is used for saving the plots.
    """
    df_log = open_tensorboard(path_class)

    df_log.plot.line(x='step', y='train_loss', title="Training Loss"
                    ).get_figure().savefig(os.path.join(path_class, "tensorboard", "train_loss.png"))
    
    df_log.plot.line(x='step', y='val_acc', title="Validation top-1 accuracy"
                    ).get_figure().savefig(os.path.join(path_class, "tensorboard", "val_top1.png"))

def create_crossentropy_plots(path):
    """
    Creates plots of learning_rate, train_loss and train_acc, val_acc from a tensorboard log

    Parameters
    --------
    path: str
        path to the tenorboard log file (even*) in the form <path>/tensorboard/event*.
        The same path is used for saving the plots.
    """
    df_log = open_tensorboard(path)

    df_log.plot.line(x='step', y='learning_rate', title="Learning rate scheduling"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "learning_rate.png"))

    df_log.plot.line(x='step', y='train_loss', title="Training Loss"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "loss.png"))
    
    df_log.plot.line(x='step', y='train_acc', title="Training top-1 accuracy"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "train_top1.png"))
    
    df_log.plot.line(x='step', y='val_acc', title="Validation top-1 accuracy"
                    ).get_figure().savefig(os.path.join(path, "tensorboard", "val_top1.png"))


# csv files for training parameters
def create_csv_file_training(opt, csv_file, other_dict:dict=dict()):
    params = vars(opt)
    for k in other_dict:
        params[k] = other_dict[k]

    with open(csv_file, 'w') as f:
        w = csv.DictWriter(f, vars(opt).keys())
        w.writeheader()
        w.writerow(vars(opt))


def create_csv_file_with_best_acc(opt, best_acc, csv_file):
    other_dict = {"best_acc": best_acc}
    create_csv_file_training(opt, other_dict, csv_file)


def create_classifier_csv_file(opt, best_acc, csv_file):
    params = vars(opt)
    params["best_acc"] = best_acc

    with open(csv_file, 'w') as f:
        w = csv.DictWriter(f, vars(opt).keys())
        w.writeheader()
        w.writerow(vars(opt))


def open_csv_file(csv_file):
    params = dict()

    with open(csv_file, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            for key in row:
                value = try_eval(row[key])
                params[key] = None if value == '' else value

    return params


# mark down files for nicer visualization of training parameters and validation results
def md_table_from_params(keys, value_dict):
    table_head = "|"
    table_middle = "|"
    table_values = "|"
    for k in keys:
        if k in value_dict:
            table_head += f"{k}|"
            table_middle += "--|"
            table_values += f"{value_dict[k]}|"
    table_head += "\n"
    table_middle += "\n"
    table_values += "\n\n"

    return [table_head, table_middle, table_values]


def md_table_from_dict(value_dict, keys=None):
    df_table = pd.DataFrame.from_dict(value_dict)
    if keys is not None:
        df_table = df_table[keys]

    table_str = df_table.to_markdown(index=False)
    table_lines = [t_line + "\n" for t_line in table_str.split("\n")]
    table_lines[-1] += "\n"

    return table_lines

# run.md file for the trained models
def create_run_md(path_folder):
    params_dataset = ["dataset", "mean", "std", "size", "num_classes", "aug", "resizedCrop", "horizontalFlip", "colorJitter", "grayscale", "diff_p"]
    params_training = ["model", "method", "related_factor", "temp", "batch_size", "batch_size_val", "epochs", "learning_rate", "momentum", "lr_decay_epochs", "lr_decay_rate", "weight_decay", "cosine", "warm"]
    params_other = ["tag", "trial", "save_freq", "print_freq", "num_workers", "num_workers_val"]

    params_csv = open_csv_file(os.path.join(path_folder, "params.csv"))

    # set the title
    if "method" in params_csv:
        title = params_csv["method"]
    else:
        title = "CE"
    title += f" {params_csv['dataset']} {params_csv['tag']}"

    lines = [f"# {title}\n\n",
             "## Training Parameters\n\n"]
    
    # dataset parameters
    lines.append("#### Dataset\n\n")
    lines.extend(md_table_from_params(keys=params_dataset, value_dict=params_csv))

    # training parameters
    lines.append("#### Training\n\n")
    lines.extend(md_table_from_params(keys=params_training, value_dict=params_csv))

    # other parameters
    lines.append("#### Other\n\n")
    lines.extend(md_table_from_params(keys=params_other, value_dict=params_csv))

    # tensorboard path
    lines.extend(["#### Tensorboard\n\n",
                  "```\n",
                  f"tensorboard --logdir={params_csv['tb_folder']}\n"
                  "```\n\n"])
    
    # model checkpoint path
    lines.extend(["#### Model checkpoints\n\n",
                  "```\n",
                  f"{params_csv['save_folder']}\n",
                  "```\n\n"])
    
    # Training plots
    tb_plots = [plot.split('/')[-1] for plot in glob.glob(os.path.join(path_folder, "tensorboard", "*.png"))]
    if "learning_rate.png" in tb_plots and "loss.png" in tb_plots:
        lines.extend(["## Training\n\n",
                      "Leaning rate | Loss\n",
                      ":--:|:--:\n",
                      "![plot of learning rate scheduling](./tensorboard/learning_rate.png)|![plot of training loss](./tensorboard/loss.png)\n\n"])
    if "train_top1.png" in tb_plots and "val_top1.png" in tb_plots:
        lines.append("## Training Validation\n\n")
        
        df_tb = open_tensorboard(path_folder)
        if df_tb is not None:
            lines.append(f"**best top-1 validation accuracy: {df_tb['val_acc'].max():.2f}**\n")

        lines.extend(["top-1 training accuracy | top-1 validation accuracy\n",
                      ":--:|:--:\n",
                      "![plot of classifier top-1 training acc](./tensorboard/train_top1.png)|![plot of classifier top-1 validation acc](./tensorboard/val_top1.png)\n\n"])

    # plots for additionally trained classifier (with same/similar conditions as encoder)
    classifier_plots = glob.glob(os.path.join(path_folder, "val_*", "classifier", "*", "tensorboard", "*.png"))
    if len(classifier_plots) > 0:
        lines.append("## Classifier\n\n")

        epochs_classifier = [e.replace(' ', '') for e in sorted(set([f"{plot_path.split('/')[-5].replace('val_', ''):>3}"for plot_path in classifier_plots]))]

        for e in epochs_classifier:
            c_plot_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "classifier", "*", "tensorboard", f"*.png"))

            if len(c_plot_paths) == 2:
                lines.append(f"### Epoch {e}\n\n")

                c_plot_paths_split = c_plot_paths[0].split('/')
                df_tb = open_tensorboard(os.path.join(*c_plot_paths_split[:-2]))
                if df_tb is not None:
                    lines.append(f"**best top-1 validation accuracy: {df_tb['val_acc'].max():.2f}**\n")

                lines.extend(["top-1 training accuracy | top-1 validation accuracy\n",
                              ":--:|:--:\n",
                              f"![plot of classifier training loss]({os.path.join(*c_plot_paths_split[-5:-1], 'train_loss.png')})|",
                              f"![plot of classifier top-1 validation acc]({os.path.join(*c_plot_paths_split[-5:-1], 'val_top1.png')})\n\n"])
        
    # write the markdown file
    with open(os.path.join(path_folder, "run.md"), "w") as f:
        f.writelines(lines)


# cm.md for classification accuracies and plots of confusion matrices
def save_confusion_matrix(C, classes, path, title="Confusion Matrix"):
    # save the confusion matrix as csv file
    df_cm = pd.DataFrame(C)
    df_cm.to_csv(path.replace(".png", ".csv"), index=False)

    # save plot of the confusion matrix
    seaborn.set_theme(style="ticks")

    disp = ConfusionMatrixDisplay(C, display_labels=classes)
    disp.plot(xticks_rotation=(45 if len(classes)<=10 else 90))
    for labels in disp.text_.ravel():
        labels.set_fontsize(10)
    disp.ax_.set_title(title)
    disp.figure_.tight_layout(pad=0.5)

    plt.savefig(path)


def load_confusion_matrix(path):
    df_cm = pd.read_csv(path)
    C = df_cm.to_numpy()

    return C


def compute_accuracies_form_cm(C):
    c_lens = C.sum(1)
    all_len = C.sum()

    # accuracy
    acc = 0.0
    for i in range(len(c_lens)):
        acc += C[i,i]
    acc /= all_len
    # balanced accuracy
    acc_b = 0.0
    for i, n in enumerate(c_lens):
        acc_b += C[i,i] / n
    acc_b /= len(c_lens)

    return acc, acc_b


def create_cm_md(path_folder):
    params_csv = open_csv_file(os.path.join(path_folder, "params.csv"))

    # set the title
    if "method" in params_csv:
        title = f"Classification validation: {params_csv['method']}"
    else:
        title = "Classification validation: CE"
    title += f" {params_csv['dataset']} {params_csv['tag']}"

    lines = [f"# {title}\n\n",
             "## Accuracies\n\n"]
    
    # collect all accuracies anc cm plots
    cm_csv_paths = glob.glob(os.path.join(path_folder, "val_*", "*", "cm", "cm_*.csv"))
    datasets_cm = sorted(set([cm_path.split('/')[-3] for cm_path in cm_csv_paths]))
    epochs_cm = [e.replace(' ', '') for e in sorted(set([f"{cm_path.split('/')[-4].replace('val_', ''):>3}"for cm_path in cm_csv_paths]))]

    cm_plot_dict = dict()
    for e in epochs_cm:
        cm_plot_dict[e] = []
    acc_dict_train = {"epoch": epochs_cm}
    acc_dict_val = {"epoch": epochs_cm}
    for dset in datasets_cm:
        acc_dict_train[dset] = []
        acc_dict_val[dset] = []
        for e in epochs_cm:
            cm_plot_paths = glob.glob(os.path.join(path_folder, f"val_{e}", dset, "cm", f"cm_*_epoch_{e}.png"))
            cm_path_train = glob.glob(os.path.join(path_folder, f"val_{e}", dset, "cm", f"cm_train_epoch_{e}.csv"))
            cm_path_val = glob.glob(os.path.join(path_folder, f"val_{e}", dset, "cm", f"cm_val_epoch_{e}.csv"))
            if 0 < len(cm_plot_paths) <= 2 or len(cm_path_train) == 1 or len(cm_path_val) == 1:
                cm_plot_dict[e].append(f"#### Dataset: {dset}\n\n")


            if len(cm_path_train) == 1:
                C_train = load_confusion_matrix(cm_path_train[0])
                acc_train, acc_b_train = compute_accuracies_form_cm(C_train)

                acc_dict_train[dset].append(f"{acc_train*100:.2f} ({acc_b_train*100:.2f})")
                cm_plot_dict[e].append(f"- **Training Data: Accuracy: {acc_train*100:.2f}, Class Balanced Accuracy: {acc_b_train*100:.2f}**\n")
            else:
                acc_dict_train[dset].append("")

            if len(cm_path_val) == 1:
                C_val = load_confusion_matrix(cm_path_val[0])
                acc_val, acc_b_val = compute_accuracies_form_cm(C_val)

                acc_dict_val[dset].append(f"{acc_val*100:.2f} ({acc_b_val*100:.2f})")
                cm_plot_dict[e].append(f"- **Validation Data: Accuracy: {acc_val*100:.2f}, Class Balanced Accuracy: {acc_b_val*100:.2f}**\n")
            else:
                acc_dict_val[dset].append("")

            if len(cm_plot_paths) == 2:
                cm_plot_dict[e].extend([f"\n{dset} Trainings Data | {dset} Test Data\n",
                                        ":--:|:--:\n",
                                        f"![plot of confusion matrix trainings data]({os.path.join('.', f'val_{e}', dset, 'cm', f'cm_train_epoch_{e}.png')})|",
                                        f"![plot of confusion matrix test data]({os.path.join('.', f'val_{e}', dset, 'cm', f'cm_val_epoch_{e}.png')})\n\n"])
            elif len(cm_plot_paths) == 1:
                if os.path.isfile(os.path.join(path_folder, f'val_{e}', dset, 'cm', f'cm_train_epoch_{e}.png')):
                    cm_plot_dict[e].extend([f"\n| {dset} Trainings Data |\n",
                                            "|:--:|\n",
                                            f"|![plot of confusion matrix trainings data]({os.path.join('.', f'val_{e}', dset, 'cm', f'cm_train_epoch_{e}.png')})|\n\n"])
                elif os.path.isfile(os.path.join(path_folder, f'val_{e}', dset, 'cm', f'cm_val_epoch_{e}.png')):
                    cm_plot_dict[e].extend([f"\n| {dset} Test Data |\n",
                                            "|:--:|\n",
                                            f"|![plot of confusion matrix test data]({os.path.join('.', f'val_{e}', dset, 'cm', f'cm_val_epoch_{e}.png')})|\n\n"])

    # tables for accuracies
    if len(acc_dict_train["epoch"]) > 0:
        lines.append("**training data accuracies: (class balanced accuracies in brackets)**\n")
        lines.extend(md_table_from_dict(acc_dict_train))
    if len(acc_dict_val["epoch"]) > 0:
        lines.append("**validation data accuracies: (class balanced accuracies in brackets)**\n")
        lines.extend(md_table_from_dict(acc_dict_val))

    # confusion matrix plots
    for e in epochs_cm:
        if len(cm_plot_dict[e]) > 0:
            lines.append(f"### Epoch {e}\n\n")
            lines.extend(cm_plot_dict[e])

    # # write the markdown file
    if len(lines) > 2:
        with open(os.path.join(path_folder, "cm.md"), "w") as f:
            f.writelines(lines)
        

# tSNE.md for t-SNE plots
def create_tsne_md(path_folder):
    params_csv = open_csv_file(os.path.join(path_folder, "params.csv"))

    # set the title
    if "method" in params_csv:
        title = f"t-SNE Plots: {params_csv['method']}"
    else:
        title = "t-SNE Plots: CE"
    title += f" {params_csv['dataset']} {params_csv['tag']}"

    lines = [f"# {title}\n\n"]

    # collect epochs and datasets
    tsne_paths = glob.glob(os.path.join(path_folder, "val_*", "*", "embeddings", "tSNE_epoch_*.png"))
    datasets_tsne = sorted(set([tsne_path.split('/')[-3] for tsne_path in tsne_paths]))
    epochs_tsne = [e.replace(' ', '') for e in sorted(set([f"{tsne_path.split('/')[-4].replace('val_', ''):>3}"for tsne_path in tsne_paths]))]

    # add plots to lines  
    for e in epochs_tsne:
        plot_lines = []
        for dset in datasets_tsne:
            tsne_plot_paths = glob.glob(os.path.join(path_folder, f"val_{e}", dset, "embeddings", f"tSNE_epoch_{e}_*.png"))

            if len(tsne_plot_paths) == 2:
                plot_lines.extend([f"#### Dataset: {dset}\n\n",
                                   f"{dset} Trainings Data | {dset} Test Data\n",
                                   ":--:|:--:\n",
                                   f"![t-SNE plot of epoch last training data]({os.path.join('.', f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_train.png')})",
                                   f"|![t-SNE plot of epoch last test data]({os.path.join('.', f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_test.png')})\n\n"])
            elif len(tsne_plot_paths) == 1:
                if os.path.isfile(os.path.join(path_folder, f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_train.png')):
                    plot_lines.extend([f"#### Dataset: {dset}\n\n",
                                       f"| {dset} Trainings Data |\n",
                                       "|:--:|\n",
                                       f"|![t-SNE plot of epoch last training data]({os.path.join('.', f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_train.png')})|\n\n"])
                elif os.path.isfile(os.path.join(path_folder, f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_test.png')):
                    plot_lines.extend([f"#### Dataset: {dset}\n\n",
                                       f"| {dset} Test Data |\n",
                                       "|:--:|\n",
                                       f"|![t-SNE plot of epoch last test data]({os.path.join('.', f'val_{e}', dset, 'embeddings', f'tSNE_epoch_{e}_test.png')})|\n\n"])
                
        if len(plot_lines) > 0:
            lines.append(f"### Epoch {e}\n\n")
            lines.extend(plot_lines)

    # write the markdown file
    if len(lines) > 1:
        with open(os.path.join(path_folder, "tSNE.md"), "w") as f:
            f.writelines(lines)


# distances.md tables of cosine distances of the feature embeddings
def create_distances_md(path_folder):
    params_csv = open_csv_file(os.path.join(path_folder, "params.csv"))

    # set the title
    if "method" in params_csv:
        title = f"Feature Space Distances: {params_csv['method']}"
    else:
        title = "Feature Space Distances: CE"
    title += f" {params_csv['dataset']} {params_csv['tag']}"

    lines = [f"# {title}\n\n"]

    # collect epochs and datasets
    dist_paths = glob.glob(os.path.join(path_folder, "val_*", "*", "embeddings", "*_dist_to_*.csv"))
    datasets_1_dist = sorted(set([dist_path.split('/')[-3] for dist_path in dist_paths]))
    datasets_2_dist = sorted(set([dist_path.split("_dist_to_")[-1].replace(".csv", '') for dist_path in dist_paths]))
    epochs_dist = [e.replace(' ', '') for e in sorted(set([f"{dist_path.split('/')[-4].replace('val_', ''):>3}"for dist_path in dist_paths]))]

    # # create one distance table for each epoch
    # for e in epochs_dist:
    #     dist_dict = {"Datasets": [], "Related Images": [], "Same Class": [], "All versus all": []}
    #     for dset1 in datasets_1_dist:
    #         for dset2 in datasets_2_dist:
    #             dist_path = glob.glob(os.path.join(path_folder, f"val_{e}", dset1, "embeddings", f"{dset1}_dist_to_{dset2}.csv"))

    #             if len(dist_path) == 1:
    #                 df_dist = pd.read_csv(dist_path[0])
    #                 dist_rel, dist_class, dist_all = df_dist.loc[0]

    #                 dist_dict["Datasets"].append(f"{dset1} to {dset2}")
    #                 dist_dict["Related Images"].append(f"{dist_rel:.4f} ({dist_rel/dist_all:.4f})")
    #                 dist_dict["Same Class"].append(f"{dist_class:.4f} ({dist_class/dist_all:.4f})")
    #                 dist_dict["All versus all"].append(f"{dist_all:.4f} ({dist_all/dist_all:.4f})")
        
    #     if len(dist_dict["Datasets"]) > 0:
    #         lines.extend([f"### Epoch {e}\n\n",
    #                     "**(in brackets are the distances divided by the All versus all distance)**\n"])
    #         lines.extend(md_table_from_dict(dist_dict))

    # create one distance table for each epoch
    for e in epochs_dist:
        dist_dict = {"Datasets": [], "Mean Related": [], "Std Related": [], "Mean in Class": [], "Std in Class": [],
                     "Mean All vs. all": [], "Std All vs. all": []}
        lines_plots = []
        for dset1 in datasets_1_dist:
            for dset2 in datasets_2_dist:
                dist_path = glob.glob(os.path.join(path_folder, f"val_{e}", dset1, "embeddings", f"{dset1}_dist_to_{dset2}.csv"))

                if len(dist_path) == 1:
                    df_dist = pd.read_csv(dist_path[0], index_col=0)
                    mean_rel, mean_class, mean_all = df_dist.T.loc[:,["mean_distance_related", "mean_distance_classes", "mean_distance_all_vs_all"]].iloc[0]
                    std_rel, std_class, std_all = df_dist.T.loc[:,["std_distance_related", "std_distance_classes", "std_distance_all_vs_all"]].iloc[0]

                    dist_dict["Datasets"].append(f"{dset1} to {dset2}")
                    dist_dict["Mean Related"].append(f"{mean_rel:.4f} ({mean_rel/mean_all:.4f})")
                    dist_dict["Std Related"].append(f"{std_rel:.4f}")
                    dist_dict["Mean in Class"].append(f"{mean_class:.4f} ({mean_class/mean_all:.4f})")
                    dist_dict["Std in Class"].append(f"{std_class:.4f}")
                    dist_dict["Mean All vs. all"].append(f"{mean_all:.4f} ({mean_all/mean_all:.4f})")
                    dist_dict["Std All vs. all"].append(f"{std_all:.4f}")

                dist_plot_path = glob.glob(os.path.join(path_folder, f"val_{e}", dset1, "embeddings", f"distance_hist_between_{dset1}_and_{dset2}.png"))
                if len(dist_plot_path) == 1:
                    lines_plots.extend([f"**Histogram of cosine distances between {dset1} and {dset2}**\n",
                                        f"![histogram of distances]({os.path.join('.', f'val_{e}', dset1, 'embeddings', f'distance_hist_between_{dset1}_and_{dset2}.png')})\n\n"])
        
        if len(dist_dict["Datasets"]) > 0:
            lines.extend([f"### Epoch {e}\n\n",
                        "**(in brackets are the mean distances divided by the mean All versus all distance)**\n"])
            lines.extend(md_table_from_dict(dist_dict))

            if len(lines_plots) > 0:
                lines.append("#### Distance Histograms Plots\n\n")
                lines.extend(lines_plots)

    # write the markdown file
    if len(lines) > 1:
        with open(os.path.join(path_folder, "distances.md"), "w") as f:
            f.writelines(lines)

def create_shape_bias_md(path_folder):
    params_csv = open_csv_file(os.path.join(path_folder, "params.csv"))

    # set the title
    if "method" in params_csv:
        title = f"Shape Bias Metrics: {params_csv['method']}"
    else:
        title = "Shape Bias Metrics: CE"
    title += f" {params_csv['dataset']} {params_csv['tag']}"

    lines = [f"# {title}\n\n"]

    # Cue Conflict Shape Bias Metric
    # collect epochs and datasets
    cue_conf_paths = glob.glob(os.path.join(path_folder, "val_*", "shapeBiasMetrics", "CueConflict", "*", "shape_bias.csv"))
    cue_conf_datasets = sorted(set([cue_conf_path.split('/')[-2] for cue_conf_path in cue_conf_paths]))
    epochs_cue_conf = [e.replace(' ', '') for e in sorted(set([f"{cue_conf_path.split('/')[-5].replace('val_', ''):>3}" for cue_conf_path in cue_conf_paths]))]
    if len(cue_conf_paths) > 0:
        lines.append("## Texture Shape Cue Conflict Shape Bias Metric\n\n")

    # create for each epoch one table with shape biases for all datasets
    for e in epochs_cue_conf:
        list_df_bias = []

        # for each data set create one table with class shape biasses (and plots if they where found)
        lines_datasets = dict()
        for dset in cue_conf_datasets:
            sb_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "shapeBiasMetrics", "CueConflict", dset, "shape_bias.csv"))
            lines_datasets[dset] = []

            if len(sb_paths) == 1:
                df_bias = pd.read_csv(sb_paths[0], index_col=0)
                list_df_bias.append(df_bias)

                class_sb_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "shapeBiasMetrics", "CueConflict", dset, "classes_shape_bias.csv"))
                if len(class_sb_paths) == 1:
                    df_class_bias = pd.read_csv(class_sb_paths[0], index_col=0).reset_index(names="metrics")
                    lines_datasets[dset].append(f"#### Dataset {dset}\n\n")
                    lines_datasets[dset].extend(md_table_from_dict(df_class_bias))

                    bias_plot_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "shapeBiasMetrics", "CueConflict", dset, "shape_bias.png"))
                    recall_plot_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "shapeBiasMetrics", "CueConflict", dset, "classes_recall.png"))
                    if len(bias_plot_paths) == 1 and len(recall_plot_paths) == 1:
                        lines_datasets[dset].extend([f"Shape Bias | Class Recall\n",
                                                     ":--:|:--:\n",
                                                     f"![plot of the class shape biasses]({os.path.join('.', f'val_{e}', 'shapeBiasMetrics', 'CueConflict', dset, 'shape_bias.png')})",
                                                     f"|![plot of the class recalls]({os.path.join('.', f'val_{e}', 'shapeBiasMetrics', 'CueConflict', dset, 'classes_recall.png')})\n\n"])
                        
        if len(list_df_bias) > 0:
            lines.append(f"### Epoch {e}\n\n")
            df_biases = pd.concat(list_df_bias, axis=1).reset_index(names="metrics")
            lines.extend(md_table_from_dict(df_biases))

            for dset in cue_conf_datasets:
                if len(lines_datasets[dset]) > 0:
                    lines.extend(lines_datasets[dset])

    # Correlation Coefficient Shape Bias Metric
    # collect epochs and datasets
    corr_coef_paths = glob.glob(os.path.join(path_folder, "val_*", "shapeBiasMetrics", "CorrelationCoefficient", "*", "pred_dims.csv"))
    corr_coef_datasets = sorted(set([corr_coef_path.split('/')[-2] for corr_coef_path in corr_coef_paths]))
    epochs_corr_coef = [e.replace(' ', '') for e in sorted(set([f"{corr_coef_path.split('/')[-5].replace('val_', ''):>3}" for corr_coef_path in corr_coef_paths]))]
    if len(cue_conf_paths) > 0:
        lines.append("## Correlation Coefficient Shape Bias Metric\n\n")

    # create for each epoch one table with estimated dimensions for all datasets
    for e in epochs_corr_coef:
        list_df_dims = []
        for dset in corr_coef_datasets:
            dims_paths = glob.glob(os.path.join(path_folder, f"val_{e}", "shapeBiasMetrics", "CorrelationCoefficient", dset, "pred_dims.csv"))

            if len(dims_paths) == 1:
                df_dims = pd.read_csv(dims_paths[0], index_col=0)
                list_df_dims.append(df_dims)
        
        if len(list_df_dims) > 0:
            lines.extend([f"### Epoch {e}\n\n",
                          "**(in brackets are the estimated image component dimensions divided by the total number of dimensions of the feature space)**\n"])
            df_dimensions = pd.concat(list_df_dims, axis=0)
            embedding_size = np.nan_to_num(df_dimensions.values[0]).sum()
            df_dimensions = df_dimensions.map(lambda x: "" if np.isnan(x) else f"{int(x)} ({x/embedding_size:.4f})")

            columns = df_dimensions.columns.to_numpy()
            rem_dim_idx = np.where(columns == "remaining_dims")[0][0]
            if rem_dim_idx < len(columns)-1:
                columns = np.concatenate([columns[:rem_dim_idx], columns[rem_dim_idx+1:], [columns[rem_dim_idx]]])
                df_dimensions = df_dimensions[columns]
            df_dimensions = df_dimensions.reset_index(names="datasets")

            lines.extend(md_table_from_dict(df_dimensions))

    # write the markdown file
    if len(lines) > 1:
        with open(os.path.join(path_folder, "shape_bias.md"), "w") as f:
            f.writelines(lines)
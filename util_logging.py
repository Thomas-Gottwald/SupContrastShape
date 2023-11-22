import os
import glob
import re
import pandas as pd
import seaborn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

seaborn.set_theme(style="darkgrid")


# Convert teonsorboard logs into plots
def open_tensorboard(path):
    """
    Gets the scalar data from a tensorboard log in a dataFrame

    Parameters
    ---------
    path: str
        path to the tenorboard log file (even*) in the form <path>/tensorboard/event*
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


# run.md file for the trained models
def create_run_md(opt, mode="SupCon"):
    """
    Creates a run.md file containing the training parameters

    Parameters
    ---------
    opt
        parse options of the training call
    mode: str
        ether "SupCon" for (supervised) contrastive learning
        of "SupCE" for normal supervised crossentropy loss
    """
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = opt.mean
        std = opt.std

    if mode == "SupCon":
        lines = [
                    "<!-- param,tb -->\n"
                    "# Contrastive Training\n\n",
                    "## Training Parameters\n\n",
                    "#### Dataset\n\n",
                    "| dataset | mean | std | size |\n",
                    "|--|--|--|--|\n",
                    f"|{opt.dataset}|{mean}|{std}|{opt.size}|\n\n",
                    "#### Training\n\n",
                    "| model | method | temp | learning rate | lr decay epochs | lr decay rate | weight decay | momentum | batch size | epochs | cosine | warm |\n",
                    "|--|--|--|--|--|--|--|--|--|--|--|--|\n",
                    f"|{opt.model}|{opt.method}|{opt.temp}|{opt.learning_rate}|{opt.lr_decay_epochs}|{opt.lr_decay_rate}|{opt.weight_decay}|{opt.momentum}|{opt.batch_size}|{opt.epochs}|{opt.cosine}|{opt.warm}|\n\n",
                    "#### Loging\n\n",
                    "| print freq | save freq | num workers | trail | tag |\n",
                    "|--|--|--|--|--|\n",
                    f"|{opt.print_freq}|{opt.save_freq}|{opt.num_workers}|{opt.trial}|{opt.tag}|\n\n",
                    "#### Tensorboard\n\n",
                    "```\n",
                    f"tensorboard --logdir={opt.tb_folder}\n",
                    "```\n\n",
                ]
    elif mode == "SupCE":
        lines = [
                    "<!-- param,tb -->\n"
                    "# Contrastive Training\n\n",
                    "## Training Parameters\n\n",
                    "#### Dataset\n\n",
                    "| dataset | mean | std | size | num classes |\n",
                    "|--|--|--|--|--|\n",
                    f"|{opt.dataset}|{mean}|{std}|{opt.size}|{opt.n_cls}|\n\n",
                    "#### Training\n\n",
                    "| model | learning rate | lr decay epochs | lr decay rate | weight decay | momentum | batch size | batch size val | epochs | cosine | warm |\n",
                    "|--|--|--|--|--|--|--|--|--|--|--|\n",
                    f"|{opt.model}|{opt.learning_rate}|{opt.lr_decay_epochs}|{opt.lr_decay_rate}|{opt.weight_decay}|{opt.momentum}|{opt.batch_size}|{opt.batch_size_val}|{opt.epochs}|{opt.cosine}|{opt.warm}|\n\n",
                    "#### Loging\n\n",
                    "| print freq | save freq | num workers | num workers val | trail | tag |\n",
                    "|--|--|--|--|--|--|\n",
                    f"|{opt.print_freq}|{opt.save_freq}|{opt.num_workers}|{opt.num_workers_val}|{opt.trial}|{opt.tag}|\n\n",
                    "#### Tensorboard\n\n",
                    "```\n",
                    f"tensorboard --logdir={opt.tb_folder}\n",
                    "```\n\n",
                ]
    else:
        raise ValueError(mode)


    with open(os.path.join(opt.model_path, opt.model_name, "run.md"), "w") as f:
        f.writelines(lines)


def open_run_md(path):
    with open(os.path.join(path, "run.md"), "r") as f:
        lines = f.readlines()
    head = lines[0].replace("<!-- ", '').replace(" -->\n", '').split(',')

    return head, lines


def insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry):
    lines.insert(inset_idx, text_entry)
    head.insert(head_idx, head_entry)
    lines[0] = "<!-- " + ','.join(head) + " -->\n"

    with open(os.path.join(path, "run.md"), "w") as f:
        f.writelines(lines)


def get_insert_line(entry, head, epoch="last"):
    lines_dict = {"param": 22, "tb": 6, "train": 6, "val": 2, "epoch": 2, "tsne": 6, "class": 7}
    re_pattern = {"param": "param", "tb": "tb", "train": "train", "val": "val", "epoch": "last|[0-9]+", "tsne": "tsne(last|[0-9]+)", "class": "class(last|[0-9]+)"}
    entry_dict = {"train": (2,""), "val": (3,""), "epoch": (7,""), "tsne": (7,"last|[0-9]+"), "class": (7,"(tsne)?(last|[0-9]+)")}

    num_head, re_check = entry_dict[entry]
    new_epoch = f"{epoch}"

    count_lines = 1
    count_entries = 0
    for h in head:
        for n in list(lines_dict)[:num_head]:
            if re.fullmatch(re_pattern[n], h):

                if n in ["epoch", "tsne", "class"]:
                    check = re.fullmatch(re_check, h)
                    numbers = re.search("[0-9]+", h)
                    if new_epoch != "last" and numbers:
                        val = int(numbers.group(0))
                        if val < int(new_epoch) or (check and val <= int(new_epoch)):
                            count_lines += lines_dict[n]
                            count_entries += 1
                    elif new_epoch != "last" or (not numbers and check):
                        count_lines += lines_dict[n]
                        count_entries += 1
                else:
                    count_lines += lines_dict[n]
                    count_entries += 1
    return count_lines, count_entries


def add_val_to_run_md(path):
    head, lines = open_run_md(path)

    head_entry = "val"
    if head_entry in head:
        print(f"Validation headline already in {os.path.join(path, 'run.md')}.")
        return

    inset_idx, head_idx = get_insert_line("val", head)

    text_entry = "# Validation\n\n"

    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)


def add_epoch_to_run_md(path, epoch):
    head, lines = open_run_md(path)

    head_entry = f"{epoch}"
    if head_entry in head:
        print(f"Epoch {epoch} headline already in {os.path.join(path, 'run.md')}.")
        return

    if "val" not in head:
        add_val_to_run_md(path)
        head, lines = open_run_md(path)

    inset_idx, head_idx = get_insert_line("epoch", head, epoch)

    text_entry = f"## Epoch {epoch}\n\n"

    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)


def add_train_to_run_md(path):
    """Adds links plots of training learning rate and loss to the run.md file"""
    head, lines = open_run_md(path)

    head_entry = "train"
    if head_entry in head:
        print(f"The training info is already in {os.path.join(path, 'run.md')}.")
        return
    
    inset_idx, head_idx = get_insert_line("train", head)

    text_entry = "## Training\n\n"\
               + "Learning rate | Loss\n"\
               + ":--:|:--:\n"\
               + "![plot of learning rate scheduling](./tensorboard/learning_rate.png)|"\
               + "![plot of training loss](./tensorboard/loss.png)\n\n"

    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)


def add_tsne_to_run_md(path, epoch):
    """Adds links plots of t-SNE embeddings of a specific epoch to the run.md file"""
    head, lines = open_run_md(path)

    head_entry = f"tsne{epoch}"
    if head_entry in head:
        print(f"The t-SNE entry for epoch {epoch} is already in {os.path.join(path, 'run.md')}.")
        return
    
    if f"{epoch}" not in head:
        add_epoch_to_run_md(path, epoch)
        head, lines = open_run_md(path)

    inset_idx, head_idx = get_insert_line("tsne", head, epoch)

    text_entry = "### t-SNE Embedding\n\n"\
               + "Training data | Test data\n"\
               + ":--:|:--:\n"\
               + f"![t-SNE plot of epoch {epoch} training data](./val_{epoch}/embeddings/tSNE_epoch_{epoch}_train.png)|"\
               + f"![t-SNE plot of epoch {epoch} test data](./val_{epoch}/embeddings/tSNE_epoch_{epoch}_test.png)\n\n"

    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)


def add_class_to_run_md(path_class, best_acc):
    """Adds plots of classifier train loss and val acc of a specific epoch to the run.md file"""
    path = path_class.split('/')
    epoch = path[-3].replace("val_", '')
    path_plots = os.path.join(*path[-3:], "tensorboard")
    path = os.path.join(*path[:-3])
    head, lines = open_run_md(path)

    head_entry = f"class{epoch}"
    if head_entry in head:
        print(f"The classification info for epoch {epoch} is already in {os.path.join(path, 'run.md')}.")
        return
    
    if f"{epoch}" not in head:
        add_epoch_to_run_md(path, epoch)
        head, lines = open_run_md(path)

    inset_idx, head_idx = get_insert_line("class", head, epoch)

    text_entry = "### Classifier\n\n"\
               + f"**best top-1 accuracy: {best_acc:.2f}**\n"\
               + "Training loss | top-1 accuracy\n"\
               + ":--:|:--:\n"\
               + f"![plot of classifier training loss]({os.path.join(path_plots, 'train_loss.png')})|"\
               + f"![plot of classifier top-1 validation acc]({os.path.join(path_plots, 'val_top1.png')})\n\n"
    
    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)


def add_class_CE_to_run_md(path, best_acc):
    """
    Adds plots of train acc and val acc to the run.md file.
    Only for crossentropy loss
    """
    head, lines = open_run_md(path)

    head_entry = "classlast"
    if head_entry in head:
        print(f"The classification info is already in {os.path.join(path, 'run.md')}.")
        return
    
    inset_idx, head_idx = get_insert_line("class", head)

    text_entry = "## Classification\n\n"\
               + f"**best top-1 validation accuracy: {best_acc:.2f}**\n"\
               + "top-1 training accuracy | top-1 validation accuracy\n"\
               + ":--:|:--:\n"\
               + "![plot of classifier top-1 training acc](./tensorboard/train_top1.png)|"\
               + "![plot of classifier top-1 validation acc](./tensorboard/val_top1.png)\n\n"
    
    insert_into_run_md(path, head, lines, inset_idx, text_entry, head_idx, head_entry)

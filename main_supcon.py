from __future__ import print_function

import os
import sys
import argparse
import time
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util.util import TwoCropTransform, AverageMeter
from util.util import adjust_learning_rate, warmup_learning_rate
from util.util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from util.util_diff import DiffLoader, DiffTransform
from util.util_diff import SameTwoRandomResizedCrop, SameTwoColorJitter, SameTwoApply
from util.util_logging import create_csv_file_training, create_run_md, create_training_plots


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--diff_folder', type=str, default=None, help='path to diffused dataset')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'SupConHybrid'], help='choose method')
    parser.add_argument('--related_factor', type=float, default=1.0,
                        help='factor to adjust the effect of the related positives in the supCon loss')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    # augmentation
    parser.add_argument('--aug', nargs='*', default=['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale'],
                        choices=['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale', 'sameResizedCrop', 'sameHorizontalFlip', 'sameColorJitter', 'sameGrayscale'],
                        type=str, help='list of the used image augmentations')
    defaultResizedCrop = [0.2, 1.0, 3/4, 4/3]
    parser.add_argument('--resizedCrop', nargs='+', default=defaultResizedCrop,
                        type=float, help='crop scale lower and upper bound and resize lower and upper bound for aspect ratio')
    parser.add_argument('--horizontalFlip', default=0.5, type=float, help='probability for horizontal flip')
    defaultColorJitter = [0.8, 0.4, 0.4, 0.4, 0.4]
    parser.add_argument('--colorJitter', nargs='+', default=defaultColorJitter,
                        type=float, help='probability to apply colorJitter and how much to jitter brightness, contrast, saturation and hue')
    parser.add_argument('--grayscale', default=0.2, type=float, help='probability for random grayscale')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # optional identifier tag
    parser.add_argument('--tag', type=str, default='')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset not in ['cifar10', 'cifar100']:
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # check that of each augmentation type only one of the independent or identical is passed
    assert not('resizedCrop' in opt.aug and 'sameResizedCrop' in opt.aug)\
        and not('horizontalFlip' in opt.aug and 'sameHorizontalFlip' in opt.aug)\
        and not('colorJitter' in opt.aug and 'sameColorJitter' in opt.aug)\
        and not('grayscale' in opt.aug and 'sameGrayscale' in opt.aug)

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.related_factor == 1.0:
        opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
            format(opt.method, opt.dataset, opt.model, opt.learning_rate,
                   opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
    else:
        opt.model_name = '{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
            format(opt.method, opt.related_factor, opt.dataset, opt.model, opt.learning_rate,
                   opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    # add identifier tag to model name
    if opt.tag != '':
        opt.model_name = '{}_{}'.format(opt.model_name, opt.tag)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # parameters for randomResizedCrop
    if len(opt.resizedCrop) < len(defaultResizedCrop):
        opt.resizedCrop.extend(defaultResizedCrop[len(opt.resizedCrop):])
    elif len(opt.resizedCrop) > len(defaultResizedCrop):
        opt.resizedCrop = opt.resizedCrop[:len(defaultResizedCrop)]
    # parameters for colorJitter
    if len(opt.colorJitter) < len(defaultColorJitter):
        opt.colorJitter.extend(defaultColorJitter[len(opt.colorJitter):])
    elif len(opt.colorJitter) > len(defaultColorJitter):
        opt.colorJitter = opt.colorJitter[:len(defaultColorJitter)]

    opt.tb_folder = os.path.join(opt.model_path, opt.model_name, "tensorboard")
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, "models")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean = eval(opt.mean)
        std = eval(opt.std)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_same_transform = None
    same_transform_list = []
    if 'sameResizedCrop' in opt.aug:
        scaleMin, scaleMax, ratioMin, ratioMax = opt.resizedCrop
        same_transform_list.append(SameTwoRandomResizedCrop(size=opt.size, scale=(scaleMin, scaleMax), ratio=(ratioMin, ratioMax)))
    if 'sameHorizontalFlip' in opt.aug:
        same_transform_list.append(transforms.RandomApply([
            SameTwoApply(transforms.RandomHorizontalFlip(p=1.0))
        ], p=opt.horizontalFlip))
    if 'sameColorJitter' in opt.aug:
        pJitter, brightness, contrast, saturation, hue = opt.colorJitter
        same_transform_list.append(transforms.RandomApply([
            SameTwoColorJitter(brightness, contrast, saturation, hue)
        ], p=pJitter))
    if 'sameGrayscale' in opt.aug:
        same_transform_list.append(transforms.RandomApply([
            SameTwoApply(transforms.RandomGrayscale(p=1.0))
        ], p=opt.grayscale))
    if len(same_transform_list) > 0:
        train_same_transform = transforms.Compose(same_transform_list)

    transform_list = []
    if 'resizedCrop' in opt.aug:
        scaleMin, scaleMax, ratioMin, ratioMax = opt.resizedCrop
        transform_list.append(transforms.RandomResizedCrop(size=opt.size, scale=(scaleMin, scaleMax), ratio=(ratioMin, ratioMax)))
    if 'horizontalFlip' in opt.aug:
        transform_list.append(transforms.RandomHorizontalFlip(p=opt.horizontalFlip))
    if 'colorJitter' in opt.aug:
        pJitter, brightness, contrast, saturation, hue = opt.colorJitter
        transform_list.append(transforms.RandomApply([
            transforms.ColorJitter(brightness, contrast, saturation, hue)
        ], p=pJitter))
    if 'grayscale' in opt.aug:
        transform_list.append(transforms.RandomGrayscale(p=opt.grayscale))
    transform_list.extend([
        transforms.ToTensor(),
        normalize
    ])
    train_transform = transforms.Compose(transform_list)

    if opt.diff_folder:
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            loader=DiffLoader(path_orig=opt.data_folder, path_diff=opt.diff_folder),
                                            transform=DiffTransform(train_transform, train_same_transform))
    else:
        if opt.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform),
                                            download=True)
        elif opt.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform),
                                            download=True)
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                transform=TwoCropTransform(train_transform))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels, related_factor=opt.related_factor)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        elif opt.method == 'SupConHybrid':
            loss = criterion(features, labels, hybrid=True)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # create a run.md file containing the training parameters
    create_csv_file_training(opt, os.path.join(opt.model_path, opt.model_name, "params.csv"))
    create_run_md(os.path.join(opt.model_path, opt.model_name))

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # tensorboard
    writer.close()

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # add training details to the run.md file
    create_training_plots(path=os.path.join(opt.model_path, opt.model_name))
    create_run_md(os.path.join(opt.model_path, opt.model_name))


if __name__ == '__main__':
    main()

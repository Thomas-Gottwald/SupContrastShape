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

from util.util import AverageMeter
from util.util import adjust_learning_rate, warmup_learning_rate, accuracy
from util.util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet
from util.util_diff import DiffLoader, SelectTransform
from util.util_logging import create_csv_file_training, create_run_md, create_crossentropy_plots


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
    parser.add_argument('--batch_size_val', type=int, default=256,
                        help='batch_size for validation')
    parser.add_argument('--num_workers_val', type=int, default=8,
                        help='num of workers to use for validation')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
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
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset training data')
    parser.add_argument('--test_folder', type=str, default=None, help='path to custom dataset validation data')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes in the custom dataset')
    parser.add_argument('--diff_folder', type=str, default=None, help='path to diffused dataset. When given for training an original image gets replaced with its diffused version with p=diff_p.')
    parser.add_argument('--diff_p', default=0.5, type=float, help='probability to select diffused image if diff_folder is given')

    # augmentation
    parser.add_argument('--aug', nargs='*', default=['resizedCrop', 'horizontalFlip'],
                        choices=['resizedCrop', 'horizontalFlip', 'colorJitter', 'grayscale'],
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
            and opt.test_folder is not None \
            and opt.mean is not None \
            and opt.std is not None \
            and opt.num_classes is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCE/{}'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        opt.n_cls = opt.num_classes

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

    transform_list = []
    if opt.diff_folder:
        # if diff_folder is given original and diffused images get loaded and SelectTransform than selects one at random
        transform_list.append(SelectTransform(p=opt.diff_p))
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

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        if opt.diff_folder:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 loader=DiffLoader(path_orig=opt.data_folder, path_diff=opt.diff_folder),
                                                 transform=train_transform)
        else:
            train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                                 transform=train_transform)
        
        val_dataset = datasets.ImageFolder(root=opt.test_folder,
                                           transform=val_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size_val, shuffle=False,
        num_workers=opt.num_workers_val, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()


    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
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
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # create a run.md file containing the training parameters
    create_csv_file_training(opt, os.path.join(opt.model_path, opt.model_name, "params.csv"))
    create_run_md(os.path.join(opt.model_path, opt.model_name))

    # build data loader
    train_loader, val_loader = set_loader(opt)

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
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt)
        writer.add_scalar('val_loss', loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc

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

    print('best accuracy: {:.2f}'.format(best_acc))

    # add training and validation details to the run.md file
    create_crossentropy_plots(path=os.path.join(opt.model_path, opt.model_name))
    create_run_md(os.path.join(opt.model_path, opt.model_name))


if __name__ == '__main__':
    main()

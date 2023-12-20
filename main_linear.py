from __future__ import print_function

import os
import sys
import argparse
import time
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.backends.cudnn as cudnn

from main_ce import set_loader
from util.util import AverageMeter
from util.util import adjust_learning_rate, warmup_learning_rate, accuracy
from util.util import set_optimizer, save_model
from util.util_pre_com_feat import set_feature_loader, IdentityWrapperNet
from networks.resnet_big import SupConResNet, LinearClassifier
from util.util_logging import create_classifier_training_plots, add_class_to_run_md


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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
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

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--pre_comp_feat', action='store_true',
                        help='Use pre computed feature embedding')
    parser.add_argument('--md_file', type=str, default=None,
                        help='Name of the markdown file to write the results. Use only for different than training datasets!')

    # optional identifier tag
    parser.add_argument('--tag', type=str, default='')

    opt = parser.parse_args()

    # check if flag pre_comp_feat is set or dataset is path
    # that passed required arguments
    if opt.pre_comp_feat:
        assert opt.data_folder is not None \
            and opt.test_folder is not None
        if opt.dataset not in ['cifar10', 'cifar100']:
            assert opt.num_classes is not None
    elif opt.dataset not in ['cifar10', 'cifar100']:
        assert opt.data_folder is not None \
            and opt.test_folder is not None \
            and opt.mean is not None \
            and opt.std is not None \
            and opt.num_classes is not None
    # check if GPU is available when no precomputed feature embedding is given
    if not opt.pre_comp_feat and not torch.cuda.is_available():
        raise NotImplementedError('This code requires GPU without precomputed feature embedding')

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    if opt.pre_comp_feat:
        path_split = opt.test_folder.split('/')
        assert len(path_split) > 2
        opt.model_path = os.path.join(*path_split[:-2], "classifier")
    else:
        path_split = opt.chkp.split('/')
        assert len(path_split) > 2
        epoch = path_split[-1].replace(".pth", '').split('_')[-1]
        opt.model_path = os.path.join(*path_split[:-2], f"val_{epoch}", "classifier")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    # add identifier tag to model name
    if opt.tag != '':
        opt.model_name = '{}_{}'.format(opt.model_name, opt.tag)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.pre_comp_feat:
        opt.model_name = '{}_pre_comp_feat'.format(opt.model_name)

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


def set_model(opt):
    # for precomputed feature embedding use a dummy model witch acts as the identity
    if opt.pre_comp_feat:
        model = IdentityWrapperNet()
    else:
        model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    
    if not opt.pre_comp_feat:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        elif not opt.pre_comp_feat:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        if not opt.pre_comp_feat:
            model.load_state_dict(state_dict)
    # else:
    #     raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
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


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            top5.update(acc5[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@1 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f}, Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg


def main():
    best_acc = 0
    best_acc_top5 = 0
    opt = parse_option()

    # build data loader
    if opt.pre_comp_feat:
        train_loader, val_loader = set_feature_loader(opt)
    else:
        train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # tensorboard
    writer = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        
        # tensorboard logger
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # eval for one epoch
        loss, val_acc, val_acc_top5 = validate(val_loader, model, classifier, criterion, opt)
        writer.add_scalar('val_loss', loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_acc_top5', val_acc_top5, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            # save the best classifier
            save_best_file = os.path.join(
                opt.save_folder, 'best.pth')
            save_model(classifier, optimizer, opt, opt.epochs, save_best_file)
        if val_acc_top5 > best_acc_top5:
            best_acc_top5 = val_acc_top5

    # tensorboard
    writer.close()

    # save the last classifier
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(classifier, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}, best top5 accuracy {:.2f}'.format(best_acc, best_acc_top5))

    # add classifier training details to the run.md file
    create_classifier_training_plots(path_class=os.path.join(opt.model_path, opt.model_name))
    add_class_to_run_md(path_class=os.path.join(opt.model_path, opt.model_name), best_acc=best_acc, md_file=opt.md_file)


if __name__ == '__main__':
    main()

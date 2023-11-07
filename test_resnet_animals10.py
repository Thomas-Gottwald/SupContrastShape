import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from animals10loader import animals10Dataset


def set_loader(dataset='animals10'):
    # construct data loader
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
        batchsize = 512
    else:
        # for animasl10 160to300
        mean = (0.3890, 0.3757, 0.3122)
        std = (0.3279, 0.3201, 0.3073)
        size = 300
        batchsize = 16
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root="./datasets/",
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    else:
        train_dataset = animals10Dataset(transform=TwoCropTransform(train_transform))
    
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=(train_sampler is None),
        num_workers=16, pin_memory=True, sampler=train_sampler)

    return train_loader


def main():

    # cuda_device = 1

    # build data loader
    train_loader = set_loader(dataset='cifar10')

    print("create model")
    model = SupConResNet(name="resnet18")
    criterion = SupConLoss(temperature=0.1)

    if torch.cuda.is_available():
        print("put model on GPU")
        # model.cuda(device=cuda_device)
        model.cuda()
        print("model on GPU")
        # criterion = criterion.cuda(device=cuda_device)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    
    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            # images = images.cuda(device=cuda_device, non_blocking=True)
            # labels = labels.cuda(device=cuda_device, non_blocking=True)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        print(f"bsz={bsz}, images.shape={images.shape}, labels.shape={labels.shape}")

        # create features
        features = model(images)
        print(f"features.shape={features.shape}")
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        print(f"f1.shape={f1.shape}, f2.shape={f2.shape}")
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        print(f"features.shape={features.shape}")

        # Loss
        loss_supcon = criterion(features, labels)
        print(f"SupCon loss={loss_supcon}")
        loss_simclr = criterion(features)
        print(f"SimCLR loss={loss_simclr}")
        break


if __name__ == '__main__':
    main()

# For 300x300
# create model
# put model on GPU
# model on GPU
# bsz=1, images.shape=torch.Size([2, 3, 300, 300]), labels.shape=torch.Size([1])
# encoder in: torch.Size([2, 3, 300, 300])
# torch.Size([2, 3, 300, 300])
# torch.Size([2, 64, 300, 300])
# torch.Size([2, 256, 300, 300])
# torch.Size([2, 512, 150, 150])
# torch.Size([2, 1024, 75, 75])
# torch.Size([2, 2048, 38, 38])
# torch.Size([2, 2048, 1, 1])
# torch.Size([2, 2048])
# encoder out: torch.Size([2, 2048])
# head out: torch.Size([2, 128])
# features.shape=torch.Size([2, 128])
# f1.shape=torch.Size([1, 128]), f2.shape=torch.Size([1, 128])
# features.shape=torch.Size([1, 2, 128])
# SupCon loss=2.1287373641598606e-08
# SimCLR loss=2.1287373641598606e-08

# For 32x32
# create model
# put model on GPU
# model on GPU
# bsz=1, images.shape=torch.Size([2, 3, 32, 32]), labels.shape=torch.Size([1])
# encoder in: torch.Size([2, 3, 32, 32])
# torch.Size([2, 3, 32, 32])
# torch.Size([2, 64, 32, 32])
# torch.Size([2, 256, 32, 32])
# torch.Size([2, 512, 16, 16])
# torch.Size([2, 1024, 8, 8])
# torch.Size([2, 2048, 4, 4])
# torch.Size([2, 2048, 1, 1])
# torch.Size([2, 2048])
# encoder out: torch.Size([2, 2048])
# head out: torch.Size([2, 128])
# features.shape=torch.Size([2, 128])
# f1.shape=torch.Size([1, 128]), f2.shape=torch.Size([1, 128])
# features.shape=torch.Size([1, 2, 128])
# SupCon loss=-4.257474728319721e-08
# SimCLR loss=-4.257474728319721e-08
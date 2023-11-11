import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from networks.resnet_big import SupConResNet, LinearClassifier

class featureEmbeddingDataset(Dataset):

    def __init__(self, root):
        
        with open(root, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data = entry['data']
            self.targets = entry['labels']

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        feature = torch.from_numpy(self.data[index])
        target = self.targets[index]

        return feature, target
        

class IdentityNet(nn.Module):
    """Just an Identity"""
    def __init__(self):
        super(IdentityNet, self).__init__()

    def forward(self, x):
        return x.float()
    

class IdentityWrapperNet(nn.Module):
    """Just the identity has an encoder wich is also the identity"""
    def __init__(self):
        super(IdentityWrapperNet, self).__init__()
        self.encoder = IdentityNet()

    def forward(self, x):
        return self.encoder(x)
    

def set_feature_loader(opt):
    """Set tain and validation DataLoader for precomputed feature embeddings"""
    train_dataset = featureEmbeddingDataset(root=opt.data_folder)

    val_dataset = featureEmbeddingDataset(root=opt.test_folder)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size_val, shuffle=False,
        num_workers=opt.num_workers_val, pin_memory=True)
    
    return train_loader, val_loader



def main():
    # Testing

    train_dataset = featureEmbeddingDataset("./save/embeddings/animals10/resnet18/embedding_train")

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=(train_sampler is None),
        num_workers=8, pin_memory=True, sampler=train_sampler)

    model = IdentityWrapperNet()

    classifier = LinearClassifier(name="resnet18", num_classes=10)

    print(torch.cuda.is_available())
    model = model.cuda(device=2)
    classifier = classifier.cuda(device=2)

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):

            images = images.float().cuda(device=2)
            print(images.get_device())

            features = model.encoder(images)
            print(labels)
            print(features.shape, features.get_device())

            output = classifier(features)

            print(output.shape, output.get_device())

            break


    # model.eval()
    # classifier.train()
    # for idx, (images, labels) in enumerate(train_loader):

    #     images = images.cuda(device=2, non_blocking=True)
    #     print(images.get_device())
        
    #     with torch.no_grad():
    #         features = model.encoder(images)
    #     print(labels)
    #     print(features.shape, features.get_device())

    #     output = classifier(features.detach())

    #     print(output.shape, output.get_device())

    #     break



if __name__ == '__main__':
    main()
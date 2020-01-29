import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.models as models
import pdb


def prepare_data(batch_size, dataset, dataroot):

    if dataset == 'celeba':
        celebTrans = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize(64),
            transforms.ToTensor()
        ])

        trainset = dsets.CelebA(root=dataroot,split='train',transform=celebTrans,download=False)

        testset = dsets.CelebA(root=dataroot,split='test',transform=celebTrans,download=False)

        train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,num_workers=4)

        test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False,num_workers=4)

    elif dataset == 'mnist':

        trans = transforms.ToTensor()

        trainset = dsets.MNIST(root=dataroot, train=True,transform=trans,download=True)

        testset = dsets.MNIST(root=dataroot, train=False,transform=trans,download=True)

        train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,num_workers=4)

        test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False,num_workers=4)

    elif dataset == 'cifar':

        trans = transforms.ToTensor()

        trainset = dsets.CIFAR10(root=dataroot, train=True,transform=trans,download=False)

        testset = dsets.CIFAR10(root=dataroot, train=False,transform=trans,download=False)

        train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True,num_workers=4)

        test_loader = DataLoader(dataset=testset,batch_size=batch_size,shuffle=False,num_workers=4)

    return train_loader, test_loader








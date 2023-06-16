import time                                                                                 
from typing import Optional, List
import torch
import torch.nn as nn
Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
from torchvision import datasets, transforms
import numpy as np
import math
import torchvision
import matplotlib.pyplot as plt
from functools import partial 
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_
Uniform = torch.distributions.Uniform
#from sklearn.datasets import make_moons, make_circles, make_classification
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF

def make_dataloaders(trainer):
    
    dset = trainer.config.dataset
            
    # sizes
    if dset == 'mnist':
        img_size, channels = 28, 1
    elif dset == 'cifar':
        img_size, channels = 32, 3
    elif dset == 'flowers':
        img_size, channels = 128, 3 
    else:
        assert False
    flat_d = img_size * img_size * channels

    # transforms
    if dset == 'mnist':
        train_tf = transforms.Compose([transforms.ToTensor(),])
        test_tf = train_tf
    elif dset == 'cifar':
        train_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(), # TODO CHECK
                # transforms.Normalize([0.5], [0.5]),
        ])
        test_tf = transforms.Compose([transforms.ToTensor()])
    elif dset == 'flowers':
        assert False
    else:
        assert False

    # datasets  
    if dset == 'mnist':
        dset_fn = datasets.MNIST
    elif dset == 'cifar':
        dset_fn = datasets.CIFAR10
    elif dset == 'flowers':
        assert False
    else:
        assert False

    trainset = dset_fn('./data', train=True, download=True, transform=train_tf)
    testset = dset_fn('./data', train=False, download=True, transform=test_tf)

    # loaders
    use_cuda = trainer.config.use_cuda
    num_workers = trainer.config.num_workers
    train_kwargs = {'batch_size': trainer.config.bsz, 'shuffle': True, 'drop_last': True}
    test_kwargs = {'batch_size': trainer.config.bsz, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': num_workers, 
            'pin_memory': True, 
            'persistent_workers': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

     
    trainer.train_loader = DataLoader(trainset, **train_kwargs)
    trainer.test_loader = DataLoader(testset, **test_kwargs)
    
    trainer.config.d = flat_d 
    trainer.config.is_image = True
    trainer.config.C = channels
    trainer.config.H = img_size
    trainer.config.W = img_size

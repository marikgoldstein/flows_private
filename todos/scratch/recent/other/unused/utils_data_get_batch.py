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

# local
from sklearn.datasets import make_moons, make_circles, make_classification

def get_batch_mnist(trainer, loader, bsz):                                                                                     
    batch = next(iter(loader))
    x, _ = batch
    x = x.to(trainer.device)
    return x[:bsz]

def get_batch_gmm(trainer, bsz):
    return trainer.gmm.q1.sample(sample_shape=(bsz,))

def get_batch_moons(trainer, bsz):
    p = torch.randperm(trainer.num_moons,)
    rand_moons = trainer.moons[p]
    some_moons = rand_moons[:bsz]
    return some_moons


def build_data(trainer,):                                                                                                      
    if trainer.dataset == 'mnist':
        data_dict = build_mnist(trainer.bsz, trainer.use_cuda, trainer.config.num_workers)
    else:
        assert False

    for key in data_dict:
        setattr(trainer, key, data_dict[key])

    assert trainer.train_loader is not None
    assert trainer.test_loader is not None

    if trainer.dataset == 'mnist':
        trainer.get_batch = partial(get_batch_mnist, trainer, trainer.train_loader)
        trainer.get_batch_eval = partial(get_batch_mnist, trainer, trainer.test_loader)

# trainload, test_loader
def build_mnist(bsz, use_cuda, num_workers):

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    testset = datasets.MNIST('../data', train=False, transform=transform)
    train_kwargs = {'batch_size': bsz, 'shuffle': True, 'drop_last': True}
    test_kwargs = {'batch_size': bsz, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    return {
        'd': 784,
        'is_image': True,
        'C': 1,
        'W': 28,
        'H': 28,
        'train_loader': train_loader,
        'test_loader': test_loader
    }


'''

        self.d=2
        self.d=2
        self.num_moons = 4096

        self.get_batch = self.get_batch_mnist


    elif self.dataset=='gmm':

        self.get_batch = self.get_batch_gmm

    elif self.dataset=='moons':
                    self.moons = torch.tensor(make_moons(n_samples=self.num_moons, noise=0.1, shuffle=True, random_state=0)[0]).float()
        self.get_batch = self.get_batch_moons

'''

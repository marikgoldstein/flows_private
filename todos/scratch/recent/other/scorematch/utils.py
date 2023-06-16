import glob
import tqdm
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN

# custom / local
#from ncsn import NCSN, NCSNdeeper
#from ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
#from losses import get_optimizer
from get_sigmas import get_sigmas


def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False


def dsm(scorenet, samples, sigmas, labels=None, anneal_power=2.):                                                                                                                                     
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    #target = - 1 / (used_sigmas ** 2) * noise # TODO ????
    target = - noise / used_sigmas  
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    return loss.mean(dim=0)


@torch.no_grad()
def ald(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008, final_only=False, verbose=False, denoise=True):
    images = []
    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images




def get_optimizer(config, parameters):
    if config.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.lr, weight_decay=config.wd, betas=(config.beta1, 0.999), amsgrad=config.amsgrad, eps=config.optim_eps)
    elif config.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.lr, weight_decay=config.wd)
    elif config.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def make_cifar_data(trainer):
    
    # resize transforms? # note p=0.5
    
    if trainer.config.random_flip:
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    trainer.train_dataset = CIFAR10(
            './data',
            train=True,
            download=True,
            transform = train_transform
    )
    trainer.test_dataset = CIFAR10(
            './data',
            train=False,
            download=True,
            transform = test_transform
    )

    trainer.train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=trainer.config.train_bsz,
            shuffle=True,
            num_workers=trainer.config.num_workers,
            pin_memory=True,
            drop_last=True
    )
    trainer.test_loader = DataLoader(
            trainer.test_dataset,
            batch_size=trainer.config.test_bsz,
            shuffle=False,
            num_workers=trainer.config.num_workers,
            pin_memory=True,
            drop_last=True
    )
    trainer.train_iter = iter(trainer.train_loader)
    trainer.test_iter = iter(trainer.test_loader)


import pathlib
import os
import pickle
from typing import Optional, List


import os
import click
import pickle
import numpy as np
import scipy.linalg
import torch
import random
from glob import glob
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import numpy as np
import math
import wandb
import copy
import uuid
from pathlib import Path                              
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os                        

from cleanfid import fid 
# local
from utils import (
    is_type_for_logging,
    requires_grad,
    create_logger,
)
from diffusers.models import AutoencoderKL


class Trainer:

    def __init__(self, num_samples):


        device = torch.device('cuda')
        self.num_samples = num_samples
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

        RESULTS_DIR = './results'
        self.results_dir = RESULTS_DIR
        os.makedirs(RESULTS_DIR, exist_ok=True)
        experiment_index = len(glob(f"{RESULTS_DIR}/*"))
        experiment_dir = f"{RESULTS_DIR}/{experiment_index:03d}-moo"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        sample_dir = os.path.join(checkpoint_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Will be Saving png samples at {sample_dir}")
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.bsz = 1000
        train_transform = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(
                './data',
                train=True,
                download=True,
                transform = train_transform
        )

        self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.bsz,
                shuffle=True, 
                num_workers=4,
                pin_memory=True,
                drop_last=True
        )


    def generate_image_grid(self, images):
        """Simple helper to generate a single image from a mini batch."""
        images = images.cpu()
        batch_size = images.shape[0]
        grid_size = int(np.floor(np.sqrt(batch_size)))   
        images = images[0:grid_size**2]                  
        grid = torchvision.utils.make_grid(images, nrow=grid_size)   
        grid = grid.permute(1,2,0)                       
        return grid.numpy()                              
       
    def make_sample_dirs_this_step(self,):
        D = self.get_sample_dirs_this_step()
        os.makedirs(D, exist_ok = True)

    def get_sample_dirs_this_step(self,):
        subdir_name = 'step1'
        D = os.path.join(self.sample_dir, subdir_name) 
        return D

    def sample(self,):
        self.total = 0
        self.make_sample_dirs_this_step()
        total_num_samples = self.num_samples
        iterations = int(total_num_samples / self.bsz)
        train_iter = iter(self.train_loader)
        for sampling_round in range(iterations):
            print(f"iter is {sampling_round}/{iterations}")
            (x,y) = next(train_iter)
            D = self.get_sample_dirs_this_step()
            self.write_samples(x, D, sampling_round)
        self.total = 0
   
    def write_samples(self, samples, folder, sampling_round):
        bsz = samples.shape[0]  
        assert torch.all(samples >= 0.) and torch.all(samples <= 1.)


        for i, sample in enumerate(samples):
            index = self.total
            path_i = os.path.join(folder, f'im_{index:06d}.png')
            t = 1e-3
            assert torch.all(sample <= 1.) and torch.all(sample >= 0.)

            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            vae.decode(samples / 0.18215).sample

            sample = self.center(smaple)
            sample = self.noise(sample, t)
            sample = self.uncenter(sample)
            
            
            USE_PIL = True
            #USE_PIL = False
            if USE_PIL:
                sample = torch.clamp(sample * 255. , 0, 255)
                sample = sample.permute(1,2,0)
                sample = sample.to('cpu', dtype=torch.uint8).numpy()
                assert sample.shape[-1] == 3
                Image.fromarray(sample, 'RGB').save(path_i, compress_level=0)
            else:
                assert sample.shape[0] == 3
                save_image(sample, path_i)

            self.total+=1


    def noise(self, x, t):
        mean_coef_squared = 1-t
        variance = t
        mean_coef = np.sqrt(mean_coef_squared)
        return mean_coef * x + np.sqrt(variance) * torch.randn_like(x)

    def center(self, x):
        return x * 2. - 1.

    def uncenter(self, x):
        return (x+.1) / 2.0

    def compute_fid(self, path = None):
        dataset_split="train"                      
        #dataset_split='test'
        if path is not None:
            D = path
        else:
            D = self.get_sample_dirs_this_step()
        score = fid.compute_fid(D, 
                    dataset_name = 'cifar10',
                    dataset_res = 32,        
		    model_name = 'inception_v3',
                    dataset_split = 'train',
                    mode='clean'                   
        )                 
        print("SCORE", score)


if __name__ == '__main__':
    trainer = Trainer(num_samples=50000)
    trainer.sample()
    trainer.compute_fid(path=None)





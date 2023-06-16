import pathlib
import os
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
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

class ImageProcessing:

    def __init__(self, config):
        self.config = config
        self.C = self.config.C
        self.H = self.config.H
        self.W = self.config.W

    def assert_BCHW(self, x, bsz):
        assert x.shape == (bsz, self.C, self.H, self.W)

    def assert_CHW(self, x):
        assert x.shape == (self.C, self.H, self.W)

    def clamp_unit(self, x):
        return torch.clamp(x, 0., 1.)

    def to_255(self, x):
        return torch.clamp(x * 255., 0., 255.)

    def CHW_TO_HWC(self, x):
        return x.permute(1,2,0)

    def to_torch_cpu_uint8(self, x):
        return x.to('cpu', dtype=torch.uint8)

    def to_numpy(self, x):
        return x.numpy()

    def assert_channel_first(self, x):
        assert x.shape[0] == self.C

    def assert_channel_last(self, x):
        assert x.shape[-1] == self.C

    def decide_pil_mode(self,):
        if self.config.dataset == 'cifar':
            mode = 'RBG'
        elif self.config.dataset == 'mnist':
            mode = 'L'
        else:
            assert False
        return mode

    def fid_pil_preprocess(self, sample):
        
        saining = True # trying out a different method, will pick one and stick with it
        self.assert_CHW(sample)
        if saining:
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255).permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy()
        else:
            sample = self.to_255(sample)
            sample = self.CHW_TO_HWC(sample)
            sample = self.to_torch_cpu_uint8(sample)
            sample = self.to_numpy(sample)
        self.assert_channel_last(sample)
        mode = self.decide_pil_mode()
        return sample, mode
   
    def fid_torchvision_preprocess(self, sample):
        sample = self.clamp_unit(sample)
        self.assert_channel_first(sample)

    def save_images_for_fid(self, samples, folder):
    
        for i, sample_i in enumerate(samples):

            path_i = os.path.join(folder, f'img_{i}.png')
            saining = True
            if saining:
                sample_i, mode = self.fid_pil_preprocess(sample_i)
                
                if self.config.dataset == 'mnist':
                    Image.fromarray(sample_i, mode).save(path_i)
                else:
                    Image.fromarray(sample_i).save(path_i)

            elif self.config.use_pil:
                sample_i, mode = self.fid_pil_preprocess(sample_i)
                Image.fromarray(sample_i, mode).save(path_i, compress_level=0)
            else:
                sample_i = self.fid_torchvision_preprocess(sample_i)
                save_image(sample, path_i)
        
        print("images saved")
    
    def process_images_for_wandb(self, samples):
        samples = self.to_255(samples)
        samples = samples.byte()
        samples = samples.cpu()
        bsz = samples.shape[0]
        self.assert_BCHW(samples, bsz)
        grid_size = int(np.floor(np.sqrt(bsz)))
        samples = samples[:grid_size**2]
        grid = torchvision.utils.make_grid(samples, nrow=grid_size)
        grid = grid.permute(1,2,0)
        grid = grid.numpy()
        grid = grid[None, ...] 
        assert grid.shape[0] == 1
        assert grid.shape[-1] == 3
        samples = [
                wandb.Image(np.array(x)) for x in grid
        ]
        return samples



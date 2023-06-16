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
from diffusers.models import AutoencoderKL
from cleanfid import fid 
# local
from utils import (
    is_type_for_logging,
    requires_grad,
)
from image_processing import ImageProcessing
from data_utils import setup_data
from diffusion import Diffusion
from flow import Flow
from dit import get_dit
from encoder import Encoder
from config import ExperimentConfig
from diffusers.models import AutoencoderKL

# requires
#config.T_min
#config.T_max
#config.n_sample_steps
#config.num_classes
#config.C
#config.H
#config.W
#config.original_lr
#config.wd
#config.seed
#config.use_wandb
#config.wandb_entity
#config.wandb_name
#config.wandb_project
#config.num_sampled_images
#config.loss_type
#config.sample_every
#config.log_every
#config.total_train_steps
#config.debug_cfg
#config.arch_size
#config.dataset
#config.use_pil

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
 
class Trainer(nn.Module):
    
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.device = torch.device('cuda')
        self.use_vae = False
        os.makedirs(self.config.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{self.config.results_dir}/*"))
        model_string_name = self.config.arch
        experiment_dir = f"{self.config.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        self.checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("results dir", self.config.results_dir)
        print("experiment dir", experiment_dir)
        print("ckpt dir", self.checkpoint_dir)
        self.seed = 0
        torch.manual_seed(self.seed)
        setup_data(trainer = self, ddp=False, rank=None)
        
        if self.use_vae:
            assert self.config.dataset == 'imagenet'
            self.config.C_dgm = 4
            self.config.H_dgm = self.config.H // 8
            self.config.W_dgm = self.config.W // 8
        else:
            self.config.C_dgm = self.config.C
            self.config.H_dgm = self.config.H
            self.config.W_dgm = self.config.W

        self.model = get_dit(self.config)
        self.ema_model = deepcopy(self.model).to(self.device)  
        requires_grad(self.ema_model, False)
        self.model.to(self.device)
        if self.use_vae:
            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(self.device)
        
        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.original_lr, # saining uses 1e-4
            weight_decay=self.config.wd
        )
        self.dgm_fn = Diffusion if self.config.dgm_type == 'diffusion' else Flow
        self.dgm = self.dgm_fn(self.config)
        
        if not self.use_vae:
            self.encoder = Encoder(self.config)
        self.train_steps = 0
        self.epoch = -1
        self.setup_wandb()
        self.image_processing = ImageProcessing(self.config)
        self.update_ema_model(decay=0.) # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema_model.eval()  # EMA model should always be in eval mode        
        self.start_time = time()                          
 
    def setup_wandb(self):
        print("setting up wandb")
        if self.config.use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name = self.config.wandb_name,
            )               
            self.config.wandb_id = self.wandb_run.id
            for key in vars(self.config):
                item = getattr(self.config, key)
                if is_type_for_logging(item):
                    setattr(wandb.config, key, getattr(self.config, key))
        print("wandb setup")

    def update_ema_model(self, decay):
        print("updating ema model")
        ema_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        for name, param in model_params.items():
            # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        print("updated ema")

    def save_ckpt(self):
        print("saving ckpt")
        checkpoint = {
            "model": self.model.state_dict(),                                                                                                      
            "ema": self.ema_model.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.config,
        } 
        checkpoint_path = f"{self.checkpoint_dir}/{self.train_steps:09d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print("saved checkpoint at", checkpoint_path)

    def step_opt(self, loss):
        loss.backward()
        if self.train_steps % self.config.grad_accum == 0:
            for param in self.model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 2.0)
            self.opt.step()
            self.opt.zero_grad()

    @torch.no_grad()
    def sample(self, which):
        assert which in ['nonema', 'ema']
        if which == 'nonema':
            model = self.model
        else:
            model = self.ema_model
        model.eval()
        print(f"sampling from model: {which}")
        config = self.config
        num_images = self.config.num_sampled_images
        zhat = self.dgm.sample(n_samples = num_images, model=model, device=self.device)
        #xhat = self.encoder.generate_x(zhat)
        if self.use_vae:
            xhat = self.vae.decode(zhat / 0.18215).sample
        else:
            xhat = self.encoder.inverse_scaler(zhat)
            
        if self.config.use_wandb:
            samples = self.image_processing.process_images_for_wandb(xhat.clone())
            wandb.log({f'{which}_samples': samples}, step = self.train_steps)
        model.train()

    def training_loop(self,):
        print("Starting training")
        self.sample('nonema')
        self.model.train()
        while self.train_steps < self.config.total_train_steps:
            self.epoch += 1
            for batch_idx, (x, y) in enumerate(self.train_loader):
                #centered_data, y, ldj = self.prepare_batch(batch, y)
                x,y = x.to(self.device), y.to(self.device)

                if self.use_vae:

                    with torch.no_grad():
                        # Map input images to latent space + normalize latents:
                        z = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                else:

                    z, ldj = self.encoder.scaler(x)


                bsz = z.shape[0]
                loss = self.dgm.loss_fn(z, y, self.model, self.device, loss_type = self.config.loss_type)['loss']
                loss = loss.mean()
                loss_item = loss.item()
                self.step_opt(loss)
                self.train_steps += 1
                if self.train_steps % self.config.log_every == 0:
                    print(f"Step:{self.train_steps}, Loss:{loss_item}")
                    wandb.log({'loss':loss_item},step=self.train_steps)
                if self.train_steps % self.config.update_ema_every == 0 and self.train_steps > self.config.update_ema_after:
                    self.update_ema_model(self.config.ema_decay)
                if self.train_steps % self.config.sample_every == 0:
                    self.sample('nonema')
                if self.train_steps % self.config.sample_ema_every == 0 and self.train_steps > self.config.update_ema_after:
                    self.sample('ema')
                if self.train_steps % self.config.ckpt_every == 0 and self.train_steps > 0:
                    self.save_ckpt()
 
if __name__=='__main__':

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--dgm_type', type=str, default='flow')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--do_resume', type=int, default=0)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='./ckpts')
    parser.add_argument('--load_ckpt_name', type=str, default='demowtoshdiudhg')
    parser.add_argument('--print_hp', type=int, default=0)
    parser.add_argument('--s_denom', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='imagenet')

    main_args = parser.parse_args()
    conf = ExperimentConfig(main_args)
    np.random.seed(conf.global_seed)  
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
  
    for key in vars(conf):
        print(key,":",getattr(conf,key))

    trainer = Trainer(conf)
    trainer.training_loop()

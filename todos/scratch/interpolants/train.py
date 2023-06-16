import os
import torch
import torch.nn as nn
import numpy as np
import math
import wandb
import copy
import uuid
from PIL import Image
from time import time
import argparse
import logging

# local files
from data_utils import setup_data
from flow import ContinuousTimeDGM
from config import ExperimentConfig
from boilerplate import make_directories, setup_wandb, maybe_label_drop, save_ckpt, setup_overfit
from fid_layer import maybe_compute_fid

class Trainer(nn.Module):
    
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.config.device = self.device
        self.config, self.logger = make_directories(self.config)
        setup_data(trainer = self, ddp=False, rank=None)
        self.dgm = ContinuousTimeDGM(self.config)
        self.x, self.y = setup_overfit(self.config, self.train_loader, mode = 'batch')
        
        # training state
        self.train_steps = 0
        self.grad_steps = 0 # differs if using grad accum
        self.epoch = 0
        setup_wandb(self.config)
        self.start_time = time()                          

    def maybe_save_ckpt(self):        
        steps = self.train_steps
        ckpt_every = self.config.ckpt_every 
        if steps % ckpt_every == 0 and steps > 0:
            D = self.dgm.get_ckpt_dict()
            save_ckpt(self.config, steps, D)

    def step_optimizer(self):
        actually_stepped = self.dgm.step_optimizer(self.train_steps, log_norm = True)
        if actually_stepped:
            self.grad_steps += 1
        self.train_steps += 1

    def plot_real_data(self,):
    
        conf = self.config
        x, y = self.x, self.y if conf.debug else next(iter(self.train_loader))
        if conf.use_wandb:
            x = self.dgm.image_processing.process_images_for_wandb(x)
            wandb.log({'real_data': x}, step = self.train_steps)

    def maybe_sample_nonema(self, cheap):
        conf = self.config
        steps = self.train_steps
        if steps % conf.sample_every == 0:
            self.sample(use_ema = False, cheap = cheap)

    def maybe_sample_ema(self, cheap):
        conf = self.config
        steps = self.train_steps
        cond1 = conf.sample_with_ema
        cond2 = steps > conf.update_ema_after
        cond3 = steps % conf.update_ema_every == 0
        if cond1 and cond2 and cond3:
            self.sample(use_ema = True, cheap = cheap)
            
    @torch.no_grad()
    def sample(self, use_ema = False, cheap = False):

        D = self.dgm.sample(
            use_ema = use_ema,
            cheap = cheap,
            train_steps = self.train_steps
        )
        if conf.use_wandb:
            wandb.log(D, step = self.train_steps)

        print("done sampling")

    def maybe_log(self, loss_dict):
       
        loss_item = loss_dict['loss'].item()
        steps = self.train_steps
        conf = self.config
        if steps % conf.log_every == 0:
        
            print(f"Step:{steps}, Loss:{loss_item}")
            
            if conf.use_wandb:
                wandb.log(loss_dict, step = steps)


    def prepare_batch(self, x, y):
        conf = self.config
        if conf.debug:
            x, y = self.x, self.y
        x, y = x.to(self.device), y.to(self.device)
        if not conf.debug:
            y = maybe_label_drop(conf, y)
        z, _ = self.dgm.encoder.encode(x)
        return x, y, z

    def one_epoch(self,):

        conf = self.config

        for batch_idx, (x, y) in enumerate(self.train_loader):

            x, y, z = self.prepare_batch(x, y)
            loss_dict = self.dgm.loss_fn(z, y)
            loss_dict['loss'].backward()
            self.step_optimizer()
            self.maybe_log(loss_dict)
            self.dgm.maybe_update_emas(self.train_steps)
            self.maybe_sample_nonema(cheap = False)
            self.maybe_sample_ema(cheap = False)
            maybe_compute_fid(trainer = self, which = 'flow_ode', use_ema = False, train_steps = self.train_steps)
            self.maybe_save_ckpt()

    def preamble(self,):
        print("Checks before training")
        #self.sample(use_ema = False, cheap = True) # only a couple of steps to test
        self.plot_real_data()
        print("done")

    def training_loop(self,):
      
        conf = self.config
        self.preamble()
        self.logger.info(f"Training! (message from logger)")
        
        for m in self.dgm.models:
            if 'ema' not in m:
                self.dgm.models[m].train()

        while self.train_steps < conf.total_train_steps:
            
            self.one_epoch()
            self.epoch += 1 # everything on level of "steps" but still track epochs

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='./ckpts')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--debug', type=int, default=1)
    main_args = parser.parse_args()
    
    conf = ExperimentConfig(
        use_wandb = True if main_args.use_wandb == 1 else False,
        debug = True if main_args.debug == 1 else False,
        results_dir = main_args.results_dir,
        dataset = main_args.dataset
    )
    np.random.seed(conf.global_seed)  
    torch.manual_seed(conf.global_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
  
    for key in vars(conf):
        print(key,":",getattr(conf,key))

    trainer = Trainer(conf)
    trainer.training_loop()

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
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

# local files
from data_utils import setup_data
from dgm import Flow
from config import ExperimentConfig
from boilerplate import (
    maybe_sample, 
    maybe_fid,
    maybe_ckpt,
    maybe_update_emas,
    spmd_boilerplate,
    maybe_label_drop,
    setup_torch_backends,
    maybe_log, 
    make_directories_distributed,
    setup_ddp,
    TrainState,
    setup_overfit,
    setup_wandb_distributed
)

class Trainer(nn.Module):
    
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.state = TrainState()
        self.rank, self.device = setup_ddp(config)
        args = {
            'config': self.config,
            'rank': self.rank,
            'device': self.device,
            'ddp': True,
            'mode': 'batch'
        }
        self.loader, self.sampler = setup_data(**args)
        self.config, self.logger = make_directories_distributed(**args)
        self.dgm = Flow(**args)
        self.x, self.y = setup_overfit(loader = self.loader, **args)
        setup_wandb_distributed(**args)

    def step_optimizer(self, loss_dict):
        loss_dict['loss'].backward()
        stepped = self.dgm.step_optimizer(
            rank = self.rank,
            steps = self.state.train_steps, 
            log_norm = True
        )
        if stepped:
            self.state.grad_steps += 1
        self.state.train_steps += 1

    def plot_real_data(self,):
        conf = self.config
        x, y = self.x, self.y if conf.debug else next(iter(self.loader))
        steps = self.state.train_steps
        if self.rank == 0 and conf.use_wandb:
            x = self.dgm.process_images_for_wandb(x)
            wandb.log({'real_data': x}, step = steps)
        dist.barrier()

    def prepare_batch(self, x, y):
        conf = self.config
        if conf.debug:
            x, y = self.x, self.y
        x, y = x.to(self.device), y.to(self.device)
        y = maybe_label_drop(conf, y)
        z, _ = self.dgm.encoder.encode(x)
        return x, y, z

    def one_step(self, batch_idx, x, y):
        conf = self.config
        x, y, z = self.prepare_batch(x, y)
        loss_dict = self.dgm.loss_fn(z, y, self.device)
        self.step_optimizer(loss_dict)
        self.state.running_loss += loss_dict['loss'].item()
        self.state.log_steps += 1
        args = {
            'dgm': self.dgm,
            'state': self.state, 
            'config': self.config, 
            'logger': self.logger, 
            'rank': self.rank, 
            'device': self.device,
            'cheap': False,
            'models': self.dgm.models,
        }
        maybe_log(**args)
        maybe_update_emas(**args)
        maybe_sample(name='flow_ode', **args)
        #maybe_sample(name='interpolant_ode', **args)
        #maybe_sample(name='interpolant_sde', **args)
        #maybe_fid(**args)
        maybe_ckpt(**args)

    def one_epoch(self):
        self.sampler.set_epoch(self.state.epoch)
        self.logger.info(f"Beginning epoch {self.state.epoch}...") 
        for batch_idx, (x, y) in enumerate(self.loader):
            self.one_step(batch_idx, x, y)
            
    def training_loop(self,):
        conf = self.config
        #self.sample(use_ema = False, cheap = True) # only a couple of steps to test
        #self.plot_real_data()
        self.logger.info(f"Training! (message from logger)")
        for key in self.dgm.models:
            if 'ema' not in key:
                self.dgm.models[key].train()

        while self.state.train_steps < conf.total_train_steps:
            self.one_epoch()
            self.state.epoch += 1 

# record is important for multi gpu debugging
@record
def _main(conf):  
    spmd_boilerplate()
    trainer = Trainer(conf)
    trainer.training_loop()
    dist.destroy_process_group()

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
    setup_torch_backends(conf)
    
    for key in vars(conf):
        print(key,":",getattr(conf,key))

    _main(conf)


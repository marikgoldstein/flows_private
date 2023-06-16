import torch
import numpy as np
from simple_helpers import Trainer
 
class ExperimentConfig:
    def __init__(self):
 
        # overall 
        self.seed = 1
        self.dgm_type = 'diffusion'
        self.use_wandb = True
        self.dataset = 'cifar'
        self.num_workers = 4
        self.bsz = 128
        self.wd = 0.0 
        self.grad_clip_norm = 2
        self.original_lr = 2e-4
        self.total_steps = 1_000_000
        self.sample_every = 400
        self.skip_sampling = False
        self.n_sample_steps = 500 #1000  # num integration steps
        self.num_sampled_images = 64
       
        # debug
        self.handle_nan_grads = False
        self.dequantize = False #True
        self.clip_samples = False
        self.tweedie = False
        self.T_min = 1e-3
        self.T_max = 1. # - 1e-5
        self.arch = 'bigunet'
        ####################

        self.use_bigunet_fourier = True
        self.bigunet_dropout = 0.0
        # specific to diffusion
        self.max_beta = 20.0
        self.const_beta = 10.0
        self.which_beta = 'linear_beta'
        self.wandb_project = 'var_reduce'
        self.wandb_entity = 'marikgoldstein'
        self.wandb_name = 'test'

if __name__ == '__main__':

    config = ExperimentConfig()
    trainer = Trainer(config)
    trainer.training_loop()


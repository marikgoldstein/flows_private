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

# custom / local
from ncsn import NCSN, NCSNdeeper
from ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from get_sigmas import get_sigmas
from utils import get_optimizer, EMAHelper,  make_cifar_data, dsm, ald 

# consider score matching (no times) for gaussian data as a simple example versus 
# maximum likelihood (use a normalizing flow and really try to overfit the training objective)

class Config:

    def __init__(self,):

        # wandb
        self.use_wandb = True
        self.wandb_project = 'score_match'
        self.wandb_entity = 'marikgoldstein'
        #self.wandb_name = TODO


        # training steps etc
        self.start_epoch = 0
        self.num_epochs = 500_000
        self.total_steps = 300_001
        self.eval_every = 100
        self.ckpt_every = 100
        self.sample_every = 100
        self.print_every = 10
        self.resume_training = False
        self.ckpt_dir = './ckpts'
        self.seed = 1
        
        # optim
        self.train_bsz = 128
        self.lr = 0.0001 # 1e-4
        self.wd = 0.0
        self.optimizer = 'Adam'
        self.beta1 = .9        
        self.optim_eps = 0.00000001
        self.amsgrad = False
      
        # data 
        self.dataset = 'cifar'
        if self.dataset == 'cifar':
            self.C = 3
            self.H = 32
            self.W = 32
            self.d = 32 * 32 * 3
            self.flat_d = self.d
            self.input_dim = self.d
        self.num_workers = 4


        self.logit_transform = False
        self.uniform_dequantization = False
        self.gaussian_dequantization = False
        self.random_flip = True
        self.rescaled = False

        self.training_anneal_power = 2
        self.sampling_bsz = 100
        self.sampling_data_init = False
        self.sampling_step_lr = 0.0000062
        self.sampling_n_steps_each = 5 # 1 #5
        self.sampling_ckpt_id = 300_000
        self.sampling_final_only = True
        self.sampling_fid = False
        self.sampling_denoise = False
        self.sampling_num_samples_fid = 10_000 
        self.sampling_inpainting = False
        self.sampling_interpolation = False
        self.sampling_n_interpolations = 15

        self.test_begin_ckpt = 5000
        self.test_endckpt = 300_000
        self.test_bsz = 100

        self.use_ema = False
        self.ema_rate = 0.999

        self.model_sigma_begin = 50
        self.model_num_classes = 232
        self.model_spec_norm = False
        self.model_sigma_dist = 'geometric'
        self.model_sigma_end = 0.01
        self.model_normalization = 'InstanceNorm++'
        self.model_nonlinearity = 'elu'
        self.model_ngf = 128


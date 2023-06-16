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
from utils import (
    is_type_for_logging,
    requires_grad,
    create_logger,
)
from diffusion import Diffusion
from dit import get_dit
from encoder import Encoder
from config import ExperimentConfig
from torch.distributed.elastic.multiprocessing.errors import record

#clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                        
class ImageLib:

    def __init__(self,):

        self.hello = 'hello'
 
    def make_grid(self, z):

        def assert_BCHW(x, bsz):                               
            assert x.shape == (bsz, 3, 32, 32)                 
                                                               
        def assert_CHW(x):                                     
            assert x.shape == (3, 32, 32)                      
                                                               
        def clamp_unit(x):                                     
            return torch.clamp(x, 0., 1.)                      
                                                               
        def to_255(x):                                         
            return torch.clamp(x * 255., 0., 255.)             
                                                               
        def CHW_TO_HWC(x):                                     
            return x.permute(1,2,0)                            
                                                               
        def to_torch_cpu_uint8(x):                             
            return x.to('cpu', dtype=torch.uint8)              
                                                               
        def fid_pil_preprocess(sample):                        
            assert_CHW(zi)                                     
            sample = to_255(sample)                            
            sample = CHW_TO_HWC(sample)                        
            sample = to_torch_cpu_uint8(sample)                
            sample = sample.numpy()                            
            assert sample.shape[-1] == 3                       
            return sample                                      
                                                               
        samples = z                                            
        samples = torch.clamp(samples, 0. ,1.)                 
        samples = samples.cpu()                                
        bsz = samples.shape[0]                                 
        assert_BCHW(samples, bsz)                              
        grid_size = int(np.floor(np.sqrt(bsz)))                
        samples = samples[:grid_size**2]                       
        grid = torchvision.utils.make_grid(samples, nrow=grid_size)
        return grid



def setup_cifar(trainer):
   
    trainer.config.C = 3
    trainer.config.H = 32
    trainer.config.W = 32
    trainer.config.d = 32 * 32 * 3
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    trainer.train_dataset = datasets.CIFAR10(
            './data',
            train=True,
            download=True,
            transform = train_transform
    )

    trainer.test_dataset = datasets.CIFAR10(
            './data',
            train=False,
            download=True,
            transform = test_transform
    )

    trainer.train_loader = DataLoader(
            trainer.train_dataset,
            batch_size=trainer.config.bsz,
            shuffle=False,
            num_workers=trainer.config.num_workers,
            pin_memory=True,
            drop_last=True
    )

    trainer.test_loader = DataLoader(
            trainer.test_dataset,
            batch_size=trainer.config.bsz,
            shuffle=False,
            num_workers=trainer.config.num_workers,
            pin_memory=True,
            drop_last=True
    )

   


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.config.bsz = 16
        self.config.n_sample_steps = 100
        self.device = torch.device('cuda')
        local_seed = self.config.global_seed
        torch.manual_seed(local_seed)
        setup_cifar(trainer = self)
        self.diffusion = Diffusion(self.config)
        self.encoder = Encoder(self.config)

    def prepare_batch(self, x):
        x = x.to(self.device)
        z0, _ = self.encoder.scaler(x)
        return z0

    def wide(self, x):
        return x[:, None, None, None]

    def sample_and_save(self,):
        
        folder = './test'
        raw = self.sample(n_samples = 16)
        z = self.encoder.generate_x(raw)
        # z here is continuous roughly in [0,1] but not exactly
        grid = ImageLib().make_grid(z)
        save_image(grid, 'grid.png')

    @torch.no_grad()
    def sample(self, n_samples):

        config = self.config
        N = config.n_sample_steps - 1
        n_discrete_steps = N
        
        def empirical_score2(zti, t):
            print("t", t) 
            alpha_t, sigma_t = self.diffusion.get_alpha_sigma(t[0])
            Ntrain = len(self.train_dataset)
            num_points = 20000
            bsz = zti.shape[0]
            alpha = []
            S = []
            assert zti.shape == (bsz, 3, 32, 32)
            for j, data_j in enumerate(self.train_dataset):
                if j >= num_points:
                    break

                xj, yj = data_j
                z0j = self.prepare_batch(xj)
                muj = alpha_t * z0j
                stdj = sigma_t
                varj = stdj.pow(2)

                wj = 1 / num_points

                '''                
                assert muj.shape == (3, 32, 32)
                muj_vec = muj.view(3*32*32,)
                assert stdj.shape == (1,)
                eye=torch.eye(3*32*32)

                lrmvn = torch.distributions.LowRankMultivariateNormal

                dist = lrmvn(
                    loc=muj_vec, 
                    cov_factor = torch.zeros_like(muj_vec)[:,None],
                    cov_diag = torch.ones_like(muj_vec) * stdj
                )
                lp1 = dist.log_prob(zti.view(bsz, 3*32*32))
                assert lp1.shape == (bsz,)


                lp2 = torch.distributions.Normal(
                    loc = muj,
                    scale = stdj
                ).log_prob(zti)
                assert lp2.shape == (bsz, 3, 32, 32)
                lp2 = lp2.sum(-1).sum(-1).sum(-1)
                assert lp2.shape == (bsz,)

                print("all close", torch.allclose(lp1, lp2))
                assert False
                '''

                lp = torch.distributions.Normal(
                    loc = muj,
                    scale = stdj
                ).log_prob(zti)
                assert lp.shape == (bsz, 3, 32, 32)
                lp = lp.sum(-1).sum(-1).sum(-1)
                assert lp.shape == (bsz,)
                
                prob =  lp.exp()
                aj = torch.log(wj * prob)
                aj = torch.nan_to_num(aj, nan=-3e+30, neginf=-3e+30, posinf=3e+30)
                assert aj.shape == (bsz,)
                alpha.append(aj)
                # for MVN Sk = -Sigma_k^{-inv}(x - mu)
                # here, that equals (1/var) * (x - mu) # should be (bsz, dim)
                Sij = -(zti - muj) / varj
                assert Sij.shape == (bsz, 3, 32, 32)
                S.append(Sij)
            
            alpha = torch.stack(alpha, dim=-1)
            S = torch.stack(S, dim=-1)

            assert alpha.shape == (bsz, num_points)

            assert S.shape == (bsz, 3, 32, 32, num_points)
            S = S.view(bsz, 3*32*32, num_points)
            assert S.shape == (bsz, 3*32*32, num_points)

            def notbad(x):
                assert not torch.any(torch.isnan(x))
                assert not torch.any(torch.isinf(x))

            #print("alpha")
            #print(alpha.min(), alpha.mean(), alpha.max())
            notbad(alpha)
            #print("S")
            #print(S.min(), S.mean(), S.max())
            notbad(S)

            Delta = torch.nn.Softmax(dim=-1)(alpha)
            score = (S * Delta[:, None, :])
            assert score.shape == (bsz, 3*32*32,num_points)
            score = score.sum(-1)
            assert score.shape == (bsz, 3*32*32)
            score = score.view(bsz, 3, 32, 32)
            return score

        def reverse_sde(u_t, _t):
            score_hat = empirical_score2(u_t, 1. - _t)
            f, g, g2 = self.diffusion.get_fG(u_t, 1. - _t)
            g2score = self.diffusion.wide(g2) * score_hat
            rev_drift = g2score  - f
            rev_diff = g
            return rev_drift, rev_diff
           
        def one_step_EM(state, state_mean, t_scalar, dt):
            u, u_mean = state, state_mean
            n_samples = u.shape[0]
            noise = torch.randn_like(u)
            t = t_scalar * torch.ones(n_samples).type_as(u)
            drift, diffusion = reverse_sde(u, t)
            u_mean = u + drift * dt
            root_dt = torch.sqrt(dt)
            diff_term = self.diffusion.wide(diffusion) * noise * root_dt
            u = u_mean + diff_term
            return u, u_mean
           
        def main_loop(N, ts, u, u_mean):
            for i in range(N):
                if i % 500 == 0:
                    print("sampling, step {} / {}".format(i, N))
                dt = ts[i + 1] - ts[i]
                u, u_mean = one_step_EM(u, u_mean, ts[i], dt)
            return {'sample': u, 'mean': u_mean}
                  
    
        u_init = self.diffusion.sample_from_prior(n_samples).to(self.device)
        TMIN = config.T_min
        TMAX = config.T_max
    
        ts = torch.linspace(TMIN, TMAX, n_discrete_steps + 1)
        u_init = u_init.to(self.device)
        ts = ts.type_as(u_init)
        with torch.no_grad():
            ret = main_loop(n_discrete_steps, ts, u_init, u_init)
            _, out = one_step_EM(ret['sample'], ret['mean'], ts[-1], torch.tensor([TMIN]).type_as(u_init))
        return out
           

        
    @torch.no_grad()
    def main(self,):
        '''
        ts = torch.linspace(1e-5, 1 - 1e-5, 100)
        ts = torch.flip(ts, dims=(0,))

        for tscalar in ts:

            t = torch.ones(1,) * tscalar
            alpha_t, sigma_t = self.diffusion.get_alpha_sigma(t)
            alpha_t, sigma_t = alpha_t.to(self.device), sigma_t.to(self.device)
            Ntrain = len(self.train_dataset)
            start_whole = time()
            for i, data_i in enumerate(self.test_loader):

                xi, yi = data_i
                z0i = self.prepare_batch(xi)
                bsz = z0i.shape[0]
                mui = alpha_t * z0i
                stdi = sigma_t
                zti = mui + stdi * torch.randn_like(mui)

                numer = torch.zeros_like(zti)
                denom = torch.zeros_like(zti)

                start = time()
                for j, data_j in enumerate(self.train_dataset):
                    xj, yj = data_j
                    z0j = self.prepare_batch(xj)
                    muj = alpha_t * z0j
                    stdj = sigma_t
                    varj = stdj.pow(2)
                    assert muj.shape == z0j.shape
                    wj = 1 / Ntrain
                    
                    # numer
                    A = wj
                    B = (zti - muj) / (np.sqrt(2 * np.pi) * stdj.pow(3))
                    C = torch.exp(-.5 * ((zti - muj).pow(2) / varj ))
                    numer += -(A*B*C)

                    # denom
                    B = 1 / torch.sqrt(2 * np.pi * varj)
                    denom += (A*B*C) # same A,C, different B, no negative
                
                score_zti = numer / denom
                print("numer shape", numer.shape)
                print("denom shape", denom.shape)
                print("score zti shape", score_zti.shape)
                print("zti shape", zti.shape)
                end = time()
                elapsed = end - start
                print("time for this one eval batch", elapsed)
            end_whole = time()
            elapsed_whole = end_whole - start_whole
            print("done with whole val set")
            print("elapsed total time", elapsed_whole)
            assert False
        '''
if __name__=='__main__':


    #os.environ['CUDA_LAUNCH_BLOCKING']='1'

    parser = argparse.ArgumentParser(description='hello')
    parser.add_argument('--dgm_type', type=str, default='diffusion')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--do_resume', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--use_wandb', type=int, default=1)
    parser.add_argument('--results_dir', type=str, default='./ckpts')
    parser.add_argument('--load_ckpt_name', type=str, default='demowtoshdiudhg')
    parser.add_argument('--print_hp', type=int, default=0)
    parser.add_argument('--s_denom', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='mnist')
    #parser.add_argument('--local_world_size', type=int, default=1)

    # just the command line top level args, dgm_type, index
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
    trainer.sample_and_save()



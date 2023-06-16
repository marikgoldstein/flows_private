import time                                                                                 
from typing import Optional, List                                                            
import torch                                                                                
import torch.nn as nn                                                                       
import time                                                
from typing import Optional, List
from torchvision import datasets, transforms
import numpy as np
import math
import wandb
import argparse
from itertools import product, starmap
from collections import namedtuple
from torch import optim
from typing import Optional, List
from tqdm.auto import tqdm
import os, sys
import pickle
import sys
from functorch import vmap
from math import pi
import time

# local
from models_bigunet import get_big_unet
import copy
from utils import check, param_no_grad
from torch.distributions import Normal
from torchvision import datasets, transforms                                                
import torchvision                                                                          
from functools import partial                                                               
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                            
Uniform = torch.distributions.Uniform                                                       
from torch.utils.data import DataLoader                                                     
from torch.utils import data                                                                
from torchvision import datasets, transforms, utils                                         
from torchvision.transforms import functional as TF  
from utils_numerical import (
    randn, cat, ones, ones_like, eye, linspace, stack
)        
from utils_numerical import (
    cat, chunk, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    eye,
    trace_fn
)

Adam = torch.optim.Adam                                                                     
AdamW = torch.optim.AdamW                                                                   
Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal
ones_like = torch.ones_like
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                        
     
class CenterTransform(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
         
    def forward_transform(self, x, logpx=None):
        # Rescale from [0, 1] to [-1, 1]
        y = x * 2.0 - 1.0
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1)
         
    def reverse(self, y, logpy=None, **kwargs):
        # Rescale from [-1, 1] to [0, 1]
        x = (y + 1.0) / 2.0
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).view(x.size(0), -1).sum(1)
         
    def _logdetgrad(self, x):
        return (ones_like(x) * 2).log()
         
    def __repr__(self):
        return "{name}({alpha})".format(name=self.__class__.__name__, **self.__dict__)
         

class Encoder(nn.Module):
         
    def __init__(self, config):
        super().__init__()
        self.dataset = config.dataset
        self.is_image = True
        self.C = config.C
        self.H = config.H
        self.W = config.W
        self.dimx = self.C * self.W * self.H
        self.transform = CenterTransform()
        self.dequantize = config.dequantize
    
    def nelbo_to_bpd(self, nelbo, ldj):
        elbo = -nelbo
        elbo += ldj
        elbo_per_dim = elbo / self.dimx
        nelbo_per_dim = -elbo_per_dim
        nelbo_per_dim_log2 = nelbo_per_dim / np.log(2)
        offset = 8.0 if self.dequantize else 0.0
        bpd = nelbo_per_dim_log2 + offset
        return bpd
         
    def forward(self, x):
        assert False
         
    def uniform_dequantization(self, batch):
        return (batch * 255 + torch.rand_like(batch)) / 256
         
    def generate_x(self, u):
        bsz = u.shape[0]
        u = self.inverse_scaler(u) # "from [-1,-1] to [0,1]" #
        for_wandb = torch.clamp(u * 255.0, min=0.0,max=255.0).byte()
        return {'for_wandb' : for_wandb}
         
    def scaler(self, x):
        assert x.min() >= 0.
        assert x.max() <= 1.
        return self.transform.forward_transform(x, 0)
         
    def inverse_scaler(self, x):
        y = self.transform.reverse(x)
        return y
         
    def preprocess(self, batch):
        x = batch
        #bsz, C, H, W = x.shape
        #x = x.reshape(bsz, C*H*W)
        if self.dequantize:
            x = self.uniform_dequantization(x)
        x, ldj = self.scaler(x) # to [-1, 1]
        return x, ldj

class BModule(nn.Module):
 
    def __init__(self, b0, b1, bconst, which_beta):
        super().__init__()
        assert which_beta in ['linear_beta', 'const_beta', 'linear_mean_coef_squared']
        self.which_beta = which_beta
        self._b0 = nn.Parameter(torch.tensor([b0]), requires_grad=False)
        self._b1 = nn.Parameter(torch.tensor([b1]), requires_grad=False)
        self.bconst = bconst
 
    def get_b0_b1(self,):
        return self._b0, self._b1
 
    def beta_fn(self, t):
        if self.which_beta == 'linear_beta':
            return self.linear_beta_fn(t)
        elif self.which_beta ==  'linear_mean_coef_squared':
            return self.linear_mean_coef_squared(t)
        elif self.which_beta == 'const_beta':
            return self.const_beta_fn(t)
        else:
            assert False
 
    def int_beta_fn(self, t):
        if self.which_beta == 'linear_beta':
            return self.int_linear_beta_fn(t)
        elif self.which_beta == 'linear_mean_coef_squared':
            return self.int_linear_mean_coef_squared(t)
        elif self.which_beta == 'const_beta':
            return self.int_const_beta_fn(t)
        else:
            return False
 
    def const_beta_fn(self, t):
        return self.bconst * torch.ones_like(t)
 
    def int_const_beta_fn(self, t):
        return self.bconst * t
 
    # beta fn 2
    def linear_mean_coef_squared(self, t):
        return  1. / (1-t)
 
    def int_linear_mean_coef_squared(self, t):
        return -(1-t).log()
 
    # beta fn 1
    def linear_beta_fn(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0 + t*(b1 - b0)
 
    def int_linear_beta_fn(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0*t + (b1-b0)*(t**2/2)
 

class VP(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.d = config.flat_d
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.which_beta = config.which_beta
        self.device = config.device
        self.prior_loc = param_no_grad(zeros(1).to(config.device)) 
        self.prior_scale = param_no_grad(ones(1).to(config.device)) 
        self.bmodule = BModule(b0=0.1, b1=config.max_beta, bconst = config.const_beta, which_beta = config.which_beta)
 
    def forward(self, u, t, s = None):
        D = self.sample_from_transition_kernel(u, t, s = s)
        return D
 
    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)
 
    def prior_logp(self, u):
        return self.get_prior_dist().log_prob(u).sum(-1)
 
    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, 3, 32 ,32)
 
    def beta_fn(self,t):
        return self.bmodule.beta_fn(t)
 
    def int_beta_fn(self,t):
        return self.bmodule.int_beta_fn(t)
 
    def get_fG(self, u , t):
        return self.f(u, t), self.G(t), self.G2(t)
 
    # output bsz by D
    def f(self, u, t):
        return -0.5 * self.beta_fn(t)[...,None, None, None] * u
 
    # output bsz,
    def G(self, t):
        return sqrt(self.beta_fn(t))
 
    # output bsz,
    def G2(self, t):
        return self.beta_fn(t)
 
    # output bsz,
    def div_f(self, u, t):
         return -0.5 * self.beta_fn(t) * self.d
 
    # output bsz
    def transition_mean_coefficient(self, t, s=None):
        bt = self.int_beta_fn(t)
        bs = 0.0 if s is None else self.int_beta_fn(s)
        return torch.exp(-0.5 * (bt - bs))
 
    def transition_mean(self, u, t, s=None):
        coef = self.transition_mean_coefficient(t, s=s)
        return coef[:,None, None, None] * u, coef
 
    # output bsz,
    def transition_var(self, t, s=None):
        bt = self.int_beta_fn(t)
        bs = 0.0 if s is None else self.int_beta_fn(s)
        return 1 - torch.exp(-(bt-bs))
 
    # output bsz,
    def transition_std(self, t, s=None):
        return sqrt(self.transition_var(t, s=s))
 
    def sample_from_transition_kernel(self, u, t, s=None):
        bsz = u.shape[0]
        mean, mean_coef = self.transition_mean(u, t, s=s)
        std = self.transition_std(t, s=s)
        var = std.pow(2)
        noise = torch.randn_like(u)
        u_t = mean + noise * std[:, None, None, None]
        if s is None:
            s = torch.zeros_like(t)
        return {'u_0': u, 'u_t':u_t, 'noise':noise, 'mean_coef':mean_coef, 'std':std, 'var': var, 'mean':mean, 't': t, 's': s}
 
    # ouput bsz,
    def cross_ent_helper(self, u_0):
        bsz = u_0.shape[0]
        T = ones(bsz).type_as(u_0) * self.T_max
        u_T = self.sample_from_transition_kernel(u_0, T, s=None)['u_t']
        lp = self.prior_logp(u_T)
        assert lp.shape==(bsz,)
        return lp

class DataObj:

    def __init__(self, config):

        config.img_size = 32
        config.channels = 3
        config.flat_d = config.img_size * config.img_size * config.channels
        train_transform = transforms.Compose([                                                     
                transforms.ToTensor(),                                                      
                #transforms.RandomHorizontalFlip(),
        ])                                                                                  
        test_transform = transforms.Compose([transforms.ToTensor()])                               
        dset_fn = datasets.CIFAR10
        trainset = dset_fn('./data', train=True, download=True, transform=train_transform)
        testset = dset_fn('./data', train=False, download=True, transform=test_transform)
        train_kwargs = {'batch_size': config.bsz, 'shuffle': True, 'drop_last': True}
        test_kwargs = {'batch_size': config.bsz, 'shuffle': False}
        if config.use_cuda:                  
            cuda_kwargs = {           
                'num_workers': config.num_workers,       
                'pin_memory': True,   
                'persistent_workers': True        
            }                         
            train_kwargs.update(cuda_kwargs)      
            test_kwargs.update(cuda_kwargs)       
        self.train_loader = DataLoader(trainset, **train_kwargs)
        self.test_loader = DataLoader(testset, **test_kwargs)
        config.d = config.flat_d     
        config.is_image = True
        config.C = config.channels 
        config.H = config.img_size 
        config.W = config.img_size          

class ModelObj:
    
    def __init__(self, config):
        self.config = config
        self.model = get_big_unet(self.config)
        self.model.to(self.config.device)
        self.train_mode()
        self.opt = AdamW(
            self.model.parameters(), 
            lr=self.config.original_lr, 
            weight_decay=self.config.wd
        )
        self.total_params = self.count_params()
        print("total params", self.total_params)

    def count_params(self,):
        total = 0
        for n, p in self.model.named_parameters():
            total += p.numel()
        return total

    def __call__(self, u, t):
        return self.model(u, t)

    def train_mode(self,):
        self.model.train()
    
    def eval_mode(self,):
        self.model.eval()

    def maybe_handle_nan_grads(self,):
        if not self.config.handle_nan_grads:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, 
                    nan=0, 
                    posinf=1e5, 
                    neginf=-1e5, 
                    out=param.grad
                )

    def maybe_clip_grads(self):
        gcn = self.config.grad_clip_norm
        if gcn is None:
            gcn = np.inf
        norm = clip_grad_norm_(self.model.parameters(), max_norm = gcn)
        return norm
    
    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.maybe_handle_nan_grads()
        grad_norm = self.maybe_clip_grads()
        self.opt.step()
        return grad_norm


def EM(trainer, n_samples):

                                                                                                                                                                  
    conf = trainer.config
    sde = trainer.sde
    prior_sample_fn = sde.sample_from_prior
    T_min = conf.T_min
    T_max = conf.T_max
    n_discrete_steps = conf.n_sample_steps - 1
    device = conf.device
    clip_samples = conf.clip_samples
    tweedie = conf.tweedie
 
    def reverse_sde(u_t, _t, probability_flow=False):
 
        batch_size = u_t.shape[0]
        rev_t = 1 - _t
        D = {'u_t':u_t, 't': rev_t}
        std = trainer.sde.transition_std(rev_t, s=None)
        var = std.pow(2)
        mean_coef = trainer.sde.transition_mean_coefficient(rev_t, s=None)
        D['std'] = std
        D['var'] = var
        D['mean_coef'] = mean_coef
        score_hat = trainer.score_pred(D)
        f, g, g2 = sde.get_fG(u_t, rev_t)
        g2score = g2[:,None, None, None] * score_hat
        rev_drift = g2score * (0.5 if probability_flow else 1.0) - f
        rev_diff = zeros_like(g) if probability_flow else g
        return rev_drift, rev_diff
 
    def one_step_EM(t_scalar, dt, u, u_mean):
        n_samples = u.shape[0]
        eps = randn_like(u).type_as(u)
        t = t_scalar * ones(n_samples).type_as(u)
        drift, diffusion = reverse_sde(u, t)
        u_mean = u + drift * dt
        root_dt = torch.sqrt(dt)
        u = u_mean + (diffusion[:,None, None, None] * eps * root_dt)
        return u, u_mean
 
    # aggressive
    def clip_func(x):
        return x.clamp(min=-1, max=1)
 
    def get_sampler_t_array(N, T_begin, T_final):
        t = linspace(T_begin, T_final, N + 1)
        #t = T_final * flip(1 - (t / T_final) ** 2.0, dims=[0]) # quad
        #t = t.clamp(T_begin, T_final)
        return t
 
    def main_loop(N, ts, u, u_mean, clipping):
        for i in range(N):
            if i % 500 == 0:
                print("sampling, step {} / {}".format(i, N))
            dt = ts[i + 1] - ts[i]
            u, u_mean = one_step_EM(ts[i], dt, u, u_mean)
            if clipping:
                u, u_mean = clip_func(u), clip_func(u_mean)
        return u, u_mean
  
    def tweedie_step_fn(u_eps, u_eps_mean, T_min, clipping):
 
        assert T_min <= 1e-3
        T_min_tensor = torch.tensor([T_min]).type_as(u_eps)
 
        bsz = u_eps.shape[0]
        eps = T_min_tensor
        eps_vec = eps * torch.ones(bsz,).type_as(u_eps)
 
        std = trainer.sde.transition_std(eps, s=None)
        var = std.pow(2)
        mean_coef = trainer.sde.transition_mean_coefficient(eps, s=None)
 
        # get stheta
        score_pred_args = {
            'D': {'u_t': u_eps, 't': eps_vec, 'std': std, 'var': var, 'mean_coef': mean_coef}
        }
        score_hat = trainer.score_pred(**score_pred_args)
 
        #N(x | \frac{x'}{a} + \frac{beta^2}{a}s_\theta(x', eps), variance = \frac{beta^2}{a^2} I). a is mean coef, beta^2 is var
        mu_term1 = u_eps / mean_coef[:, None, None, None]
        mu_term2 = (var / mean_coef)[:, None, None, None] * score_hat
        mu = mu_term1 + mu_term2
        # only need conditional mean
        #variance = var / mean_coef.pow(2)
        #sigma = variance.sqrt()
        return mu
 
    def denoising_step_fn(u_eps, u_eps_mean, T_min, T_final, clipping):
        # t = t final = .9999 (since it will be reversed)
        T_min_tensor = torch.tensor([T_min]).type_as(u_eps)
        u_0, u_0_mean = one_step_EM(T_final, T_min_tensor, u_eps, u_eps_mean)
        if clipping:
            u_0, u_0_mean = clip_func(u_0), clip_func(u_0_mean)
        return u_0, u_0_mean
 
    print("Sampling")
    u_init = prior_sample_fn(n_samples)
    u_init = u_init.to(trainer.config.device) # should not be needed but just in case
    T_begin = 1.0 - T_max
    T_final = 1.0 - T_min
    ts = get_sampler_t_array(N=n_discrete_steps, T_begin = T_begin, T_final = T_final)
    u_init = u_init.to(device)
    ts = ts.type_as(u_init)
    with torch.no_grad():
        u_eps, u_eps_mean = main_loop(n_discrete_steps, ts, u_init, u_init, clipping = clip_samples)
        if tweedie:
            u_0_mean = tweedie_step_fn(u_eps, u_eps_mean, T_min = T_min, clipping = clip_samples)
        else:
            _, u_0_mean = denoising_step_fn(u_eps, u_eps_mean, T_min = T_min, T_final = T_final, clipping = clip_samples)
 
    return u_0_mean
 
def generate_image_grid(images):
    """Simple helper to generate a single image from a mini batch."""
    images = images.cpu()
    batch_size = images.shape[0]
    grid_size = int(np.floor(np.sqrt(batch_size)))
    images = images[0:grid_size**2]
    grid = torchvision.utils.make_grid(images, nrow=grid_size)
    grid = grid.permute(1,2,0)
    return grid.numpy()
 
def im_2_wandb(samples):
    samples = samples[None, ...]
    samples = [wandb.Image(np.array(x)) for x in samples]
    return samples

class Trainer:

    def __init__(self, config):
    
        self.config = config

        self.wandb_run = wandb.init(
	    project=self.config.wandb_project,
	    entity=self.config.wandb_entity,
	    name = self.config.wandb_name,
        )


        self.config.use_cuda = torch.cuda.is_available()
        if self.config.use_cuda:
            self.config.device = torch.device('cuda')
        else:
            self.config.device = torch.device('cpu')
        
        self.data_obj = DataObj(config)
        self.sde = VP(config)
        self.model_obj = ModelObj(config)
        self.encoder = Encoder(config)
        self.step = 0

    # main step
    def _step(self, batch):        
        self.model_obj.train_mode()  
        loss = self.loss_fn(batch)
        grad_norm = self.model_obj.step(loss)
        self.step += 1
        if self.step % 100 == 0:
            print("Loss is : {}".format(loss.item()))
    
    def get_times(self, bsz):
        return Uniform(low=self.config.T_min, high=1.0).sample(sample_shape=(bsz,)).squeeze(-1).to(self.config.device)

    def loss_fn(self, batch):
        u_0, ldj = self.encoder.preprocess(batch)
        bsz = u_0.shape[0]
        t = self.get_times(bsz)
        D = self.sde(u_0, t)
        eps_hat = self.noise_pred(D)
        noise_diff = (D['noise'] - eps_hat).pow(2).sum(-1).sum(-1).sum(-1)
        assert noise_diff.shape == (bsz,) 
        return (0.5 * noise_diff).mean()

    def noise_pred(self, D):
        return self.model_obj(D['u_t'], D['t'])
        
    def score_pred(self, D):
        noise_hat = self.model_obj(D['u_t'], D['t'])
        score_hat = -noise_hat / D['std'][:, None, None, None]
        return score_hat

    def training_loop(self,):

        self.sampling()
        while self.step <= 1000000:

            for batch_idx, (batch, _) in enumerate(self.data_obj.train_loader):
                self._step(batch.to(self.config.device))
            
                if self.step % self.config.sample_every == 0:
                    self.sampling()

    @torch.no_grad()
    def sampling(self,):
        self.model_obj.eval_mode()
        num_images = self.config.num_sampled_images
        raw = EM(trainer = self, n_samples = self.config.num_sampled_images)
        D = self.encoder.generate_x(raw)
        model_samples = D['for_wandb']
        model_samples = generate_image_grid(model_samples)
        if self.config.use_wandb:
            model_samples = im_2_wandb(model_samples)
            key = 'model_samples'
            wandb.log({key : model_samples}, step = self.step)


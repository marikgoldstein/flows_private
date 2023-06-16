import math
import time 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Optional, List
from tqdm.auto import tqdm

# local
#from importance_sampling import VPImpSamp
from utils_numerical import (
    cat, chunk, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    eye,
    batch_transpose,
    trace_fn
)
from utils import check, param_no_grad
from torch.distributions import Normal

Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal
ones_like = torch.ones_like
atan = torch.atan
tan = torch.tan

class VP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.d = config.d
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.prior_loc = 0.0
        self.prior_scale = 1.0

    def get_alpha_sigma(self, t):
	alpha_t = torch.sqrt(torch.sigmoid(self.logsnr(t)))
	sigma_t = torch.sqrt(torch.sigmoid(-self.logsnr(t)))
        return alpha_t, sigma_t

    def diffuse(self, x, alpha_t, sigma_t):
	noise = torch.randn_like(x)
	mean = alpha_t * x
	z = mean + sigma_t * noise
	return z, mean, noise

    def forward(self, x, t, s = None):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
	z_t , mu_t, eps_t = self.diffuse(x, alpha_t , sigma_t )
	return {'u_t': z_t, 'noise': noise, 'mean': mu_t, 'mean_coef': alpha_t, 'std': sigma_t, 'var': sigma_t.pow(2), 't': t}
  
    def zv_to_x(self, alpha, sigma, z, v):
        return alpha * z - sigma * v

    def zv_to_eps(self, alpha, sigma, z, v):
        return sigma * z + alpha * v
    
    def atanexp(self, t):
        return atan(torch.exp(t))

    def tmin_tmax(self, dummy, logsnr_min = -15, logsnr_max = 15):
        t1 = torch.tensor([-0.5 * logsnr_max]).type_as(dummy)
        t0 = torch.tensor([-0.5 * logsnr_min]).type_as(dummy)
        t_min = self.atanexp(t1)
        t_max = self.atanexp(t0)
        return t_min, t_max

    # cos
    def logsnr(self, t , logsnr_min = -15 , logsnr_max =15):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return -2 * torch.log(tan( _t))

    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)

    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, 3, 32 ,32)



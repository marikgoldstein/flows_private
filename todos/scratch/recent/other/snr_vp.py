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
        #self.bmodule = BModule(b0=0.1, b1=config.max_beta, bconst = config.const_beta, which_beta = config.which_beta)
        #t = noise_uniform ( size = x . shape [0]) # Sample a batch of timesteps .
        #logsnr_t = logsnr_schedule ( t )
        #alpha_t = sqrt ( sigmoid ( logsnr ) )
        #sigma_t = sqrt ( sigmoid ( - logsnr ) )
        #z_t , eps_t = diffuse (x , alpha_t , sigma_t )
        #v_pred = uvit ( z_t , logsnr_t )
        #eps_pred = sigma_t * z_t + alpha_t * v_t
        #return mse ( eps_pred , eps_t )

    def tmin_tmax(self, dummy, logsnr_min = -15, logsnr_max = 15):
        t_min = atan(torch.exp(torch.tensor([-0.5 * logsnr_max]).type_as(dummy)))
        t_max = atan(torch.exp(torch.tensor([-0.5 * logsnr_min]).type_as(dummy)))
        return t_min, t_max

    #meancoef cos(pi t / 2), 
    #std = sin(pi t / 2). 
    #snr alpha/sig = 1/tan(pi t/2)
    #log a/sig = -logtan(pit/2).  
    #meancoef = root sig log snr. 
    #sig = sqrt sig -logsnr.

    # cos
    def logsnr(self, t , logsnr_min = -15 , logsnr_max =15):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return -2 * torch.log(tan( _t))

    def snr(self, t, logsnr_min=-15, logsnr_max=15):
        #exp[log snr]
        #exp[ -2 log(tan(_t)) ]
        #exp[log(tan(_t))]^2
        #tan(_t)^2
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return tan(_t).pow(2)

    #def beta_fn(self, t, logsnr_min = -15, logsnr_max = 15):
        #meancoef = root(sig(logsnr))
        #meancoesquared = sig(logsnr) = exp[- int beta]
        #log sig log snr = - int beta
        #- log sig log snr = int beta
        #beta = - 1/(sig log snr) * ddt sig log snr
        #     = - 1/(sig log snr) * (sig log snr) * (1 - sig log snr)  * ddt log snr
        #     = -(1 - sig log snr) / (snr) * (d/dt snr)
        #    = (-1/snr  + siglogsnr/snr) * ddtsnr 
    #    return ((-1 / self.snr(t)) + self.logsnr(t).sigmoid()) * self.ddt_snr(t)

    # int_0^t beta(s) ds = B(t) - B(0)
    def int_beta_fn(self, t):
        #return - torch.nn.LogSigmoid()(self.logsnr(t))
        return -self.logsnr(t).sigmoid().log()

    def beta_fn(self, t, logsnr_min = -15, logsnr_max = 15):
        '''
        - d/dt log sig  logsnr
        (-1/sig log snr) * ddt siglogsnr
        (-1/sig log snr) * (siglogsnr) * (1-siglogsnr) * d/dt logsnr

        -(1-siglogsnr) ddt logsnr

        ddt logsnr = -2 ddt log tan _t 
                   = -2/tan(_t) * ddt tan _t
                   = -2/tan(_t) * (1/cos^2(_t)) ddt _t
                   = -2/tan(_t) * (1/cos^2(_t)) width
    
        -1 * (1-siglogsnr) * -2/tan(_t) * (1/cos^2(_t)) * width
        (1-siglogsnr) * 2/tan(_t) * (1/cos^2(_t)) * width
        (2 * (1-siglogsnr) * width) / (tan(_t) * _t.cos().pow(2))
        '''
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        numer = 2 * (1 - self.logsnr(t).sigmoid()) * width
        denom = tan(_t) * _t.cos().pow(2)
        return numer / denom


    def logsnr(self, t , logsnr_min = -15 , logsnr_max =15):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return -2 * torch.log(tan( _t))




    def ddt_snr(self, t, logsnr_min=-15 ,logsnr_max=15):
        # ddt tan t = 1/cos^2(t) = sec^2(t)
        #ddt tan(_t).pow(2) 
        #= 2 tan(_t) * ddt tan(_t)
        #= 2 tan(_t) * (1/cos^2(_t)) * ddt _t
        #= \frac{2 tan(_t)}{cos^2(_t)} * (tmax - t_min)
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return 2 * tan(_t) / _t.cos().pow(2) * width

    def transition_mean_coefficient2(self, t, s=None):
        return torch.exp(-0.5 * self.int_beta_fn(t))

    def transition_std2(self, t, s=None):
        return torch.sqrt(1 - torch.exp(- self.int_beta_fn(t)))

    def transition_mean_coefficient(self, t, s=None):
        return torch.sqrt(torch.sigmoid(self.logsnr(t)))

    def transition_std(self, t, s=None):
        return torch.sqrt(torch.sigmoid(-self.logsnr(t)))
    
    def f(self, u, t):
        return -0.5 * self.beta_fn(t)[...,None, None, None] * u

    def G(self, t):
        return sqrt(self.beta_fn(t))

    def forward(self, u, t, s = None):
        #{'u_t':u_t, 'eps':eps, 'mean_coef':mean_coef, 'std':std, 'mean':mean}
        D = self.sample_from_transition_kernel(u, t, s = s)
        return D

    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)

    def prior_logp(self, u):
        return self.get_prior_dist().log_prob(u).sum(-1).sum(-1).sum(-1)

    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, 3, 32 ,32)

    def get_fG(self, u , t):
        return self.f(u, t), self.G(t), self.G2(t)

    # output bsz,
    def G2(self, t):
        return self.beta_fn(t)

    # output bsz,
    def div_f(self, u, t):
         return -0.5 * self.beta_fn(t) * self.d

    def transition_mean(self, u, t, s=None):
        coef = self.transition_mean_coefficient(t, s=s)
        return coef[:,None, None, None] * u, coef

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

if __name__=='__main__':

    bsz=1
    vp = VP(d=2)
    xinit = torch.randn(bsz,2)
    t = torch.ones(bsz,) * 0.5
    s = torch.ones(bsz,) * 0.25
    xt = vp.sample_from_transition_kernel(xinit, t, s=None)
    xts = vp.sample_from_transition_kernel(xinit, t, s=s)
    print(xt)
    print(xts)
    #vp = VP(d=1, max_diffusion_noise = 20.0, T_min = 1e-5, T_max = 1.0, device=torch.device('cpu'))


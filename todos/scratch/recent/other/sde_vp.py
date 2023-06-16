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
        self.d = config.d
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.prior_loc = 0.0
        self.prior_scale = 1.0
        self.bmodule = BModule(b0=0.1, b1=config.max_beta, bconst = config.const_beta, which_beta = config.which_beta)

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
        #return p.sample((n_samples, self.d)).view(n_samples, -1)
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

'''
class VPImpSamp:
    def __init__(self, sde ):
        self.sde = sde
        self.beta_0 = torch.tensor([sde.b0])
        self.beta_1 = torch.tensor([sde.b1])
        self.t_max = torch.tensor([sde.T_max])
        self.t_min = torch.tensor([sde.T_min])
        self.device = self.sde.device

    # check clamping
    def sample(self, n_samples):
        t_samples = self._sample(shape=n_samples, eps=self.t_min)
        return t_samples.clamp(min=self.t_min, max=self.t_max)

    def cumulative_weight(self, t, eps):

        exponent1 = 0.5 * eps * (eps - 2) * self.beta_0 - 0.5 * eps**2 * self.beta_1
        exponent2 = 0.5 * t * (t - 2) * self.beta_0 - 0.5 * t**2 * self.beta_1
        term1 = torch.where(
            torch.abs(exponent1) <= 1e-3, -exponent1, 1.0 - torch.exp(exponent1)
        )
        term2 = torch.where(
            torch.abs(exponent2) <= 1e-3, -exponent2, 1.0 - torch.exp(exponent2)
        )
        return 0.5 * (
            -2 * torch.log(term1)
            + 2 * torch.log(term2)
            + self.beta_0 * (-2 * eps + eps**2 - (t - 2) * t)
            + self.beta_1 * (-(eps**2) + t**2)
        )

    def _sample(self, shape, eps, quantile=None, steps=100):
        Z = self.cumulative_weight(self.t_max, eps=eps)
        if quantile is None:
            quantile = torch.rand(shape) * (Z - 0) + 0
        lb = ones_like(quantile) * eps
        ub = ones_like(quantile) * self.t_max

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.0
            value = self.cumulative_weight(mid, eps=eps)
            lb = torch.where(value <= quantile, mid, lb)
            ub = torch.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        carry = (lb, ub)
        for i in range(steps):
            carry, _ = bisection_func(carry, i)
        (lb, ub) = carry
        return (lb + ub) / 2.0

    def r(self, t):
        ratio = self.sde.g2(t) / self.sde.transition_var(t, s=None)
        return ratio

    def Z(self,):
        return self.cumulative_weight(t=self.t_max, eps=self.t_min).to(self.device)

    def weight(self, t):
        return self.r(t) / self.Z()
'''




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




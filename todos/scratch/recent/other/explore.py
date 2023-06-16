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
        #b0 = 0.0001
        #b1 = 0.02
        #lambda t: math.cos((t * 0.008) / 1.0008 * math.pi / 2) ** 2

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
    
    def __init__(self, d, max_beta, const_beta, which_beta, device): #T_min, T_max, which_beta, device):
        super().__init__()
        self.d = d
        #self.T_min = T_min
        #self.T_max = T_max
        self.which_beta = which_beta
        self.device = device
        self.prior_loc = param_no_grad(zeros(1).to(device)) #nn.Parameter(zeros(1).to(device),requires_grad=False)
        self.prior_scale = param_no_grad(ones(1).to(device)) #nn.Parameter(ones(1).to(device), requires_grad=False)
        self.b0 = 0.1
        self.b1 = max_beta
        self.bconst = const_beta
        self.bmodule = BModule(b0=self.b0, b1=self.b1, bconst = const_beta, which_beta = which_beta)

    def forward(self, u, t, s = None):
        #{'u_t':u_t, 'eps':eps, 'mean_coef':mean_coef, 'std':std, 'mean':mean}
        D = self.sample_from_transition_kernel(u, t, s = s)
        return D

    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)

    def prior_logp(self, u):
        return self.get_prior_dist().log_prob(u).sum(-1)

    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, -1)

    def beta_fn(self,t):
        return self.bmodule.beta_fn(t)

    def int_beta_fn(self,t):
        return self.bmodule.int_beta_fn(t)

    def get_fG(self, u , t):
        return self.f(u, t), self.G(t), self.G2(t)

    # output bsz by D
    def f(self, u, t):
        return -0.5 * self.beta_fn(t)[...,None] * u

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
        return coef[:,None] * u, coef

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
        u_t = mean + noise * std[:, None]
        if s is None:
            s = torch.zeros_like(t)
        return {'u_0': u, 'u_t':u_t, 'noise':noise, 'mean_coef':mean_coef, 'std':std, 'var': var, 'mean':mean, 't': t, 's': s}
    
    # ouput bsz,
    def cross_ent_helper(self, u_0):
        bsz = u_0.shape[0]
        T = ones(bsz).type_as(u_0)
        u_T = self.sample_from_transition_kernel(u_0, T, s=None)['u_t']
        lp = self.prior_logp(u_T)
        assert lp.shape==(bsz,)
        return lp

if __name__=='__main__':
    import matplotlib.pyplot as plt
    bsz=1
    vp = VP(d=1, max_beta=10.0, const_beta=10.0, which_beta='const_beta', device=torch.device('cpu'))
    xinit = torch.randn(bsz,1) + 100.0
    zero = torch.zeros(bsz,)
    t = torch.ones(bsz,) * 0.5
    delta = .1
    for s in [t-(.1/50), t-(.1/10), t-(.1/2), t-(.1/1.1)]:
        print("t, s", t[0], s[0])
        plt.scatter(zero.numpy(), xinit.numpy(), label='x0')
        s = torch.where(t > delta, s, torch.zeros_like(s))
        xs = vp.sample_from_transition_kernel(xinit, s, s=None)['u_t']
        xt = vp.sample_from_transition_kernel(xs, t, s=s)['u_t']
        print(xs)
        print(xt)
        plt.scatter(s.numpy(), xs.numpy(), label='xs')
        plt.scatter(t.numpy(), xt.numpy(), label='xt')
        plt.legend()
        plt.show()

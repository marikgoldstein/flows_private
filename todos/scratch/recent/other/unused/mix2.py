import math
import time 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Optional, List
from tqdm.auto import tqdm
import seaborn as sns
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
import torch.distributions as D
import matplotlib.pyplot as plt 
from statsmodels.nonparametric.kde import KDEUnivariate

class BModule(nn.Module):
    
    def __init__(self, b0, b1, which):
        super().__init__()
        assert which in [1, 2]
        self.which = which
        self._b0 = nn.Parameter(torch.tensor([b0]), requires_grad=False)
        self._b1 = nn.Parameter(torch.tensor([b1]), requires_grad=False)
        self.const = .1
        #b0 = 0.0001
        #b1 = 0.02
        #lambda t: math.cos((t * 0.008) / 1.0008 * math.pi / 2) ** 2

    def get_b0_b1(self,):
        return self._b0, self._b1

    def beta_fn(self, t):
        if self.which == 1:
            return self.beta_fn1(t)
        else:
            return self.beta_fn2(t)

    def int_beta_fn(self, t):
        if self.which == 1:
            return self.int_beta_fn1(t)
        else:
            return self.int_beta_fn2(t)

    def beta_fn2(self, t):
        return  1. / (1-t)

    def int_beta_fn2(self, t):
        return -2 * (1-t).sqrt().log()

    def beta_fn1(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0 + t*(b1 - b0)
        #return self.const * torch.ones_like(t)

    def int_beta_fn1(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0*t + (b1-b0)*(t**2/2)
        #return t

class VP(nn.Module):
    
    def __init__(self, d, max_beta, which_beta, device):
        super().__init__()
        self.d = d
        self.which_beta = which_beta
        self.device = device
        self.max_noise = max_beta
        self.prior_loc = param_no_grad(zeros(1).to(device)) #nn.Parameter(zeros(1).to(device),requires_grad=False)
        self.prior_scale = param_no_grad(ones(1).to(device)) #nn.Parameter(ones(1).to(device), requires_grad=False)
        self.bmodule = BModule(b0=.1, b1=self.max_noise, which = which_beta)
    
    def forward(self, u, t, s = None):
        return self.sample_from_transition_kernel(u, t, s = s)

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
        noise = torch.randn_like(u)
        u_t = mean + noise * std[:, None]
        return {'u': u, 'u_t':u_t, 'noise':noise, 'mean_coef':mean_coef, 'std':std, 'var': std.pow(2), 'mean':mean, 't': t, 's': s}
    
    # ouput bsz,
    def cross_ent_helper(self, u_0):
        bsz = u_0.shape[0]
        T = ones(bsz).type_as(u_0)
        u_T = self.sample_from_transition_kernel(u_0, T, s=None)['u_t']
        lp = self.prior_logp(u_T)
        assert lp.shape==(bsz,)
        return lp

class GMM_VP:
    
    def __init__(self, which_beta):

        self.bsz=10000
        self.d=1
        self.vp = VP(d=self.d, max_beta = 20.0, which_beta = which_beta, device=torch.device('cpu'))
        self.K = 2
        self.pi = torch.ones(self.K) / self.K
        self.mus = torch.tensor([-10., 10.])
        #1000, 1000.])
        self.stds = torch.tensor([1.0, 1.0])
        self.q0 = D.MixtureSameFamily(
            D.Categorical(self.pi),
            D.Normal(self.mus, self.stds)
        )


    def reset_means(self, mean_vec):
        self.mus = mean_vec
        self.q0 = D.MixtureSameFamily(
            D.Categorical(self.pi),
            D.Normal(self.mus, self.stds)
        )

    def sample_q0(self, bsz=None):
        if bsz is None:
            bsz = self.bsz
        return self.q0.sample(sample_shape=(bsz,))

        
    def q_xs_given_xt(self, xt, s, t):

        bsz = xt.shape[0]
        assert xt.shape == (bsz,)
        assert s.shape == (bsz,)
        assert t.shape == (bsz,)

        weights = []
        mus = []
        stds = []

        for k in range(self.K):

            pi_k = self.pi[k]
            mu_k = self.mus[k]
            std_k = self.stds[k]
            var_k = std_k.pow(2)
            m_ts = self.vp.transition_mean_coefficient(t=t, s=s)
            m_s0 = self.vp.transition_mean_coefficient(t=s, s=None)
            #print("m ts", m_ts)
            #print("m s0", m_s0)
            var_ts = self.vp.transition_var(t=t, s=s)
            var_s0 = self.vp.transition_var(t=s, s=None)

            loc = m_ts * m_s0 * mu_k
            var = var_ts + m_ts.pow(2) * (m_s0.pow(2) * var_k + var_s0)
            std = var.sqrt()
            N = Normal(loc, std)
            w = pi_k * N.log_prob(xt).exp()
            assert w.shape == (bsz,)
            weights.append(w)
            # ..................
            loc_term1 = m_s0 * mu_k
            loc_term2 = (m_ts * (m_s0.pow(2) * var_k + var_s0)) / (var_ts + m_ts.pow(2) * (m_s0.pow(2)*var_k + var_s0)) * (xt - m_ts * m_s0 * mu_k)
            loc = loc_term1 + loc_term2
            mus.append(loc)
            var_term3_numer = m_ts.pow(2) * (m_s0.pow(2)*var_k + var_s0).pow(2)
            var_term3_denom = var_ts + m_ts.pow(2)*(m_s0.pow(2) * var_k + var_s0)
            var_term3 = var_term3_numer / var_term3_denom
            var = m_s0.pow(2) * var_k + var_s0 - var_term3
            std = var.sqrt()
            stds.append(std)
    
        weights = torch.stack(weights, dim=-1)
        weights = weights / weights.sum(-1)[:, None]
        mus = torch.stack(mus, dim=-1)
        stds = torch.stack(stds, dim=-1)
        mix = D.Categorical(weights)
        comp = D.Normal(mus, stds)
        dist = D.MixtureSameFamily(mix, comp)
        return dist, weights, mus, stds
    
    def test(self, ts):
        bsz = 1
        x0 = self.sample_q0(bsz=bsz).unsqueeze(-1)
        hardnesses = []
        for t_idx, tscalar in enumerate(ts):
            
            t = torch.ones(bsz) * tscalar            
            xt = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']
            s = torch.zeros_like(t) + 0.001
            q0_given_t, weights, mus, stds = self.q_xs_given_xt(xt.squeeze(-1), s=s, t=t)
            variances = stds.pow(2)
            weights, mus, variances = weights.squeeze(0), mus.squeeze(0), variances.squeeze(0)
            first = 0.0
            second = 0.0
            K = len(mus)

            for k in range(K):
                pi_k = weights[k]
                mu_k = mus[k]
                var_k = variances[k]
                first += pi_k * (mu_k.pow(2) + var_k)
                second += pi_k * mu_k
            
            second = second.pow(2)
            var_t = first - second
            t_item = round(t.item(),3)
            var_t = round(var_t.item(),4)
            hardnesses.append(var_t)
        return hardnesses

if __name__=='__main__':


    num_t = 100
    ts = torch.linspace(.1, .99, num_t)
    ts_numpy = ts.numpy()

    hardnesses_one = []
    hardnesses_two = []
    MC = 5
    for mc in range(MC):

        gmm_vp = GMM_VP(which_beta = 1)
        hardnesses1 = gmm_vp.test(ts = ts)

        gmm_vp = GMM_VP(which_beta = 2)
        hardnesses2 = gmm_vp.test(ts = ts)

        hardnesses_one.append(torch.tensor(hardnesses1))
        hardnesses_two.append(torch.tensor(hardnesses2))


    hardnesses_one = torch.stack(hardnesses_one, dim=0)
    assert hardnesses_one.shape == (MC, num_t)
    hardnesses_one = hardnesses_one.mean(0).numpy()
    hardnesses_two = torch.stack(hardnesses_two, dim=0).mean(0).numpy()

    plt.plot(ts_numpy, hardnesses1, label='hardnesses beta1')
    plt.plot(ts_numpy, hardnesses2, label='hardnesses beta2')
    plt.ylim(0, .5)
    plt.legend()
    plt.show()


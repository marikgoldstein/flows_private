import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# local
from timechange import TimeSampler

CAT = D.Categorical
MVN = D.MultivariateNormal
MIX = D.MixtureSameFamily
UNIF = D.Uniform
torch.manual_seed(0)
Linear = nn.Linear
Parameter = nn.Parameter
ModuleList = nn.ModuleList
Sequential = nn.Sequential


class GMM_Config:

    def __init__(self, layout, N0, N1):

        self.N0 = N0
        self.N1 = N1
        self.d = 2 # some code assumes 2 for now
        self.layout = layout
        # note if changing these, change 
        # ranges and limits for plots
        self.min_x = -10.0
        self.max_x = 10.0
        self.min_y = -20
        self.max_y = 20

class GMM:

    def __init__(self, config, device):
        self.d = config.d
        self.config = config
        self.device = device
        cov = torch.eye(self.d)[None,...].repeat(self.config.N1, 1, 1)
        cov = cov.to(self.device)
        mu = []
        lowx, highx, lowy, highy = config.min_x, config.max_x, config.min_y, config.max_y
        mu = torch.tensor([
            [UNIF(lowx, highx).sample(),UNIF(lowy, highy).sample()]
            for k in range(config.N1)
        ]).float().to(self.device)
        self.q1 = MIX(
            CAT(torch.ones(self.config.N1).to(self.device) / self.config.N1),
            MVN(mu, covariance_matrix=cov)
        )

class DataModule:


    def __init__(self, bsz, d, device):

        self.bsz = bsz
        self.d = d
        self.device = device
        self.z = torch.zeros(1).to(self.device)
        self.o = torch.ones(1).to(self.device)
        self.q0 = D.Normal(self.z, self.o)
        self.q1 = GMM(GMM_Config(layout='diffusion', N0=1, N1=2), device = self.device).q1
        self.compute_marginal_means()
        self.U = UNIF(low=self.z, high=self.o)
        self.reg_sampler = TimeSampler(mint=1e-5, time_change=False, optimize=False, device = self.device)
        self.change_sampler = TimeSampler(mint=1e-5, time_change=True, optimize=False, device = self.device)

    def batch_dot(self, x, y):
        return (x*y).sum(-1)

    def prior_samples(self, N):
        return self.q0.sample(sample_shape=(N, self.d)).squeeze(-1)  
    
    def get_unif_times(self,):                                                                   
        return self.U.sample(sample_shape=(self.bsz,))[:,None]

    def get_x1_batch(self, bsz=None):
        if bsz is None:
            bsz = self.bsz
        return self.q1.sample(sample_shape=(bsz,))

    def z_given_x1(self, x1):
        q = torch.distributions.Normal(x1, torch.ones_like(x1))
        z = q.sample()
        # stein score -.5 * 2 * (z-x1)/1 = -(z-x1)
        score_z = -(z - x1) # var is 1.0
        return z, score_z
    
    def f1(self, z, x1): 
        return z
    
    def f2(self, z, x1):
        return 4.0 * x1
    
    def div_f1(self, z, x1): 
        return 2.0 * torch.ones(z.shape[0])

    def div_f2(self, z, x1):
        return torch.zeros(z.shape[0])

    def control_helper(self, x1):
        x1 = self.get_x1_batch()
        bsz = x1.shape[0]
        z, score_z = self.z_given_x1(x1)        
        c1 = self.batch_dot(score_z, self.f1(z, x1)) + self.div_f1(z, x1)
        c2 = self.batch_dot(score_z, self.f2(z, x1)) + self.div_f2(z, x1)
        c = torch.cat([c1[:,None], c2[:,None]], dim=-1)
        return c

    def get_batch(self,):
        x1 = self.get_x1_batch()
        bsz = x1.shape[0]
        #t = self.get_unif_times()
        t1, w1, w2 = self.reg_sampler.sample(self.bsz)
        t1 = t1[:, None]
        assert t1.shape == (bsz, 1)
        weight1 = w1 * w2
        #print("type weight1", type(weight1), weight1)
        c = self.control_helper(x1,)

        xt, xt_mean = self.sample_xt_given_x1(x1, t1)
        x0 = self.x0_from_x1_xt_t(x1, xt, t1)
        ut = self.cond_field(x1, xt, t1)
        batch1 = x0, x1, xt, xt_mean, t1, ut, c, weight1
    
        t2, w1, w2 = self.change_sampler.sample(self.bsz)
        t2 = t2[:,None]
        assert t2.shape == (bsz, 1)
        weight2 = w1 * w2
        #print("type weight 2", type(weight2), weight2)
        c2 = self.control_helper(x1,)
        xt2, xt2_mean = self.sample_xt_given_x1(x1, t2)
        x02 = self.x0_from_x1_xt_t(x1, xt2, t2)
        ut2 = self.cond_field(x1, xt2, t2)
        batch2 = x02, x1, xt2, xt2_mean, t2, ut2, c, weight2

        return batch1, batch2


    def cond_field(self, x1, xt, t):
        return x1 + (t*x1)/(1-t) -  xt/(1-t)

    def q_xt_given_x1(self, x1, t):
        return D.Normal(loc = t*x1, scale = (1 - t).repeat(1,2))

    def sample_xt_given_x1(self, x1, t):
        q = self.q_xt_given_x1(x1, t)
        xt = q.sample()
        xt_mean = q.loc
        return xt, xt_mean

    def x0_from_x1_xt_t(self, x1, xt, t):
        return xt/(1-t) - (t*x1)/(1-t)

    def compute_marginal_means(self,):
        N = 10000
        xs = self.q1.sample(sample_shape=(N,))
        assert xs.shape == (N, 2)
        mu = xs.mean(0).unsqueeze(0)
        assert mu.shape == (1, 2)
        mu_sq = xs.pow(2).mean(0).unsqueeze(0)
        assert mu_sq.shape == (1,2)
        self.x1_mean = mu
        self.x1_sq_mean = mu_sq

    def make_reg_features(self, batch):
        x0, x1, xt, xt_mean, t, ut, weight = batch # doesn't use weight
        bsz = x0.shape[0]                                                                        
        split = int(bsz / 2)
        # later use x0
        #x0_mean = self.x0_from_x1_xt_t(x1, xt_mean, t)
        bsz = x1.shape[0]
        x1_mean = self.x1_mean.repeat(bsz,1)
        x1_sq_mean = self.x1_sq_mean.repeat(bsz,1)
        lst = [x1, x1.pow(2), xt, xt.pow(2), t]
        features = torch.cat(lst, dim=-1)
        mean_lst = [x1_mean, x1_sq_mean, t * x1_mean, (1-t).pow(2) + t.pow(2) * x1_sq_mean, t]
        mean_features = torch.cat(mean_lst, dim=-1)
        assert features.shape == (bsz, 9)
        assert mean_features.shape == (bsz, 9)
        return (features[:split], features[split:], mean_features[:split], mean_features[split:]), split

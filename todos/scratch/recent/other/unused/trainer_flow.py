import time
from typing import Optional, List
import torch
import torch.nn as nn
Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
from torchvision import datasets, transforms
import numpy as np
import math
import torchvision
import matplotlib.pyplot as plt
import wandb

# local
from sklearn.datasets import make_moons, make_circles, make_classification
from research_controlvars import regression_cv, regression_cv_no_mean, simple_dimwise_cv
from research_timechange import TimeSampler
#from sde_sampling import EM, plot_model_samples, generate_image_grid
from trainer_generic import Trainer
from utils_flows import FlowPrior
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    trace_fn,
    batch_transpose,
    eye,
    linspace,
    sqnorm
)

class FlowTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        # extras:
        self.prior_module = FlowPrior(config = self.config, trainer = self)
        self.time_min_one = self.get_time_sampler(self.T_min, 1.0)
        self.time_min_delta = self.get_time_sampler(self.T_min, self.delta)
        self.time_delta_one = self.get_time_sampler(delta, 1.0)
        self.reg_sampler = TimeSampler(mint=self.T_min, time_change=False, device = self.device, maxt=1.0)
        self.changed_sampler = TimeSampler(mint=self.T_min, time_change=True, device = self.device, maxt=1.0)
    
    def demo(self,):
       
        batch = self.get_batch(bsz = self.bsz)
        batch, ldj = self.encoder.preprocess(batch)                                                                        
        bsz = batch.shape[0]        
        x1 = batch
        x0 = self.sample_prior(bsz)
        
        # first
        t, weight1 = self.reg_sampler.sample(bsz,)
        xt1, ut1, v1 = self.get_fields(x0, x1, t)
        loss1 = (v1 - ut1).pow(2).sum(-1)
        est1 = loss1 * weight1
        assert est1.shape == (bsz,)
        est1_mean, est1_std = est1.mean(0), est1.std(0)

        # second:
        t, weight2 = self.changed_sampler.sample(bsz,)
        xt2, ut2, v2 = self.get_fields(x0, x1, t)
        loss2 = (v2 - ut2).pow(2).sum(-1)
        est2 = loss2 * weight2
        assert est2.shape == (bsz,)
        est2_mean, est2_std = est2.mean(0), est2.std(0)
        print("mse", (est1 - est2).pow(2).mean(0))
        print("means", est1_mean, est2_mean)
        print('stds', est1_std, est2_std)

    def get_fields(self, x0, x1, t):
        xt = self.phi(x0, x1, t)
        ut = self.dphi(x0, x1, t)
        v = self.apply_model(xt, t)
        return xt, ut, v

    def sample_prior(self, bsz):
        return self.prior_module.sample(bsz,)
       
    def apply_model(self, xt, t):
        bsz = xt.shape[0]
        xt = possibly_reshape_to_image(trainer = self, x = xt, bsz = bsz)
        v = self.model1(xt, t)
        v = possibly_flatten(trainer = self, x = v, bsz = bsz) 
        return v

    def possibly_compute_control_var(self, xt, mu_xt, loss, loss_per_dim):
        if self.compute_control_var:
            loss_control = simple_dimwise_cv(xt, mu_xt, loss_per_dim)
        else:
            loss_control = loss.clone()
        return loss_control

    def possibly_compute_time_change(self, x0, x1, original_loss):
        
        bsz = x0.shape[0]

        if self.compute_time_change:
            
            t, weight = self.changed_sampler.sample(bsz,)
            xt, ut, v = self.get_fields(x0, x1, t)
            loss = weight * (v - ut).pow(2).sum(-1)
        else:
            loss = origina_loss.clone()

        return loss

    def base_loss_fn(self, batch, is_train):
        return self.flow_loss_fn(batch, is_train)

    def flow_loss_fn(self, batch, is_train):
        bsz = batch.shape[0]
        x1, ldj = self.encoder.preprocess(batch)
        x0 = self.sample_prior(bsz)
        t, weight = self.reg_sampler.sample(bsz,)
        xt, ut, v = self.get_fields(x0, x1, t)
        mu_xt, std_xt = self.q_xt_given_x1(x1, t)
        loss_per_dim = (v - ut).pow(2)
        loss = loss_per_dim.sum(-1)
        assert loss.shape==(bsz,)
        loss_control = self.possibly_compute_control_var(xt, mu_xt, loss, loss_per_dim)
        loss_time_change = self.possibly_compute_time_change(x0, x1, loss)
        loss = weight * loss
        loss_control = weight * loss_control
        return {'loss': loss, 'loss_control': loss_control, 'loss_time_change': loss_time_change}

    def phi(self, x0, x1, t):
        _t = t[:,None]
        xt = _t*x1 + (1-_t)*x0
        return xt

    def dphi(self, x0, x1, t):
        return x1 - x0
    
    @torch.no_grad()
    def generate_samples(self, n_samples):
        xt = self.sample_prior(n_samples)
        ones = torch.ones(n_samples,).type_as(xt)
        assert xt.shape==(n_samples, self.d)
        ts = torch.linspace(0, 1, self.n_sample_steps).type_as(xt)
        dt = ts[1] - ts[0]
        for t in ts:
            xt = xt + self.apply_model(xt, t*ones) * dt
        xt = self.encoder.generate_x(xt)
        return xt
   
    # assumes N(0,1) base dist and path xt = tx1 + (1-t)x0
    def q_xt_given_x1(self, x1, t):
        # x0 ~ N(0,1)
        # x1 ~ data
        # xt = tx1 + (1-t)x0
        # (1-t)x0 ~ N(0, (1-t)^2)
        # so xt ~ N(tx1, (1-t)^2)
        wide_t = t[:, None]
        mu = wide_t * x1
        std = (1 - wide_t).repeat(1,2)
        return mu, std

    '''
    # assumes xt = tx1 + (1-t)x0
    def conditional_density_path(self, xt, t, x1):
        assert xt.shape==x1.shape
        tx1 = t.unsqueeze(-1) * x1
        onemt = (1 - t)[:,None].repeat(1,2)
        inverted = (xt-tx1)/onemt
        recon = tx1 + onemt*inverted # to test inversion
        deriv = 1 / (1-t)
        log_prior = self.q0.log_prob(inverted) 
        log_det = 2 * deriv.log()
        log_prob = log_prior + log_det
        prob = log_prob.exp()
        return prob

    # assumes xt = tx1 + (1-t)x0
    def conditional_field(self, x, t, x1):
        assert x.shape==x1.shape
        num = (x1-x)
        assert num.shape==x.shape
        denom = (1-t).unsqueeze(-1)
        result = num / denom
        return result

    # (consistent) monte carlo estimate of marginal 
    # field using the marginalization equation in
    # flow matching paper:
    # ut(x) = 1/pt(x) * E_{q(x1)}\Bigg[ ut(x|x1)pt(x|x1) \Bigg]
    # uses K samples of x1 for each x being evaluated
    def estimate_marginal_field(self, x, t):
        bsz = x.shape[0]
        assert x.shape == (bsz, 2)
        assert t.shape == (bsz,)
        K = 1000 # num samples of x1 per estimate of pt(xt)
        x1s = q1.sample(sample_shape=(bsz,K))
        assert x1s.shape==(bsz,K,2)
        x, t = x.unsqueeze(1), t.unsqueeze(1)
        assert x.shape==(bsz,1,2)
        assert t.shape==(bsz,1)
        x, t = x.repeat(1,K,1), t.repeat(1,K)
        assert x.shape==(bsz,K,2)
        assert t.shape==(bsz,K)
        x = x.reshape(bsz*K,2)
        x1s = x1s.reshape(bsz*K,2)
        t = t.reshape(bsz*K,)
        cond_prob = conditional_density_path(x, t, x1s)
        cond_field = conditional_field(x, t, x1s)
        assert cond_prob.shape==(bsz*K,)
        assert cond_field.shape==(bsz*K,2)
        numer = cond_prob.unsqueeze(-1) * cond_field
        assert numer.shape==(bsz*K,2)
        numer = numer.reshape(bsz, K, 2)
        denom = cond_prob.reshape(bsz, K)
        numer = numer.mean(1)
        denom = denom.mean(1)
        assert numer.shape == (bsz, 2)
        assert denom.shape == (bsz,)
        out = numer / denom.unsqueeze(-1)
        return out
    '''

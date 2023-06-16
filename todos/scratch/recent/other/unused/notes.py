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
#from sklearn.datasets import make_moons, make_circles, make_classification
from research_controlvars import regression_cv, regression_cv_no_mean
from sde_sampling import EM, plot_samples, plot_mnist, generate_image_grid
from trainer_generic import Trainer
from utils import possibly_reshape_to_image, possibly_flatten, merge_model_dicts
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    trace_fn,
    batch_transpose,
    eye,
    linspace,
    sqnorm
)
from research_timechange import TimeSampler 

class DiffusionTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.time_min_one = self.get_time_sampler(self.T_min, 1.0)
        self.time_min_delta = self.get_time_sampler(self.T_min, self.delta)
        self.time_delta_one = self.get_time_sampler(self.delta, 1.0)

    # needs score pred to be defined
    def generate_samples(self, n_samples, with_ema_model):
        raw = EM(trainer = self, n_samples = n_samples, with_ema_model = with_ema_model)
        x = self.encoder.generate_x(raw)
        return x


    def call_model_on_flat(self, D, with_ema_model):
        u_t, t = D['u_t'], D['t']
        bsz = u_t.shape[0]
        u_t = possibly_reshape_to_image(trainer = self, x=u_t, bsz=bsz)
        out = self.model_obj(u_t, t, with_ema_model)
        out = possibly_flatten(trainer = self, x=eps, bsz=bsz)
        return out

    def eps_pred(self, D, with_ema_model):
        assert self.model_parameterization == 'eps'
        return self.call_model_on_flat(D, with_ema_model)
 
    def omega_pred(self, D, with_ema_model):
        assert self.model_parameterization == 'omega'
        return self.call_model_on_flat(D, with_ema_model)

    def score_pred(self, D, with_ema_model):
        assert self.model_parameterization == 'score'
        return self.call_model_on_flat(D, with_ema_model)

    def score_from_model(self, D, with_ema_model):
        
        if self.model_parameterization == 'eps':

            eps_hat = self.eps_pred(D, with_ema_model)
            std = self.sde.transition_std(D['t'])
            score = self.noise_to_score(eps_hat, std)
            return score
        
        elif self.model_parameterization == 'omega':

            omega_hat = self.omega_pred(D, with_ema_model)
            var = self.sde.transition_var(D['t'])
            score = omeg


    def base_loss_fn(self, batch, is_train, with_ema_model):
        # diffusion loss type is either neblo or perceptual
        # this retrieves either self.nelbo_losses or self.perceptual_losses
        fn = getattr(self, '{}_losses'.format(self.diffusion_loss_type))
        D = fn(batch, is_train, with_ema_model)
        return D


    def _measure_grad_variance(self, batch):
        
        with_ema_model = False
        var_dict = {}
        N = batch.shape[0]
        for loss_type in ['nelbo', 'nelbo_st']:

            var_dict[loss_type] = {}
            for n, p in self.model_obj.model.named_parameters():
                var_dict[loss_type][n] = []
            
            
            u_0, ldj = self.encoder.preprocess(batch)
            for i in range(N):
                ui = u_0[i]
                ui = ui[None, ...]
                assert ui.shape == (1, 28*28)
                t, w = self.time_min_one.sample(1,)
                assert t.shape == (1,), t.shape
                D = self.sde(ui, t) 
                assert torch.allclose(D['t'], t) # just a debug
                # prior term
                prior = self.sde.cross_ent_helper(ui)
                if loss_type == 'nelbo':
                    int_t = w * self.dsm(D, with_ema_model)
                    nelbo = -(prior + int_t)
                    assert prior.shape == (1,)
                    assert int_t.shape == (1,)
                    loss = nelbo
                else:
                    int_st = self.loss_st(ui, 'dsm', with_ema_model)
                    nelbo_st = -(prior + int_st)
                    assert prior.shape == (1,)
                    assert int_st.shape == (1,)
                    loss = nelbo_st
                
                assert loss.shape == (1,)
                self.model_obj.compute_grads_no_step(loss)
                for n, p in self.model_obj.model.named_parameters():
                    var_dict[loss_type][n].append(p.grad)
        return var_dict



    def perceptual_losses(self, batch, is_train, with_ema_model):
        assert False, "be smart about which loss to do with grads and which not to"
        u_0, ldj = self.encoder.preprocess(batch)
        bsz = u_0.shape[0]
        t, weight = self.time_min_one.sample(bsz,)
        D = self.sde(u_0, t)
        mse_t = self.noise_loss(D, with_ema_model)
        #mse_t_control = regression_cv(D['u_t'], D['mean'], D['t'], mse_t)
        mse_st = self.loss_st(u_0, 'noise_loss', with_ema_model)
        # todo doesnt return loss sep yet like nelbo does
        return {
            'loss': mse_t, 
            'loss_st': mse_st
        }

    def score(self, D):
        noise = D['noise']
        std = D['std']
        return self.noise_to_score(noise, std)

    def div(self, D):
        return self.sde.div_f(D['u_t'], D['t'])

    def noise_to_score(self, noise, std):
        return -noise / std[:, None]

    def omega_to_score(self, omega, var):
        return omega / var[:, None]


    def dsm_sep(self, u_0):

        bsz = u_0.shape[0]
        t, w = self.time_min_one.sample(bsz,)
        g2 = self.sde.G2(t)
       
        # -.5 * E[ stheta(yt, t) | x]
        D = self.sde(u_0, t) 
        score_hat = self.score_pre(D)
        term_one = -1.0 * g2 * score_hat.pow(2).sum(-1)
        assert term_one.shape == (bsz,)

        # E [ g2 * score top stheta]
        D = self.sde(u_0, t) 
        u_t, t, noise, std = D['u_t'], D['t'], D['noise'], D['std']
        score = self.score(D)
        score_hat = self.score_pred(D)
        term_two = g2 * (score * score_hat).sum(-1)
        assert term_two.shape == (bsz,)

        # E[div f(yt, t)]
        D = self.sde(u_0, t) 
        term_three = self.div(D)
        assert term_three.shape == (bsz,) 

        total = w * (term_one + term_two + term_three)

        assert total.shape == (bsz,)
        return total

    def nelbo_losses(self, batch, is_train, with_ema_model):
        
        u_0, ldj = self.encoder.preprocess(batch)

        # int terms
        bsz = u_0.shape[0]
        t, w = self.time_min_one.sample(bsz,)
        D = self.sde(u_0, t) 
        assert torch.allclose(D['t'], t) # just a debug
        
        # prior term
        prior = self.sde.cross_ent_helper(u_0)
    
        #int_t_cv = w * regression_cv(D['u_t'], D['mean'], D['t'], int_t)
        #int_t_sep = self.dsm_sep(u_0)

        nelbo, bpd, nelbo_st, bpd_st = None, None, None, None
        nelbo_new, bpd_new = None, None

        if not is_train or self.backprop_key == 'nelbo_mean':

            int_t = w * self.dsm(D, with_ema_model)
            nelbo = -(prior + int_t)
            bpd = self.nelbo_to_bpd(nelbo, ldj)
            
        if not is_train or self.backprop_key == 'nelbo_st_mean':

            int_st = self.loss_st(u_0, 'dsm', with_ema_model)
            nelbo_st = -(prior + int_st)
            bpd_st = self.nelbo_to_bpd(nelbo_st, ldj)           


        if not is_train or self.backprop_key == 'nelbo_new':

            int_new = self.loss_new(u_0, with_ema_model)
            nelbo_new = -(prior + int_new)
            bpd_new = self.nelbo_to_bpd(nelbo_new, ldj)


        return {
            'nelbo': nelbo,
            'nelbo_st': nelbo_st,
            'bpd': bpd,
            'bpd_st': bpd_st
        }
  
 
    def loss_new(self, D, with_ema_model):        
        omega = self.omega_pred(D, with_ema_model)
        ut = D['u_t']
        mu_ts = D['mean']
        score = self.score(D)
        inside = -(ut - mu_ts) - omega
        div = self.div(D)
        term1 = -0.5 * (g2/D['var'].pow(2)) * sqnorm(inside)
        term2 = 0.5 * g2 * sqnorm(score)
        term3 = div
        return term1 + term2 + term3

    def nelbo_to_bpd(self, nelbo, ldj):
        dimx = 28*28
        elbo = -nelbo
        elbo += ldj
        bpd = -(elbo / dimx) / np.log(2) + 8
        return bpd
 
    def get_time_sampler(self, mint, maxt):
        
        return TimeSampler(mint=mint, time_change=False, optimize=False, device=self.device, maxt=maxt)

    def noise_loss(self, D, with_ema_model):
        return 0.5 * sqnorm(D['noise'] - self.eps_pred(D, with_ema_model))
    
    def dsm(self, D, with_ema_model):
        
        g2 = self.sde.G2(D['t'])
        div = self.div(D)
        score = self.score(D)
        score_hat = self.score_pred(D, with_ema_model)
        score_diff = sqnorm(score - score_hat)
        return -0.5 * g2 * score_diff + 0.5 * g2 * sqnorm(score) + div

    def check(self, x,s):
        if torch.any(torch.isnan(x)):
            print(s,"is nan")
            assert False
        if torch.any(torch.isinf(x)):
            print(s,"is inf")
            assert False
        print(s, "stats: min mean max", x.min(), x.mean(), x.max())

    def loss_st(self, u_0, which_fn, with_ema_model):

        fn = self.dsm if which_fn == 'dsm' else self.noise_loss
        bsz = u_0.shape[0]
        t, w = self.time_min_one.sample(bsz,)
        higher = t > self.delta
        higher_wide = higher[:, None]
        Dlower = self.sde(u_0, t)
        s = self.t_to_s(t)
        s = torch.where(higher, s, torch.zeros_like(s)) # wont use s where lower
        u_s = self.sde(u_0, s)['u_t']
        #self.check(t,'t')
        #self.check(s,'s')
        Dhigher = self.sde(u_s, t, s=s)
        Dmerged = {}
        

        #self.check(Dhigher['u_t'],'dhigher ut')
        #self.check(Dlower['u_t'], 'dlower ut')

        u_t = higher[:,None] * Dhigher['u_t'] + (~higher[:,None]) * Dlower['u_t']
        noise = higher[:,None] * Dhigher['noise'] + (~higher[:,None]) * Dlower['noise']
        std = higher * Dhigher['std'] + (~higher) * Dlower['std']
        
        #self.check(u_t, 'u_t')
        #self.check(noise, 'noise')
        #self.check(std, 'std')

        Dmerged = {'u_t':u_t, 'noise':noise, 'std':std, 't':t}
        

        return w * fn(Dmerged, with_ema_model)



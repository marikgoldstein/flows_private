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
from copy import deepcopy
# local
#from sklearn.datasets import make_moons, make_circles, make_classification
from research_controlvars import regression_cv, regression_cv_no_mean
from sde_sampling import EM, plot_samples, plot_mnist, generate_image_grid
from trainer_generic import Trainer
from utils import merge_model_dicts, nelbo_to_bpd, merge1d, merge2d, decide_use_weight
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
from parameterizations import Converter

class DiffusionTrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.time_min_one = self.get_time_sampler(self.T_min, 1.0)
        self.time_min_delta = self.get_time_sampler(self.T_min, self.delta)
        self.time_delta_one = self.get_time_sampler(self.delta, 1.0)
        self.model_converter = Converter()
        #xt, t, std, var, m_t = self.get_std_var_mt(D)
    
    def get_time_sampler(self, mint, maxt):        
        return TimeSampler(mint=mint, time_change=False, optimize=False, device=self.device, maxt=maxt)

    def generate_samples(self, n_samples, with_ema_model):
        raw = EM(trainer = self, n_samples = n_samples, with_ema_model = with_ema_model)
        x = self.encoder.generate_x(raw)
        return x

    def true_conditional_score(self, D):
        return self.model_converter.noise_to_score(D)

    def base_loss_fn(self, batch,is_train, with_ema_model):
        return self.nelbo_losses(batch, is_train, with_ema_model)

    def div(self, D):
        return self.sde.div_f(D['u_t'], D['t'])

    def f_pred(self, D, with_ema_model):
        return self.model_converter.f_pred(self.model_obj, D, with_ema_model)
    
    def noise_pred(self, D, with_ema_model):
        return self.model_converter.noise_pred(self.model_obj, D, with_ema_model)

    def omega_pred(self, D, with_ema_model):
        return self.model_converter.omega_pred(self.model_obj, D, with_ema_model)
    
    def score_pred(self, D, with_ema_model):
        return self.model_converter.score_pred(self.model_obj, D, with_ema_model)
    
    def fsm(self, D, with_ema_model, with_weight = None):
        score = self.true_conditional_score(D)
        g2 = self.sde.G2(D['t'])
        f_hat = self.f_pred(D, with_ema_model)
        f_diff = sqnorm(D['u_0'] - f_hat) 
        w = g2 * (D['mean_coef'].pow(2) / D['var'].pow(2)) if with_weight else ones_like(g2)
        return -0.5 * w * f_diff + 0.5 * g2 * sqnorm(score) + self.div(D)

    def dsm(self, D, with_ema_model, with_weight=None):
        g2 = self.sde.G2(D['t'])
        score = self.true_conditional_score(D)
        eps = D['noise']
        eps_hat = self.noise_pred(D, with_ema_model)
        noise_diff = sqnorm(eps - eps_hat)
        w = g2 / D['var'] if with_weight else ones_like(g2)
        return -0.5 * w * noise_diff + 0.5 * g2 * sqnorm(score) + self.div(D)
 
    def osm(self, D, with_ema_model, with_weight = None): 
        g2 = self.sde.G2(D['t'])
        score = self.true_conditional_score(D)
        omega_hat = self.omega_pred(D, with_ema_model)
        omega_diff = -(D['u_t'] - D['mean']) - omega_hat
        w = g2 / D['var'].pow(2) if with_weight else ones_like(g2)
        return -0.5 * w * sqnorm(omega_diff) + 0.5 * g2 * sqnorm(score) + self.div(D)
    
    def t_to_s(self, t):

        s = torch.where(
            t > self.delta,
            t - (self.delta / 2.0),
            torch.zeros_like(t)
        )
        return s

    def transitions(self, u_0, t):
        # dict for delta to T
        higher = t > self.delta
        Dt0 = self.sde(u_0, t, s=None)
        s = self.t_to_s(t)
        u_s = self.sde(u_0, s, s=None)['u_t']
        Dts = self.sde(u_s, t, s=s)
        Dmerged = {}
        for key in ['u_0', 'u_t', 'mean', 'noise']:
            Dmerged[key] = higher[:,None] * Dts[key] + (~higher[:,None]) * Dt0[key]
        for key in ['mean_coef', 'std', 'var', 't']:
            Dmerged[key] = higher * Dts[key] + (~higher) * Dt0[key]
        return Dt0, Dmerged

    def nelbo_losses(self, batch, is_train, with_ema_model):
        
        u_0, ldj = self.encoder.preprocess(batch)
        bsz = u_0.shape[0]
        prior = self.sde.cross_ent_helper(u_0)
  
        out = {}
        for nelbo_type in ['fsm', 'fsm_st', 'dsm', 'dsm_st', 'osm', 'osm_st']:
            t, w  = self.time_min_one.sample(bsz,)
            Dt0, Dts_t0 = self.transitions(u_0, t)
            D = Dts_t0 if '_st' in nelbo_type else Dt0
            func = getattr(self, nelbo_type[:3])
            bprop_key = f'nelbo_{nelbo_type}_mean'
            nelbo_key = f'nelbo_{nelbo_type}'
            bpd_key = f'bpd_{nelbo_type}'

            if not is_train or self.backprop_key == bprop_key:
                use_weight = decide_use_weight(self.backprop_with_weight, is_train)
                int_term = w * func(D, with_ema_model, with_weight = use_weight)
                nelbo = -(prior + int_term)
                bpd = nelbo_to_bpd(nelbo, ldj)
                out[nelbo_key] = nelbo
                out[bpd_key] = bpd
            else:
                out[nelbo_key] = None
                out[bpd_key] = None
            
        return out 

    def measure_grad_variance(self):

        print("measuring grad variance")
        self.model_obj.eval_mode()
        batch = None
        for batch_idx, b in enumerate(self.train_loader):
            batch = b
            break
        
        batch, _ = batch
        batch = batch.to(self.device)
        original_bsz = batch.shape[0]
        var_dict, norm_dict = self._measure_grad_variance_helper(batch)
        D = {}
       
        for loss_type in ['fsm', 'fsm_st', 'dsm', 'dsm_st', 'osm', 'osm_st']:
            for with_weight in [False, True]:
                name = loss_type + f'_with_weight_{with_weight}'
                D[name] = {}
                for n, p in self.model_obj.model.named_parameters():
                    grad = torch.stack(var_dict[name][n], dim=0)
                    assert grad.shape[0] == original_bsz
                    D[name][n] = grad.var(0).sum()

        summary_D = {}
        for loss_type in ['fsm', 'fsm_st', 'dsm', 'dsm_st', 'osm', 'osm_st']:
            for with_weight in [False, True]:
                name = loss_type + f'_with_weight_{with_weight}'
                key = 'total_var_' + name
                summary_D[key] = 0.0
                for n, p in self.model_obj.model.named_parameters():
                    summary_D[key] += D[name][n]
                summary_D[key] = summary_D[key] / self.model_obj.total_params
       
        for k in norm_dict:
            summary_D[k] = norm_dict[k]
        if self.config.use_wandb:
            wandb.log(summary_D, step=self.step)

    def _measure_grad_variance_helper(self, batch):
        
        with_ema_model = False
        var_dict = {}
        norm_dict = {}
        N = batch.shape[0]
        for loss_type in ['fsm', 'fsm_st', 'dsm', 'dsm_st', 'osm', 'osm_st']:
            for with_weight in [False, True]:
                name = loss_type + f'_with_weight_{with_weight}'
                norm_name = name + '_avg_grad_norm'
                norm_dict[norm_name] = 0.0
                var_dict[name] = {}
                
                for n, p in self.model_obj.model.named_parameters():
                    var_dict[name][n] = []
 
                u_0, ldj = self.encoder.preprocess(batch)
                for i in range(N):
                    ui = u_0[i]
                    ui = ui[None, ...]
                    assert ui.shape == (1, 28*28)
                    t, w = self.time_min_one.sample(1,)
                    assert t.shape == (1,), t.shape
                    D1, D2 = self.transitions(ui, t)
                    D = D2 if '_st' in loss_type else D1
                    prior = self.sde.cross_ent_helper(ui)
                    func = getattr(self, loss_type[:3])
                    int_term = w * func(D, with_ema_model, with_weight = with_weight)
                    loss = -(prior + int_term)
                    assert loss.shape == (1,)
                    self.model_obj.compute_grads_no_step(loss)
                    for n, p in self.model_obj.model.named_parameters():
                        var_dict[name][n].append(p.grad)
                        norm_dict[norm_name] += (p.grad.data.norm(2).item() / N)
        
        return var_dict, norm_dict








'''
    def get_std_var_mt(self, D):

        xt = D['u_t']
        t = D['t']

        if 'std' in D:
            std = D['std']
            var = D['var']
            m_t = D['mean_coef']
        else:
            if '_st_' in self.backprop_key:
                
                higher = t > self.delta       
                s = torch.where(higher, self.t_to_s(t), zeros_like(t))   
                
                std_ts = self.sde.transition_std(t, s=s)
                std_t0 = self.sde.transition_std(t, s=None)
                std = torch.where(higher, std_ts, std_t0)

                m_ts = self.sde.transition_mean_coefficient(t, s=s)
                m_t0 = self.sde.transition_mean_coefficient(t, s=None)
                m_t = torch.where(higher, m_ts, m_t0)

            else:
                std = self.sde.transition_std(t, s=None)
                m_t = self.sde.transition_mean_coefficient(t, s=None)

            var = std.pow(2)

        return std, var, m_t
    '''



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
from sde_sampling import EM
from trainer_generic import Trainer
from utils import merge_model_dicts, merge1d, merge2d, decide_use_weight
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
        self.time_sampler = self.get_time_sampler(self.config.T_min, 1.0)
        assert self.config.model_parameterization in ['xpred', 'noisepred'] 
    
    def get_time_sampler(self, mint, maxt):        
        return TimeSampler(mint=mint, time_change=False, optimize=False, device=self.config.device, maxt=maxt, mode='regular')

    def generate_samples(self, n_samples, with_ema_model):
        raw = EM(trainer = self, n_samples = n_samples, with_ema_model = with_ema_model)
        D = self.encoder.generate_x(raw)
        return D

    def call_model_on_flat(self, D, with_ema_model):
        u_t = D['u_t']
        bsz = u_t.shape[0]
        u_t = self.encoder.image_shape(u_t)
        out = self.model_obj(u_t, D['t'], with_ema_model)
        out = self.encoder.flatten(out)
        return out

    def x_pred(self, D, with_ema_model):
        
        out = self.call_model_on_flat(D, with_ema_model)

        # check if noise to xpred
        if self.config.model_parameterization == 'noisepred':
            noise_hat = out
            score_hat = -noise_hat / D['std'][:, None] # noise to score
            x_hat = (score_hat * D['var'][:, None] + D['u_t']) / D['mean_coef'][:, None] # score to xpred
            out = x_hat
        return out
    
    def noise_pred(self, D, with_ema_model):
   
        out = self.call_model_on_flat(D, with_ema_model)

        # check if xpred to noise
        if self.config.model_parameterization == 'xpred':
            assert False
            x_hat = out
            score_hat = (x_hat * D['mean_coef'][:, None] - D['u_t']) / D['var'][:, None]
            noise_hat = -score_hat * D['std'][:, None]
            out = noise_hat
        return out

    def score_pred(self, D, with_ema_model):
 
        out = self.call_model_on_flat(D, with_ema_model)

        if self.config.model_parameterization == 'xpred':
            assert False
            x_hat = out
            score_hat = (x_hat * D['mean_coef'][:, None] - D['u_t']) / D['var'][:, None]
            out = score_hat
        
        elif self.config.model_parameterization == 'noisepred':
            noise_hat = out
            score_hat = -noise_hat / D['std'][:, None] # noise to score
            out = score_hat
        
        else:
            assert False
 
        return out

    def base_loss_fn(self, batch,is_train, with_ema_model):
        return self.nelbo_losses(batch, is_train, with_ema_model)

    def div(self, D):
        return self.sde.div_f(D['u_t'], D['t'])
   
    def g2_score(self, D):
        g2 = self.sde.G2(D['t'])
        score = -D['noise'] / D['std'][:, None]
        return g2, score

    def xmatching(self, D, with_ema_model, with_weight = None):
        assert False
        g2, score = self.g2_score(D)
        x_hat = self.x_pred(D, with_ema_model)
        x_diff = sqnorm(D['u_0'] - x_hat)  # u_0 could be u_s depending on where the Dict D came from
        w = g2 * (D['mean_coef'].pow(2) / D['var'].pow(2)) if with_weight else ones_like(g2)
        return -0.5 * w * x_diff + 0.5 * g2 * sqnorm(score) + self.div(D)

    def dsm(self, D, with_ema_model, with_weight=None):
        g2, score = self.g2_score(D)
        eps = D['noise']
        eps_hat = self.noise_pred(D, with_ema_model)
        noise_diff = sqnorm(eps - eps_hat)
        w = g2 / D['var'] if with_weight else ones_like(g2)
        if with_weight:
            ret = -0.5 * w * noise_diff + 0.5 * g2 * sqnorm(score) + self.div(D)
        else:
            ret = -0.5 * noise_diff
        return ret

    def t_to_s(self, t):
        if self.config.backprop_ts:
            s = torch.where(
                t > self.config.delta,
                t - (self.config.delta / self.config.s_denom),
                torch.zeros_like(t)
            )
        else:
            s = torch.zeros_like(t)
        return s
 
    def transitions(self, u_0, t):
        s = self.t_to_s(t) 
        u_s = self.sde(u_0, s)['u_t']
        Dts = self.sde(u_s, t, s=s)    
        u_t = Dts['u_t']
        return Dts

    def nelbo_losses(self, batch, is_train, with_ema_model):
       
        u_0, ldj = self.encoder.preprocess(batch)
        bsz = u_0.shape[0]
        prior = self.sde.cross_ent_helper(u_0)
        is_eval = not is_train
        use_weight = decide_use_weight(self.config.backprop_with_weight, is_train)
        out = {}
        #assert self.config.backprop_key in ['xmatching', 'dsm'] 
        assert self.config.backprop_key in ['dsm'] 
        t, w  = self.time_sampler.sample(bsz,)
        D = self.transitions(u_0, t)

        if is_eval:
            int_term = w * self.dsm(D, with_ema_model, with_weight = use_weight)
            nelbo = -(prior + int_term)
            bpd = self.encoder.nelbo_to_bpd(nelbo, ldj)
            out['nelbo_dsm'] = nelbo
            out['bpd_dsm'] = bpd
            #out['nelbo_xmatching'] = None
            #out['bpd_xmatching'] = None
        else:
            if self.config.backprop_key == 'xmatching':
                assert False
                int_term = w * self.xmatching(D, with_ema_model, with_weight = use_weight)
                nelbo = -(prior + int_term)
                bpd = self.encoder.nelbo_to_bpd(nelbo, ldj)
                #out['nelbo_xmatching'] = nelbo
                #out['bpd_xmatching'] = bpd
                out['nelbo_dsm'] = None
                out['bpd_dsm'] = None
            else:
                int_term = w * self.dsm(D, with_ema_model, with_weight = use_weight)
                nelbo = -(prior + int_term)
                bpd = self.encoder.nelbo_to_bpd(nelbo, ldj)
                #out['nelbo_xmatching'] = None
                #out['bpd_xmatching'] = None
                out['nelbo_dsm'] = nelbo
                out['bpd_dsm'] = bpd

        return out 

    def aux_metrics(self):
        print("measuring grad variance")
        self.model_obj.eval_mode()
        batch = None
        with_ema_model = False
        for batch_idx, (b, _) in enumerate(self.train_loader):
            batch = b
            break
        if self.config.overfit:
            batch = self.overfit_batch 
        
        batch = batch.to(self.config.device)
        original_bsz = batch.shape[0]
        N = batch.shape[0]
        u_0, ldj = self.encoder.preprocess(batch)
      
        var_dict, norm_dict = {}, {}
        #for loss_type in ['dsm', 'xmatching']:
        for loss_type in ['dsm']:
            func = self.dsm if loss_type == 'dsm' else self.xmatching
            for with_weight in [False, True]:
                name = loss_type 
                if with_weight:
                    name += '_with_weight'

                norm_name = name + '_avg_grad_norm'
                norm_dict[norm_name] = 0.0
                var_dict[name] = {}
                for n, p in self.model_obj.model.named_parameters():
                    var_dict[name][n] = []
                
                for i in range(N):
                    ui = u_0[i][None, ...]
                    assert ui.shape == (1, self.config.H*self.config.W*self.config.C)
                    t, w = self.time_sampler.sample(1,)
                    assert t.shape == (1,), t.shape
                    prior = self.sde.cross_ent_helper(ui)
                    D = self.transitions(ui, t)
                    int_term = w * func(D, with_ema_model, with_weight = with_weight)
                    loss = -(prior + int_term)
                    self.model_obj.compute_grads_no_step(loss)
                    for n, p in self.model_obj.model.named_parameters():
                        var_dict[name][n].append(p.grad)
                        norm_dict[norm_name] += (p.grad.data.norm(2).item() / N)

        summary_D = {}
        #for loss_type in ['dsm', 'xmatching']:
        for loss_type in ['dsm']:
            for with_weight in [True, False]:
                name = loss_type
                if with_weight:
                    name += '_with_weight'
                key = 'total_var_' + name
                summary_D[key] = 0.0
                for n, p in self.model_obj.model.named_parameters():
                    grad = torch.stack(var_dict[name][n], dim=0)
                    assert grad.shape[0] == original_bsz
                    summary_D[key] += grad.var(0).sum()
                summary_D[key] = summary_D[key] / self.model_obj.total_params
      
        for k in norm_dict:
            summary_D[k] = norm_dict[k]
        if self.config.use_wandb:
            wandb.log(summary_D, step=self.step)


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
#from sde_sampling import EM, model_samples_to_wandb, real_samples_to_wandb, generate_image_grid 
from utils import merge_model_dicts, merge1d, merge2d, decide_use_weight
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, 
    sqnorm
)
from research_timechange import TimeSampler 
#from sde_vp import VP  
#from snr_vp import VP
from snr_vp2 import VP


class Diffusion:

    def __init__(self, config):
        self.config = config
        self.time_sampler = self.get_time_sampler(self.config.T_min, 1.0)
        self.sde = VP(self.config)
   
    def get_time_sampler(self, mint, maxt):        
        return TimeSampler(mint=mint, time_change=False, optimize=False, maxt=maxt)

    def xpred(self, ut, t, model):
	v = self.vpred(ut, t, model)
  	alpha, sigma = self.sde.get_alpha_sigma(t)
	return self.sde.zv_to_x(alpha, sigma, ut, v)

    def vpred(self, ut, t, model):
	return model(ut, t, torch.zeros_like(t).long())

    def noisepred(self, ut, t, model):
	v = self.vpred(ut, t, model)
  	alpha, sigma = self.sde.get_alpha_sigma(t)
	return self.sde.zv_to_eps(alpha, sigma, ut, v)

    def loss_fn(self, u_0, model):
        bsz = u_0.shape[0]
        #t, time_weight = self.time_sampler.sample(bsz,)
        #t = t.type_as(u_0)
        #time_weight = time_weight.type_as(u_0)
	t = torch.rand(bsz,).type_as(u_0)
        D = self.sde(u_0, t)
        eps_pred = self.noisepred(D['u_t'], D['t'], model)
        mse = (eps_pred - D['noise']).pow(2).sum(-1).sum(-1).sum(-1)
        assert mse.shape == (bsz,)
        return mse

    def sample(self, n_samples, model, device):
        conf = self.config
        N = conf.n_sample_steps - 1
        T_begin = 1.0 - conf.T_max # 1 - large = almost0
        T_final = 1.0 - conf.T_min  # 1 - small = almost1
        ts = torch.linspace(T_begin, T_final, N+1)
        zt = self.sde.sample_from_prior(n_samples).to(device)
        for i in range(N):
            # transition info
            t = ts[i+1]
            s = ts[i]
            logsnr_t = self.sde.logsnr(t)
            logsnr_s = self.sde.logsnr(s)
            alpha_t, std_t = self.sde.get_alpha_sigma(t)
            alpha_s, std_s = self.sde.get_alpha_sigma(s)
            alpha_ts = alpha_t / alpha_s
            var_t = std_t.pow(2)
            var_s = std_s.pow(2)
            var_ts = var_t - alpha_ts.pow(2) * var_s
            # prediction
            xpred = self.xpred(zt, t, model)
            xpred = xpred.clamp(min=-1, max=1)
            # r must be var s / var t which must be exp(logsnr_t - logsnr_s)
            # 1-t must be var_ts / var_t which must be 1 - exp(logsnr_t - logsnr_s)
            term1 = (var_s  / var_t) * (alpha_ts * zt)
            term2 = (var_ts / var_t) * (alpha_s * xpred)
            #term1 = ((alpha_ts * var_s) / var_t) * zt
            #term2 = ((alpha_s * var_ts) / var_t) * xpred
            #mu = r * mu_st + (1-r) * mu_s
            mu = term1 + term2
            #std_or_var = (var_ts * var_s ) / var_t
            r = exp(logsnr_t - logsnr_s)
            min_lvar = (1 - r) + nn.LogSigmoid()(-logsnr_s)
            max_lvar = (1 - r) + nn.LogSigmoid()(-logsnr_t)
            noise = .2
            variance = torch.exp(noise * max_lvar + (1-noise) * min_lvar))
            std = variance.sqrt()
            zt = mu + sigma * torch.randn_like(mu)
        
        print("todo last step")
        min_logsnr = self.logsnr(ts[-1])
        xpred = self.xpred(zt, t, model)
        xpred -clip_x
        return xpred

    '''
    def div(self, D):
        return self.sde.div_f(D['u_t'], D['t'])

    def dsm_just_noise(self, D, model):
        eps = D['noise']
        eps_hat = self.noise_pred(D, model)
        noise_diff = (eps - eps_hat).pow(2).sum(-1).sum(-1).sum(-1)
        return -0.5 * noise_diff

    def dsm(self, D, model, with_weight = None):

        if not with_weight:
            return self.dsm_just_noise(D, model)
        else:
            g2 = self.sde.G2(D['t'])
            eps = D['noise']
            score = -eps/ D['std'][:, None, None, None]
            eps_hat = self.noise_pred(D, model)
            noise_diff = (eps - eps_hat).pow(2).sum(-1).sum(-1).sum(-1)
            return -0.5 * (g2 / D['var']) * noise_diff + 0.5 * g2 * (score).pow(2).sum(-1).sum(-1).sum(-1) + self.div(D)

    def nelbo_fn(self, u_0, model):
        bsz = u_0.shape[0]
        prior = self.sde.cross_ent_helper(u_0)
        #use_weight = decide_use_weight(self.config.backprop_with_weight, is_train)
        with_weight = False
        #out = {}
        #assert self.config.backprop_key in ['dsm'] 
        t, time_weight = self.time_sampler.sample(bsz,)
        t = t.type_as(u_0)
        time_weight = time_weight.type_as(u_0)
        D = self.transitions(u_0, t)
        int_term = time_weight * self.dsm(D, model, with_weight = with_weight)
        nelbo = -(prior + int_term)
        return {'nelbo': nelbo}
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
        if self.config.backprop_ts:
            s = self.t_to_s(t) 
            u_s = self.sde(u_0, s)['u_t']
            Dts = self.sde(u_s, t, s=s)    
        else:
            Dts = self.sde(u_0, t, s=None)
        u_t = Dts['u_t']
        return Dts
    
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


def sde_test(self,):
bsz = 1
for tscalar in torch.linspace(.1, .9, 4):
    print("tscalar", tscalar)
    t = torch.ones(bsz,) * tscalar
    meancoef1 = self.sde.transition_mean_coefficient(t)
    meancoef2 = self.sde.transition_mean_coefficient2(t)
    std1 = self.sde.transition_std(t)
    std2 = self.sde.transition_std2(t)
    print("one: mean coef, std ", meancoef1, std1)
    print("two: mean coef, std ", meancoef2, std2)
'''

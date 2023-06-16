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
#from research_timechange import TimeSampler 


Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal
ones_like = torch.ones_like
atan = torch.atan
tan = torch.tan
       

#def get_time_sampler(self, mint, maxt):        
#    return TimeSampler(mint=mint, time_change=False, optimize=False, maxt=maxt)


class Diffusion:

    def __init__(self, config):
        self.config = config
        #self.time_sampler = self.get_time_sampler(self.config.T_min, 1.0)
        self.d = config.d
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.prior_loc = 0.0
        self.prior_scale = 1.0

    def forward(self, t):
        return None

    def get_alpha_sigma(self, t):
        alpha_t = torch.sqrt(torch.sigmoid(self.logsnr(t)))
        sigma_t = torch.sqrt(torch.sigmoid(-self.logsnr(t)))
        return alpha_t, sigma_t
       
    def diffuse(self, x, t):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        noise = torch.randn_like(x)
        mean = alpha_t[:, None, None, None] * x
        z = mean + sigma_t[:, None, None ,None] * noise
        #return {'x': x, 'z_t': z, 'noise': noise, 'mean': mean, 'mean_coef': alpha_t, 'std': sigma_t, 'var': sigma_t.pow(2), 't': t}
        return {'z_t': z, 'noise': noise, 't': t}

    def zv_to_x(self, alpha, sigma, z, v):
        return alpha[:, None, None, None] * z - sigma[:, None, None, None] * v
 
    def zv_to_eps(self, alpha, sigma, z, v):
        return sigma[:, None, None, None] * z + alpha[:, None, None, None] * v
    
    def atanexp(self, t):
        return atan(torch.exp(t))
 
    def tmin_tmax(self, dummy, logsnr_min = -15, logsnr_max = 15):
        t1 = torch.tensor([-0.5 * logsnr_max]).type_as(dummy)
        t0 = torch.tensor([-0.5 * logsnr_min]).type_as(dummy)
        t_min = self.atanexp(t1)
        t_max = self.atanexp(t0)
        return t_min, t_max
 
    # cos
    def logsnr(self, t , logsnr_min = -15 , logsnr_max =15):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return -2 * torch.log(tan( _t))
 
    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)
 
    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, 3, 32 ,32)

    def vpred(self, ut, t, model):
        return model(ut, t, torch.zeros_like(t).long())
       
    def loss_fn(self, u_0, model):
        bsz = u_0.shape[0]
        t = torch.rand(bsz,).type_as(u_0)
        D = self.diffuse(u_0, t)
        vpred = self.vpred(D['z_t'], D['t'], model)
        alpha, sigma = self.get_alpha_sigma(t)
        eps_pred = self.zv_to_eps(alpha, sigma, D['z_t'], vpred)
        mse = (eps_pred - D['noise']).pow(2).sum(-1).sum(-1).sum(-1)
        assert mse.shape == (bsz,)
        return mse


    def sample(self, n_samples, model, device):
        return self.EM(n_samples, model, device)


    def __sample(self, n_samples, model, device):
        conf = self.config
        N = conf.n_sample_steps - 1
        zt = self.sample_from_prior(n_samples).to(device)
        lowest_idx = 1
        for step in reversed ( range ( lowest_idx +1 , N+1) ) :
            # transition info
            t = (step / N) * torch.ones(n_samples,).to(device)
            s = ((step-1) / N) * torch.ones(n_samples,).to(device)
            logsnr_t = self.logsnr(t)
            logsnr_s = self.logsnr(s)
            vpred = self.vpred(zt, t, model)
            zt = self.__sampler_step(zt, t, s, vpred, logsnr_t, logsnr_s)        
        #min_logsnr = self.logsnr(lowest_idx / N * torch.ones(n_samples,).to(device))
        last_timestep = (lowest_idx / N) * torch.ones(n_samples,).to(device)
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        vpred = self.vpred(zt, last_timestep, model)
        xpred = self.zv_to_x(alpha_t, sigma_t, zt, vpred)
        #xpred = xpred.clamp(min=-1, max=1)
        return xpred

    def __sampler_step(self, zt, t, s, vpred, logsnr_t, logsnr_s):
        alpha_t, std_t = self.get_alpha_sigma(t)
        alpha_s, std_s = self.get_alpha_sigma(s)
        alpha_ts = alpha_t / alpha_s
        var_t = std_t.pow(2)
        var_s = std_s.pow(2)
        var_ts = var_t - alpha_ts.pow(2) * var_s
        xpred = self.zv_to_x(alpha_t, std_t, zt, vpred)
        
        #xpred = xpred.clamp(min=-1, max=1)
        # prediction
        # r must be var s / var t which must be exp(logsnr_t - logsnr_s)
        # 1-t must be var_ts / var_t which must be 1 - exp(logsnr_t - logsnr_s)
        weight1 = (var_s / var_t) * alpha_ts
        weight2 = (var_ts / var_t) * alpha_s
        term1 = weight1[:, None, None, None] * zt
        term2 = weight2[:, None, None, None] * xpred
        
        #term1 = (var_s  / var_t) * (alpha_ts * zt)
        #term2 = (var_ts / var_t) * (alpha_s * xpred)
        #print("term1, term2", term1.shape, term2.shape)
        #print("zt xpred", zt.shape, xpred.shape)
        #print(" wighgt", ((var_s / var_t) * alpha_ts).shape)
        #print(" wighgt2", ((var_ts / var_t) * alpha_s).shape)
        #term1 = ((alpha_ts * var_s) / var_t) * zt
        #term2 = ((alpha_s * var_ts) / var_t) * xpred
        #mu = r * mu_st + (1-r) * mu_s
        #mu = term1 + term2
        
        #mu1 = (torch.exp(logsnr_t - logsnr_s) * alpha_ts)[:,None,None,None] * zt
        #mu2 = ((1 - torch.exp(logsnr_t - logsnr_s)) * alpha_s)[:,None,None,None] * xpred
        #mu = mu1 + mu2

        mu = term1 + term2
        #std_or_var = (var_ts * var_s ) / var_t
        std = (var_ts * var_s) / var_t
        #r = torch.exp(logsnr_t - logsnr_s)
        #min_lvar = (1 - r) + nn.LogSigmoid()(-logsnr_s)
        #max_lvar = (1 - r) + nn.LogSigmoid()(-logsnr_t)
        #noise = .2
        #variance = torch.exp(noise * max_lvar + (1-noise) * min_lvar)
        #std = variance.sqrt()
        zt = mu + std[:, None, None, None] * torch.randn_like(mu)
        return zt

    def beta_fn(self, t):
        '''
        exp[- int beta t] = sig log snr t
        int beta t = -log(sigmoid(logsnr(t)))
        B(t) - B(0) = -log(sigmoid(logsnr(t)))
        d/dtB(t) - d/dtB(0) = (d/dt) -log(sigmoid(logsnr(t)))

                            = - 1/sigmoid(logsnr(t)) * (d/dt) sigmoid(logsnr(t))
                            = - 1/sigmoid(logsnr(t)) * sigmoid(logsnr(t))(1 - sigmoid(logsnr(t))) * (d/dt) logsnr(t)
                            = -(1 - sigmoid(logsnr(t))) * (d/dt)logsnr(t)
                            = -(1 - sigmoid(logsnr(t))) * (d/dt) -2 log(tan(_t))
                                                            -2 (1/tan(_t)) * (d/dt) tan(_t)
                                                                             (1/cos(_t).pow(2) * (d/dt) _t
        d/dt B(t) - d/dt B(0)
                = 
                -(1 - sigmoid(logsnr(t))) * -2 * (1/tan(_t)) * (1/_t.cos().pow(2)) * width                                           

                = 
                2 * (1 - self.logsnr(t).sigmoid()) * (1 / tan(_t)) * (1 / _t.cos().pow(2)) * width
        '''
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return 2 * (1 - self.logsnr(t).sigmoid()) * (1 / tan(_t)) * (1 / _t.cos().pow(2)) * width








    def f(self, u, t):
        return (- 0.5 * self.beta_fn(t))[:, None, None, None] * u

    def g(self, t):
        return torch.sqrt(self.beta_fn(t))

    def g2(self, t):
        return self.beta_fn(t)

    def get_fG(self, u, t):
        return self.f(u, t), self.g(t), self.g2(t)

    def EM(self, n_samples, model, device):
    
        conf = self.config
        N = conf.n_sample_steps - 1
        T_min = conf.T_min
        T_max = conf.T_max
        clip_samples = False
        tweedie = False
        n_discrete_steps = N
 
        def reverse_sde(u_t, _t, probability_flow=False):
            # _t has underscore to avoid accidentally using it instead of rev_t 
            # the correct one to use is _rev t 
            batch_size = u_t.shape[0]
            rev_t = 1 - _t
            # if doing t given s 
            #rev_s = diffusion_obj.t_to_s(rev_t)
            #std = sde.transition_std(rev_t, s=rev_s) # make sure (t, s=0) works okay
            #var = std.pow(2)
            #mean_coef = sde.transition_mean_coefficient(rev_t, s=rev_s)
            #D['std'] = std
            #D['var'] = var
            #D['mean_coef'] = mean_coef
            alpha_t, sigma_t = self.get_alpha_sigma(rev_t)
            mean_coef = alpha_t
            std_t = sigma_t
            var_t = std_t.pow(2) 
            vpred = self.vpred(u_t, rev_t, model)
            eps_hat = self.zv_to_eps(alpha_t, sigma_t, u_t, vpred)
            score_hat = - eps_hat / std_t[:, None, None, None]
            f, g, g2 = self.get_fG(u_t, rev_t)
            g2score = g2[:,None, None, None] * score_hat
            rev_drift = g2score * (0.5 if probability_flow else 1.0) - f
            rev_diff = zeros_like(g) if probability_flow else g
            return rev_drift, rev_diff
     
 
        def one_step_EM(t_scalar, dt, u, u_mean):
            n_samples = u.shape[0]
            eps = randn_like(u).type_as(u)
            t = t_scalar * ones(n_samples).type_as(u)
            drift, diffusion = reverse_sde(u, t)
            u_mean = u + drift * dt
            root_dt = torch.sqrt(dt)
            u = u_mean + (diffusion[:,None, None, None] * eps * root_dt)
            return u, u_mean
     
        # aggressive
        def clip_func(x):
            return x.clamp(min=-1, max=1)
     
        def get_sampler_t_array(N, T_begin, T_final):
            t = torch.linspace(T_begin, T_final, N + 1)
            #t = T_final * flip(1 - (t / T_final) ** 2.0, dims=[0]) # quad
            #t = t.clamp(T_begin, T_final)
            return t
     
        def main_loop(N, ts, u, u_mean, clipping):
            for i in range(N):
                if i % 500 == 0:
                    print("sampling, step {} / {}".format(i, N))
                dt = ts[i + 1] - ts[i]
                u, u_mean = one_step_EM(ts[i], dt, u, u_mean)
                if clipping:
                    u, u_mean = clip_func(u), clip_func(u_mean)
            return u, u_mean
     
        def tweedie_step_fn(u_eps, u_eps_mean, T_min, clipping):
            assert False     
            assert T_min <= 1e-3
            assert T_min <= trainer.config.delta, "bad choice for delta < T_min"
            T_min_tensor = torch.tensor([T_min]).type_as(u_eps)
     
            bsz = u_eps.shape[0]
            eps = T_min_tensor
            eps_vec = eps * torch.ones(bsz,).type_as(u_eps)
     
            std = sde.transition_std(eps, s=None)
            var = std.pow(2)
            mean_coef = sde.transition_mean_coefficient(eps, s=None)
     
            # get stheta
            score_pred_args = {
                'D': {'u_t': u_eps, 't': eps_vec, 'std': std, 'var': var, 'mean_coef': mean_coef} ,
                'mode': model,
            }
            score_hat = trainer.score_pred(**score_pred_args)
     
            #N(x | \frac{x'}{a} + \frac{beta^2}{a}s_\theta(x', eps), 
            # variance = \frac{beta^2}{a^2} I). a is mean coef, beta^2 is var
            mu_term1 = u_eps / mean_coef[:, None, None, None]
            mu_term2 = (var / mean_coef)[:, None, None, None] * score_hat
            mu = mu_term1 + mu_term2
            # only need conditional mean
            #variance = var / mean_coef.pow(2)
            #sigma = variance.sqrt()
            return mu
     
        def denoising_step_fn(u_eps, u_eps_mean, T_min, T_final, clipping):
            # t = t final = .9999 (since it will be reversed)
            # t min is like 1e-5
            #print("Denoising Step: Computing at t = ", T_final, "and dt = ", T_min)
            T_min_tensor = torch.tensor([T_min]).type_as(u_eps)
            u_0, u_0_mean = one_step_EM(T_final, T_min_tensor, u_eps, u_eps_mean)
            if clipping:
                u_0, u_0_mean = clip_func(u_0), clip_func(u_0_mean)
            return u_0, u_0_mean
     
        print("Sampling")
        #u_init = prior_sample_fn(n_samples)
        #u_init = u_init.to(device) # should not be needed but just in case
        u_init = self.sample_from_prior(n_samples).to(device)
        T_begin = 1.0 - T_max
        T_final = 1.0 - T_min
        ts = get_sampler_t_array(N=n_discrete_steps, T_begin = T_begin, T_final = T_final)
        u_init = u_init.to(device)
        ts = ts.type_as(u_init)
        with torch.no_grad():
            u_eps, u_eps_mean = main_loop(n_discrete_steps, ts, u_init, u_init, clipping = clip_samples)
     
            if tweedie:
                # dont need t final of .9999, just jump from T_min = 1e-5
                u_0_mean = tweedie_step_fn(u_eps, u_eps_mean, T_min = T_min, clipping = clip_samples)
            else:
                _, u_0_mean = denoising_step_fn(u_eps, u_eps_mean, T_min = T_min, T_final = T_final, clipping = clip_samples)
     
        # these have not yet been changed from u_0 -> x images
        return u_0_mean
















class EZConfig:
    def __init__(self,):
        self.d = 32*32*3
        self.T_min = 1e-5
        self.T_max = 1 - 1e-5
        self.prior_loc = 0.0
        self.prior_scale = 1.0
        self.n_sample_steps = 500

if __name__ == '__main__':
    bsz = 10
    conf = EZConfig()
    diff = Diffusion(conf)
    N = 100
    ts = torch.linspace(conf.T_min, conf.T_max, N+1)
    ones = torch.ones(bsz,)
    '''
    for tscalar in ts:
        t = ones * tscalar
        a,s = diff.get_alpha_sigma(t)
        print(f"mean coef:{a.mean()}", f"var:{s.pow(2).mean()}")
    '''
    ts = torch.flip(ts,dims=(0,))
    for i in range(N):
        print("t",ts[i+1], 's', ts[i])







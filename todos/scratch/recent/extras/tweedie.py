import torch
import torch.nn as nn
import numpy as np
from noise_sched import get_noise_schedule
from utils import TimeSampler
Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal

def tweedie_step_fn(u_eps, u_eps_mean, T_min, clipping):
    assert False     
    assert T_min <= 1e-3
    assert T_min <= config.delta, "bad choice for delta < T_min"
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




    # if model trains with t0, doesn't do anything, makes s=0
    # if training with ts, then makes s for t above delta, sets s to zero for below delta
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


    def get_alpha_sigma_ts(self, end, start):
        t, s = end, start
        assert torch.all(start == 0.0)
        a_t, std_t = self.get_alpha_sigma(t)
        a_s, std_s = self.get_alpha_sigma(s)
        at_sq, as_sq = a_t.pow(2), a_s.pow(2)   
        var_t, var_s = std_t.pow(2), std_s.pow(2)
        a_ts = torch.where(
            s == 0.,
            a_t,
            a_t / a_s
        )
        var_ts = torch.where(
            s == 0.,
            var_t,
            1 - (1 - var_t)/(1-var_s)
        )
        #var_ts = var_t - a_ts.pow(2) * var_s
        #var_ts = torch.where(var_t == 0., 0., 1 - (1-var_t)/(1-var_s))
        std_ts = var_ts.sqrt()
        assert not torch.any(torch.isnan(a_ts)), a_ts
        assert not torch.any(torch.isnan(std_ts)), std_ts
        #assert not torch.any(end == start)
        return a_ts, std_ts

    # for sampling, need these, but dont need full sampling process
    def get_coefs(self, t):
        s = self.t_to_s(t)
        alpha_ts, sigma_ts = self.get_alpha_sigma_ts(end = t, start = s)
        return {
            'mean_coef': alpha_ts,
            'std': sigma_ts,
            'var': sigma_ts.pow(2)
        }

    def diffuse(self, z0, t, debug = False):
        
        s = self.t_to_s(t)
        time_zero = torch.zeros_like(s)

        if debug:
            print("s", s)
            print("time zero", time_zero)

        # s | 0
        alpha_s0, sigma_s0 = self.get_alpha_sigma_ts(end = s, start = time_zero)

        if debug:

            print("alpha s0", alpha_s0)
            print("sigma s0", sigma_s0)

        zs = self.wide(alpha_s0) * z0 + self.wide(sigma_s0) * torch.randn_like(z0)

        if debug:
            print("zs", zs)

        del alpha_s0, sigma_s0
        alpha_ts, sigma_ts = self.get_alpha_sigma_ts(end = t, start = s)

        # t | s


        if debug:
            print("t", t)
            print("s", s)
            print("alpha ts", alpha_ts)
            print("sigma ts", sigma_ts)

        noise = torch.randn_like(zs)
        mean = self.wide(alpha_ts) * zs
        diff = self.wide(sigma_ts) * noise
        zt = mean + diff
        return {'z_t': zt, 'noise': noise, 't': t, 's': s, 'mean_coef': alpha_ts, 'std': sigma_ts, 'var': sigma_ts.pow(2), 'mean': mean}




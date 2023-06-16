import torch
import torch.nn as nn
import numpy as np
from noise_sched import get_noise_schedule
from utils import TimeSampler
Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal

class Diffusion:

    def __init__(self, config):
        self.config = config
        self.time_sampler = TimeSampler(mint=self.config.T_min, maxt=1.0)
        self.d_modeled = config.C_dgm * config.H_dgm * config.W_dgm
        self.T_min = config.T_min
        self.T_max = config.T_max
        self.prior_loc = 0.0
        self.prior_scale = 1.0        
        self.schedule = get_noise_schedule(config)
        self.get_mean_coef_squared = self.schedule.get_mean_coef_squared
        self.beta_fn = self.schedule.beta_fn
        self.int_beta_fn = self.schedule.int_beta_fn

    def forward(self, t):
        return None

    def div_f(self, u, t):
        return -0.5 * self.beta_fn(t) * self.d_modeled

    def triple_sum(self, x):
        return x.sum(-1).sum(-1).sum(-1)

    def sum_image(self, x):
        return self.triple_sum(x)
    
    def sqnorm_image(self, x):
        return self.triple_sum(x.pow(2))

    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)
 
    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        config = self.config
        C, H, W = config.C_dgm, config.H_dgm, config.W_dgm
        z = p.sample(n_samples, self.d_modeled)
        return z.view(n_samples, C, H, W)
         
    def prior_logp(self, u):
        return self.sum_image(self.get_prior_dist().log_prob(u))
         
    def cross_ent_helper(self, z0):
        bsz = z0.shape[0]
        T = ones(bsz).type_as(z0) * self.config.T_max
        a, s = self.get_alpha_sigma(T)
        u_T = self.wide(a) * z0 + self.wide(s) * torch.randn_like(z0)
        lp = self.prior_logp(u_T)
        assert lp.shape==(bsz,)
        return lp

    def get_alpha_sigma(self, t):
        alpha_t = self.get_mean_coef(t)
        sigma_t = self.get_std(t)
        return alpha_t, sigma_t
 
    def get_coefs(self, t):
        a, s = self.get_alpha_sigma(t)
        return {
            'mean_coef': a,
            'std': s,
            'var': s.pow(2)
        }

    def diffuse(self, z0, t, debug = False):
        a,s = self.get_alpha_sigma(t)
        noise = torch.randn_like(z0)
        mean = self.wide(a) * z0
        diff = self.wide(s) * noise
        zt = mean + diff
        return {
            'z_t': zt, 'noise': noise, 't': t, 's': s, 
            'mean_coef': a, 'std': s, 'var': s.pow(2), 'mean': mean
        }

    def wide(self, x):
        return x[:, None, None, None]

    def zv_to_x(self, alpha, sigma, z, v):
        return self.wide(alpha) * z - self.wide(sigma) * v
 
    def zv_to_eps(self, alpha, sigma, z, v):
        return self.wide(sigma) * z + self.wide(alpha) * v
     
    def vpred(self, ut, t, y, model):
        return model(ut, t, y)

    def epspred(self, ut, t, y, model, alpha, sigma):
        vpred = self.vpred(ut, t, y, model)
        return self.zv_to_eps(alpha, sigma, ut, vpred)

    def scorepredmodel(self, ut, t, y, model, alpha, sigma):
        return model(ut, t, y)

    def neg_dsm(self, D, noise_diff):
        z_t, t, noise, std, variance = D['z_t'], D['t'], D['noise'], D['std'], D['var']
        g2 = self.g2(D['t'])
        score_term = self.sqnorm_image(-noise / self.wide(std))
        dsm1 = -0.5 * (g2 / variance) * noise_diff
        dsm2 = 0.5 * g2 * score_term
        dsm3 = self.div_f(z_t, t)
        neg_dsm = dsm1 + dsm2 + dsm3
        return neg_dsm
               
    def get_mean_coef(self, t):
        return torch.sqrt(self.get_mean_coef_squared(t))

    def get_variance(self, t):
        return 1 - self.get_mean_coef_squared(t)

    def get_std(self, t):
        return torch.sqrt(self.get_variance(t))
    
    def sample(self, n_samples, model, device):
        return self.EM(n_samples, model, device)
    
    def f(self, u, t):
        return self.wide(- 0.5 * self.beta_fn(t)) * u

    def g(self, t):
        return torch.sqrt(self.beta_fn(t))

    def g2(self, t):
        return self.beta_fn(t)

    def get_fG(self, u, t):
        return self.f(u, t), self.g(t), self.g2(t)

    def EM(self, n_samples, model, device):

        config = self.config                                                                                                                      
        N = config.n_sample_steps - 1
        n_discrete_steps = N
        #config.label_keep_prob
          
        def reverse_sde(u_t, t, y, probability_flow=False):

            REVT = 1-t

            coefs = self.get_coefs(REVT)
            eps_hat = self.epspred(u_t, REVT, y, model, coefs['mean_coef'], coefs['std'])
            score_hat = - eps_hat / self.wide(coefs['std'])

            f, g, g2 = self.get_fG(u_t, REVT)
            g2score = self.wide(g2) * score_hat
            rev_drift = g2score * (0.5 if probability_flow else 1.0) - f
            rev_diff = zeros_like(g) if probability_flow else g
            return rev_drift, rev_diff 
        
        def one_step_EM(zt, zt_mean, y, t_scalar, dt):
            noise = torch.randn_like(zt)
            t = t_scalar * torch.ones(zt.shape[0]).type_as(zt)
            drift, diffusion = reverse_sde(zt, t, y)
            zt_mean = zt + drift * dt
            diff_term = self.wide(diffusion) * noise * torch.sqrt(dt)
            return {'zt':zt_mean + diff_term, 'zt_mean':zt_mean}
            
        def main_loop(N, ts, zt, zt_mean, y):
            ret = {'zt': zt, 'zt_mean': zt_mean}
            for i in range(N):
                if i % 500 == 0:
                    print("sampling, step {} / {}".format(i, N))
                dt = ts[i + 1] - ts[i]
                ret = one_step_EM(ret['zt'], ret['zt_mean'], y, ts[i], dt)
            return ret                 
 
        def _sample_main():
            u_init = self.sample_from_prior(n_samples).to(device)
            TMIN = config.T_min
            TMAX = config.T_max
            ts = torch.linspace(TMIN, TMAX, n_discrete_steps + 1)
            u_init = u_init.to(device)
            ts = ts.type_as(u_init)
            high = config.num_classes + 1 if config.label_drop else config.num_classes
            if config.debug_cfg:
                y = torch.zeros_like(y)
            else:
                y = torch.randint(low=0, high=high, size=(n_samples,), device=device).long()

            with torch.no_grad():
                ret = main_loop(n_discrete_steps, ts, u_init, u_init, y)
                ret = one_step_EM(
                    ret['zt'], 
                    ret['zt_mean'], 
                    y, 
                    ts[-1], 
                    torch.tensor([TMIN]).type_as(u_init)
                )
            return ret

        return _sample_main()


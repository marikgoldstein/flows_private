import torch
import torch.nn as nn
import numpy as np
from noise_sched import get_noise_schedule
from utils import TimeSampler
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, 
    sqnorm
)
Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal
ones_like = torch.ones_like
atan = torch.atan
tan = torch.tan
Cat = torch.distributions.Categorical
Mix = torch.distributions.MixtureSameFamily
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

    def to_image_shape(self, x, bsz):
        config = self.config
        C, H, W = config.C_dgm, config.H_dgm, config.W_dgm
        return x.view(bsz, C, H, W)

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
        return self.to_image_shape(p.sample((n_samples, self.d_modeled)), n_samples)
         
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
        return {'z_t': zt, 'noise': noise, 't': t, 's': s, 'mean_coef': a, 'std': s, 'var': s.pow(2), 'mean': mean}

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

    def empirical_score_autograd(self, zt, tscalar, ref_set, encoder, max_points = None):

     
        config = self.config
        C, H, W = config.C_dgm, config.H_dgm, config.W_dgm

        # bsz C H W ->  C H W bsz
        def batch_last(x):
            return x.permute(1, 2, 3, 0)
         
        # C H W bsz -> bsz C H W
        def batch_first(x):
            return x.permute(3, 0, 1, 2)


        alpha_t, sigma_t = self.get_alpha_sigma(tscalar)
        bsz, alpha, S = zt.shape[0], [], []
        assert zt.shape == (bsz, C, H, W)

        if max_points is None:
            num_points = len(ref_set)
        else:
            num_points = max_points

        mus = []
        stds = []
        for j, data_j in enumerate(ref_set):
            if max_points is not None and j >= max_points:
                break
            xj, yj = data_j       
            z0j, _ = encoder.scaler(xj.to(zt.device))
            muj = alpha_t * z0j   
            stdj = sigma_t 
            assert muj.shape == (C, H, W)       
            mus.append(muj)     
            assert stdj.shape == (1,)
            stds.append(stdj * torch.ones_like(muj))
        mus = torch.stack(mus, dim=-1)
        stds = torch.stack(stds, dim=-1)
        assert mus.shape == (C, H, W, num_points)
        assert stds.shape == (C, H, W, num_points)
        pis = torch.ones_like(mus) / num_points
        assert pis.shape == (C, H, W, num_points)
        pisum = pis.sum(-1)
        assert torch.allclose(pisum, torch.ones_like(pisum))
        gmm = Mix(Cat(pis), Normal(mus,stds))
        #testx=gmm.sample()
        #assert testx.shape == (C, H, W)
        zt_clone = zt.clone()
        zt_clone.requires_grad = True
        lp = gmm.log_prob(zt_clone)
        score_t = torch.autograd.grad(lp.sum(), zt_clone)[0]
        score_t = torch.nan_to_num(score_t, nan=-3e+30, neginf=-3e+30, posinf=3e+30)
        assert score_t.shape == (bsz, C, H, W)
        return score_t

    def empirical_score(self, zt, tscalar, ref_set, encoder, max_points = None):

        config = self.config
        C, H, W = config.C_dgm, config.H_dgm, config.W_dgm

        alpha_t, sigma_t = self.get_alpha_sigma(tscalar)
        bsz, alpha, S = zt.shape[0], [], []
        assert zt.shape == (bsz, C, H, W)

        if max_points is None:
            num_points = len(ref_set)
        else:
            num_points = max_points

        for j, data_j in enumerate(ref_set):
            if max_points is not None and j >= max_points:
                break
            xj, yj = data_j       
            z0j, _ = encoder.scaler(xj.to(zt.device))
            muj = alpha_t * z0j   
            stdj = sigma_t        
            varj = stdj.pow(2)    
            wj = 1 / num_points   
            lp = torch.distributions.Normal(   
                loc = muj,        
                scale = stdj      
            ).log_prob(zt)       
            assert lp.shape == (bsz, C, H, W)
            lp = lp.sum(-1).sum(-1).sum(-1)    
            assert lp.shape == (bsz,)                
            prob = lp.exp()                                                                                                                                                                       
            aj = torch.log(wj * prob)
            aj = torch.nan_to_num(aj, nan=-3e+30, neginf=-3e+30, posinf=3e+30)
            assert aj.shape == (bsz,)
            alpha.append(aj)
            Sij = -(zt - muj) / varj
            assert Sij.shape == (bsz, C, H, W)
            S.append(Sij)
        
        alpha = torch.stack(alpha, dim=-1)
        S = torch.stack(S, dim=-1)
        assert alpha.shape == (bsz, num_points)
        assert S.shape == (bsz, C, H, W, num_points)
        S = S.view(bsz, C*H*W, num_points)   
        assert S.shape == (bsz, C*H*W, num_points)
        Delta = torch.nn.Softmax(dim=-1)(alpha)
        score = (S * Delta[:, None, :])        
        assert score.shape == (bsz, C*H*W, num_points)
        score = score.sum(-1)     
        assert score.shape == (bsz, C*H*W)   
        score = score.view(bsz, C, H, W)
        return score

    def loss_fn_empirical(self, z0, y, model, ref_set, encoder):
        assert self.config.empirical
        config = self.config
        C, H, W = config.C_dgm, config.H_dgm, config.W_dgm
        bsz = z0.shape[0]
        t = torch.ones(bsz,).type_as(z0) * torch.rand(1).type_as(z0)
        D = self.diffuse(z0, t)
        score_t = self.empirical_score(D['z_t'], t[0], ref_set, encoder, max_points = None) #self.config.empirical_points)
        assert score_t.shape == (bsz, C, H, W)
        alpha, sigma = D['mean_coef'], D['std']
        score_hat = self.scorepredmodel(D['z_t'], D['t'], y, model, alpha, sigma)
        loss = self.sqnorm_image(score_t - score_hat)
        return loss

    def loss_fn(self, z0, y, model, device, loss_type):

        assert loss_type in ['noise_pred', 'nelbo']
        bsz = z0.shape[0]
        t, time_weight = self.time_sampler(bsz, z0.device)

        D = self.diffuse(z0, t)
        alpha, sigma = D['mean_coef'], D['std']
        eps_hat = self.epspred(D['z_t'], D['t'], y, model, alpha, sigma)
        noise_diff = self.sqnorm_image(eps_hat - D['noise'])
        
        if loss_type == 'noise_pred':
            assert noise_diff.shape == (bsz,)
            loss = time_weight * noise_diff
        else:
            prior = self.cross_ent_helper(z0)
            int_term = time_weight * self.neg_dsm(D, noise_diff)
            nelbo = -(prior + int_term)
            loss = nelbo

        return loss
               
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
            if self.config.empirical:
                score_hat = self.scorepredmodel(u_t, REVT, y, model, coefs['mean_coef'], coefs['std'])
            else:
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
 
        def _main():
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
                ret = one_step_EM(ret['zt'], ret['zt_mean'], y, ts[-1], torch.tensor([TMIN]).type_as(u_init))['zt_mean']
            return ret

        return _main()



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


'''
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
'''


'''
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

'''



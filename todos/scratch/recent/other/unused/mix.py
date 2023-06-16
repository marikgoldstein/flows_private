import math
import time 
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from typing import Optional, List
from tqdm.auto import tqdm
import seaborn as sns
# local
#from importance_sampling import VPImpSamp
from utils_numerical import (
    cat, chunk, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp, 
    eye,
    batch_transpose,
    trace_fn
)
from utils import check, param_no_grad
from torch.distributions import Normal
import torch.distributions as D
import matplotlib.pyplot as plt 
from statsmodels.nonparametric.kde import KDEUnivariate

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

class BModule(nn.Module):
    
    def __init__(self, b0, b1):
        super().__init__()
        self._b0 = nn.Parameter(torch.tensor([b0]), requires_grad=False)
        self._b1 = nn.Parameter(torch.tensor([b1]), requires_grad=False)
        self.const = .1
        #b0 = 0.0001
        #b1 = 0.02
        #lambda t: math.cos((t * 0.008) / 1.0008 * math.pi / 2) ** 2

    def get_b0_b1(self,):
        return self._b0, self._b1

    def beta_fn(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0 + t*(b1 - b0)
        #return self.const * torch.ones_like(t)

    def int_beta_fn(self, t):
        b0, b1 = self.get_b0_b1()
        b0, b1 = b0.type_as(t), b1.type_as(t)
        return b0*t + (b1-b0)*(t**2/2)
        #return t

class VP(nn.Module):
    
    def __init__(self, d, max_beta, device):
        super().__init__()
        self.d = d
        self.devide = device
        self.max_noise = max_beta
        self.prior_loc = param_no_grad(zeros(1).to(device)) #nn.Parameter(zeros(1).to(device),requires_grad=False)
        self.prior_scale = param_no_grad(ones(1).to(device)) #nn.Parameter(ones(1).to(device), requires_grad=False)
        self.bmodule = BModule(b0=.1, b1=self.max_noise)
    
    def forward(self, u, t, s = None):
        #{'u_t':u_t, 'eps':eps, 'mean_coef':mean_coef, 'std':std, 'mean':mean}
        D = self.sample_from_transition_kernel(u, t, s = s)
        return D

    def get_prior_dist(self,):
        return Normal(loc=self.prior_loc, scale=self.prior_scale)

    def prior_logp(self, u):
        return self.get_prior_dist().log_prob(u).sum(-1)

    def sample_from_prior(self, n_samples):
        p = self.get_prior_dist()
        return p.sample((n_samples, self.d)).view(n_samples, -1)

    def beta_fn(self,t):
        return self.bmodule.beta_fn(t)

    def int_beta_fn(self,t):
        return self.bmodule.int_beta_fn(t)

    def get_fG(self, u , t):
        return self.f(u, t), self.G(t), self.G2(t)

    # output bsz by D
    def f(self, u, t):
        return -0.5 * self.beta_fn(t)[...,None] * u

    # output bsz,
    def G(self, t):
        return sqrt(self.beta_fn(t))

    # output bsz,
    def G2(self, t):
        return self.beta_fn(t)

    # output bsz,
    def div_f(self, u, t):
         return -0.5 * self.beta_fn(t) * self.d

    # output bsz
    def transition_mean_coefficient(self, t, s=None):
        bt = self.int_beta_fn(t)
        bs = 0.0 if s is None else self.int_beta_fn(s)
        return torch.exp(-0.5 * (bt - bs))

    def transition_mean(self, u, t, s=None):
        coef = self.transition_mean_coefficient(t, s=s)
        return coef[:,None] * u, coef

    # output bsz,
    def transition_var(self, t, s=None):
        bt = self.int_beta_fn(t)
        bs = 0.0 if s is None else self.int_beta_fn(s)
        return 1 - torch.exp(-(bt-bs))

    # output bsz,
    def transition_std(self, t, s=None):
        return sqrt(self.transition_var(t, s=s))

    def sample_from_transition_kernel(self, u, t, s=None):
        bsz = u.shape[0]
        mean, mean_coef = self.transition_mean(u, t, s=s)
        std = self.transition_std(t, s=s)
        noise = torch.randn_like(u)
        u_t = mean + noise * std[:, None]
        return {'u': u, 'u_t':u_t, 'noise':noise, 'mean_coef':mean_coef, 'std':std, 'var': std.pow(2), 'mean':mean, 't': t, 's': s}
    
    # ouput bsz,
    def cross_ent_helper(self, u_0):
        bsz = u_0.shape[0]
        T = ones(bsz).type_as(u_0)
        u_T = self.sample_from_transition_kernel(u_0, T, s=None)['u_t']
        lp = self.prior_logp(u_T)
        assert lp.shape==(bsz,)
        return lp

def alpha_t_to_s(alpha, t):
    s = alpha * (t) + (1-alpha) * torch.zeros_like(t)
    return s

def plot(x, label, color):
    sns.kdeplot(x.numpy(), bw_method=0.5, label=label, color=color)



class GMM_VP:
    
    def __init__(self):

        self.bsz=10000
        self.d=1
        self.vp = VP(d=self.d, max_beta = 1.0, device=torch.device('cpu'))
        self.K = 2
        self.pi = torch.ones(self.K) / self.K
        self.mus = torch.tensor([-10., 10.])
        #1000, 1000.])
        self.stds = torch.tensor([1.0, 1.0])
        self.q0 = D.MixtureSameFamily(
            D.Categorical(self.pi),
            D.Normal(self.mus, self.stds)
        )


    def reset_means(self, mean_vec):

        self.mus = mean_vec
        self.q0 = D.MixtureSameFamily(
            D.Categorical(self.pi),
            D.Normal(self.mus, self.stds)
        )

    def sample_q0(self, bsz=None):
        if bsz is None:
            bsz = self.bsz
        return self.q0.sample(sample_shape=(bsz,))


    def check_sigmas(self,):

        bsz = 1
        num_t = 10
        ts = torch.linspace(.1, .99, num_t)
        delta = 0.1
        t_to_s = lambda t: t - (delta / 2.0)
        for tscalar in ts:
            t = torch.ones(bsz,) * tscalar
            s = t_to_s(t)
            var_ts = self.vp.transition_var(t=t, s=s)
            var_t0 = self.vp.transition_var(t=t, s=None)
            print("t:{}".format(round(tscalar.item(),3)), 
                "var_ts: {}".format(round(var_ts.item(),3)), 
                "var_ts^2: {}".format(round(var_ts.pow(2).item(),3)), 
                'var_t0: {}'.format(round(var_t0.item(),3))
            )



    def test3(self,):

        num_t = 16
        ts = torch.linspace(.1, 1, num_t)
        bsz = 1
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(4, 4, sharey=True,figsize=(8, 8))
        #fig.subplots_adjust(wspace=0)

        x0 = self.sample_q0(bsz=bsz).unsqueeze(-1)
        for t_idx, tscalar in enumerate(ts):
            
            i = int(t_idx / 4)
            j = t_idx % 4

            t = torch.ones(bsz) * tscalar            
            xt = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']
            q0_given_t = self.q_xs_given_xt(xt.squeeze(-1), s=torch.zeros_like(t)+0.001, t=t)
            samps = []
            reps=1000
            for rep in range(reps):
                x0_k = q0_given_t.sample()
                samps.append(x0_k)
            samps = torch.stack(samps, dim=0).squeeze(-1)
            xgrid = np.linspace(-30, 30, 100)
            pdf = kde_statsmodels_u(samps.numpy(), xgrid, bandwidth=0.2)
            ax[i,j].plot(xgrid, pdf, color='blue',
                        alpha=0.5, lw=3, label='xt kde samples')
            #ax[0,t_idx].fill(xgrid, y, pdftrue_numpy, ec='gray',
            #            fc='gray', alpha=0.4, label='xt computed_pdf')
            ax[i,j].set_xlim(-30.0, 30.0)
            #t = tscalar * torch.ones(samps.shape[0])
            #plt.scatter(samps.numpy(), t.numpy(), label=f'{tscalar}')

            title = str(round(tscalar.item(),3))
            ax[i,j].set_title(title)
        plt.tight_layout()
        plt.show()


    def test2(self,):

        num_t = 10
        ts = torch.linspace(.1, .95, num_t)
        bsz = 256
        ts_numpy = ts.numpy()

        sns.set_style('whitegrid')
        fig, ax = plt.subplots(4, 4, sharey=False,figsize=(8, 8))

        for i, mean_mult in enumerate([0.1, 1.0, 10.0, 100.0]):
            for j, max_noise in enumerate([1.0, 10.0, 100.0, 1000.0]): 
                
                self.vp = VP(d=self.d, max_beta = max_noise, device=torch.device('cpu'))
                vs = []
                means = torch.tensor([-1.0, 1.0])
                means = mean_mult * means
                self.reset_means(means)

                for t_idx, tscalar in enumerate(ts):
                    x0 = self.sample_q0(bsz=bsz).unsqueeze(-1)
                    t = torch.ones(bsz) * tscalar            
                    xt = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']
                    del x0
                    MC = 100
                    scores = []
                    for k in range(MC):
                        q0_given_t = self.q_xs_given_xt(xt.squeeze(-1), s=torch.zeros_like(t), t=t)
                        x0_k = q0_given_t.sample().unsqueeze(-1)
                        dict_t_given_0  = self.vp.sample_from_transition_kernel(x0_k, t=t, s=None)
                        mu_t0, var_t0 = dict_t_given_0['mean'], dict_t_given_0['var']
                        assert xt.shape == (bsz, 1)
                        assert mu_t0.shape == (bsz,1)
                        assert var_t0.shape == (bsz,)
                        #score = -(xt - mu_t0) / var_t0[:, None]
                        #assert score.shape == (bsz, 1)
                        #scores.append(score)
                        scores.append(x0_k)
                    #E_{x_t} [var{x_s | x_t} (s_{t | s)}]
                    scores = torch.stack(scores, dim=0)
                    assert scores.shape == (MC, bsz, 1)
                    scores = scores.squeeze(-1)
                    assert scores.shape == (MC, bsz)
                    variances = scores.var(dim=0)
                    assert variances.shape == (bsz,)
                    expected_variance = variances.mean()
                    vs.append(expected_variance.item())
                    print(f"t:{tscalar}, var:{expected_variance}")
                title = f'mult:{mean_mult},b1:{max_noise}'
                ax[i,j].set_title(title)
                ax[i,j].plot(ts_numpy, vs)

        plt.xlabel("t")
        plt.ylabel('var score')
        plt.tight_layout()
        plt.show()


    def test(self,):

        num_t = 10
        num_s = 10
        heatmap = torch.zeros(num_t, num_s)

        ts = torch.linspace(.1, .95, num_t)
        alphas = torch.linspace(0, .99, num_s)
        bsz = 128
        for t_idx, tscalar in enumerate(ts):
            print("------t is {} --------".format(round(tscalar.item(),3)))
            x0 = self.sample_q0(bsz=bsz).unsqueeze(-1)
            t = torch.ones(bsz) * tscalar
            xt = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']
            for s_idx, alpha in enumerate(alphas):
                s = alpha_t_to_s(alpha, t)
                #s = torch.ones(bsz) * sscalar
                #E_{x_t} [var{x_s | x_t} (s_{t | s)}]
                MC = 100
                scores = []
                for k in range(MC):
                    # sample xsk |xt
                    dist_st = self.q_xs_given_xt(xt.squeeze(-1), s, t)
                    xs_k = dist_st.sample().unsqueeze(-1)
                    _dict = self.vp.sample_from_transition_kernel(xs_k, t=t, s=s)
                    mu_ts, var_ts = _dict['mean'], _dict['var']
                    #m_ts = _dict['mean_coef']
                    #print("xt", xt.shape)
                    #print("muts",mu_ts.shape)
                    assert xt.shape == (bsz, 1)
                    assert mu_ts.shape == (bsz,1)
                    assert var_ts.shape == (bsz,)
                    score = -(xt - mu_ts) / var_ts[:, None].sqrt() # / var_ts[:, None]
                    assert score.shape == (bsz, 1)
                    scores.append(score)
                #E_{x_t} [var{x_s | x_t} (s_{t | s)}]
                scores = torch.stack(scores, dim=0)
                assert scores.shape == (MC, bsz, 1)
                scores = scores.squeeze(-1)
                assert scores.shape == (MC, bsz)
                variances = scores.var(dim=0)
                assert variances.shape == (bsz,)
                expected_variance = variances.mean()

                heatmap[t_idx, s_idx] = expected_variance
                print("s:{}, expected variance {}".format(round(s[0].item(),3), round(expected_variance.item(),4)))


        heatmap = heatmap.log()
        ts = [round(t,3) for t in ts.numpy()]
        alphas = [round(a,3) for a in alphas.numpy()]
        plt.yticks(list(range(len(ts))), ts)
        plt.xticks(list(range(len(alphas))),alphas)
        plt.imshow(heatmap.numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.xlabel("alphas")
        plt.ylabel("t values")
        plt.show()
        '''        
        bsz = 1
        tscalar = 1.5
        for col in ['red', 'blue']:
            t = torch.ones(bsz,) * tscalar
            x0 = self.sample_q0(bsz=bsz).unsqueeze(-1)
            xt = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t'].squeeze(-1)
            for alpha in alphas:
                s = alpha_t_to_s(alpha, t)
                sscalar = s[0].item()
                dist = self.q_xs_given_xt(xt, s, t)
                x = xt.numpy()
                y = np.ones((1,)) * tscalar
                plt.scatter(x, y, color=col)
                xs, ys = [], []
                for i in range(10):
                    samp = dist.sample()
                    x = samp.numpy()
                    y = s.numpy()
                    xs.append(x)
                    ys.append(y)
                plt.scatter(xs, ys, color=col, alpha=0.1)
            plt.scatter(x0.numpy(), np.ones((1,))*0.0, color=col)
        #plt.legend()
        plt.ylim(-.1,1.1)

        svals = [round(alpha_t_to_s(alpha,t)[0].item(),3) for alpha in alphas]
        ticks = svals + [tscalar]
        labels = ['Xs={}|Xt'.format(s,tscalar) for s in svals] + ['xt={}'.format(tscalar)]
        plt.yticks(ticks, labels)
        plt.show()
        '''
        
    def q_xs_given_xt(self, xt, s, t):

        bsz = xt.shape[0]
        assert xt.shape == (bsz,)
        assert s.shape == (bsz,)
        assert t.shape == (bsz,)

        weights = []
        mus = []
        stds = []

        for k in range(self.K):

            pi_k = self.pi[k]
            mu_k = self.mus[k]
            std_k = self.stds[k]
            var_k = std_k.pow(2)
            m_ts = self.vp.transition_mean_coefficient(t=t, s=s)
            m_s0 = self.vp.transition_mean_coefficient(t=s, s=None)
            #print("m ts", m_ts)
            #print("m s0", m_s0)
            var_ts = self.vp.transition_var(t=t, s=s)
            var_s0 = self.vp.transition_var(t=s, s=None)

            loc = m_ts * m_s0 * mu_k
            var = var_ts + m_ts.pow(2) * (m_s0.pow(2) * var_k + var_s0)
            std = var.sqrt()
            N = Normal(loc, std)
            w = pi_k * N.log_prob(xt).exp()
            assert w.shape == (bsz,)
            weights.append(w)
            # ..................
            loc_term1 = m_s0 * mu_k
            loc_term2 = (m_ts * (m_s0.pow(2) * var_k + var_s0)) / (var_ts + m_ts.pow(2) * (m_s0.pow(2)*var_k + var_s0)) * (xt - m_ts * m_s0 * mu_k)
            loc = loc_term1 + loc_term2
            mus.append(loc)
            var_term3_numer = m_ts.pow(2) * (m_s0.pow(2)*var_k + var_s0).pow(2)
            var_term3_denom = var_ts + m_ts.pow(2)*(m_s0.pow(2) * var_k + var_s0)
            var_term3 = var_term3_numer / var_term3_denom
            var = m_s0.pow(2) * var_k + var_s0 - var_term3
            std = var.sqrt()
            stds.append(std)
    
        weights = torch.stack(weights, dim=-1)
        weights = weights / weights.sum(-1)[:, None]
        mus = torch.stack(mus, dim=-1)
        stds = torch.stack(stds, dim=-1)
        mix = D.Categorical(weights)
        comp = D.Normal(mus, stds)
        dist = D.MixtureSameFamily(mix, comp)
        return dist

    def marginal_density(self, t):

        assert len(t.shape) == 1
        num_points = t.shape[0]
        xgrid = torch.linspace(-30, 30, num_points)
        assert len(xgrid.shape) == 1
        mean_coef_t0 =self. vp.transition_mean_coefficient(t=t, s=None)
        var_t0 = self.vp.transition_var(t=t, s=None)
        summ = 0.0
        for k in range(self.K):
            pi_k = self.pi[k]
            mu_k = self.mus[k]
            std_k = self.stds[k]
            q0k = Normal(loc = mean_coef_t0 * mu_k,
                       scale = (mean_coef_t0.pow(2) * std_k.pow(2) + var_t0).sqrt()
            )
            summ += pi_k * q0k.log_prob(xgrid).exp()

        px = summ
        return xgrid, px

    def compare(self,):

        x0 = self.sample_q0().unsqueeze(-1)
        bsz = x0.shape[0]
        assert x0.shape == (bsz, 1)
        tscalar = 0.3
        t = torch.ones(bsz) * tscalar
        xgrid, px = self.marginal_density(t)
        xt_samples = self.vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']

        xt_numpy = xt_samples.numpy()
        xgrid_numpy = xgrid.numpy()
        pdf_numpy = kde_statsmodels_u(xt_numpy, xgrid_numpy, bandwidth=0.2)
        pdftrue_numpy = px.numpy()

        sns.set_style('whitegrid')
        fig, ax = plt.subplots(1, 1, sharey=True,figsize=(13, 3))
        fig.subplots_adjust(wspace=0)
        ax.plot(xgrid_numpy, pdf_numpy, color='blue', 
                        alpha=0.5, lw=3, label='xt kde samples')
        ax.fill(xgrid_numpy, pdftrue_numpy, ec='gray', 
                        fc='gray', alpha=0.4, label='xt computed_pdf')
        ax.set_xlim(-30.0, 30.0)
        plt.legend()
        plt.show()

if __name__=='__main__':

    gmm_vp = GMM_VP()
    #gmm_vp.compare()
    #gmm_vp.test()
    #gmm_vp.check_sigmas()
    gmm_vp.test2()
    #gmm_vp.test3()


'''

mu0 = torch.ones(bsz, 1) * 10.0
v0 = torch.ones(bsz,)
eps = torch.randn_like(mu0)
x0 = mu0 + v0.sqrt()[:, None] * eps

alphas = torch.linspace(0, 1, 10)

for alpha in alphas:

    s = alpha_t_to_s(alpha, t)
    
    # marginal samples of xt
    xt = vp.sample_from_transition_kernel(x0, t=t, s=None)['u_t']
    mu_st, var_st = vp.s_given_t(mu0, v0, s, t, xt)
    std_st = var_st.sqrt()
    MC = 1000
    scores = []
    for k in range(MC):
        # score(xt | xs_k) for xs_k's sampled from xs|xt
        
        # sample xs_k | xt
        eps = torch.randn_like(mu_st)
        xs = mu_st + std_st[:, None] * eps

        # need mean and var of t|s to compute score at the previous sampled (xt, xs_k)
        t_given_s_dict = vp.sample_from_transition_kernel(xs, t=t, s=s)
        mu_ts, var_ts = t_given_s_dict['mean'], t_given_s_dict['var']
        score = -(xt - mu_ts) / var_ts[:, None]
        assert score.shape == (bsz, 1)
        scores.append(score)
    #E_{x_t} [var{x_s | x_t} (s_{t | s)}]
    scores = torch.stack(scores, dim=0)
    assert scores.shape == (MC, bsz, 1)
    scores = scores.squeeze(-1)
    variances = scores.var(dim=0)
    assert variances.shape == (bsz,)
    expected_variance = variances.mean()
    print("s:{}, expected variance {}".format(round(s[0].item(),3), round(expected_variance.item(),4)))







    # gaussian base 
    def s_given_t(self, mu0, v0, s, t, xt):
        assert False
        m_s0 = self.transition_mean_coefficient(s, s=None).unsqueeze(-1)
        m_ts = self.transition_mean_coefficient(t, s=s).unsqueeze(-1)
        v0 = v0.unsqueeze(-1)
        
        v_s0 = self.transition_var(s, s=None).unsqueeze(-1)
        v_ts = self.transition_var(t, s=s).unsqueeze(-1)

        m_s0_sq = m_s0.pow(2)
        m_ts_sq = m_ts.pow(2)

        # mu st
        a = (m_ts*(v_s0 + m_s0_sq * v0))
        b = v_ts + m_ts_sq * (v_s0 + m_s0_sq * v0)
        c = (xt - m_ts * m_s0 * mu0)
        mu_st = m_s0 * mu0 + (a/b) * c

        # var st
        a = v_s0
        b = v0 * m_s0_sq
        cnumer = m_ts_sq * (v_s0 + m_s0_sq * v0).pow(2)
        cdenom = v_ts + m_ts_sq * (v_s0 + m_s0_sq * v0)
        c = cnumer / cdenom
        var_st = a + b - c

        var_st = var_st.squeeze(-1)
        return mu_st, var_st

'''

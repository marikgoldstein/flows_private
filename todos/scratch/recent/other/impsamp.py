import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)                                                                                                     
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)
'''
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

'''

class ImpSamp:
    
    def __init__(self,):

        self.b0 = 0.1
        self.b1 = 20.0
        self.tmax = torch.tensor([1.0])
        self.tmin = torch.tensor([1e-3])

    def beta_fn(self, t):
        b0 = self.b0
        b1 = self.b1
        return b0 + (b1-b0)* t

    def int_beta_fn(self, t):
        b0 = self.b0
        b1 = self.b1
        return b0 * t + (b1 - b0) * 0.5 * t.pow(2)

    def g2(self, t):
        return beta_fn(t)

    def meancoef(self, t): 
        ibt = int_beta_fn(t)
        return torch.exp(-0.5 * ibt)

    def variance(self, t):
        ibt = int_beta_fn(t)
        return 1 - torch.exp( - ibt)

    # _r at 1e-5
    #def truncation_val(self, t):
    #    return torch.ones_like(t) * 9.7590e+10

    def _r(self, t):
        return g2(t) * meancoef(t).pow(2) / variance(t).pow(2)

    def r(self, t):
        #lower = truncation_val(t) # wont eval below 1r-5, but just in case putting this here
        #upper = _r(t)
        #return torch.where(t >= self.tmin, upper, lower)
        assert t >= self.tmin
        return _r(t)


    def endpoint(self, t):
        term1 = -199/20 * t.pow(2)
        term2 = -t/10
        return 1 - ((-199/20) * t.pow(2) - t/10).exp()

    def cumulative_weight(self, t):
        #u = 1 - exp[A] where A  = -(199/20)t^2  - t/10
        upper = self.endpoint(t)
        lower = self.endpoint(torch.ones_like(t) * self.tmin)
        return (-1 / upper ) - (-1 / lower)

    def Z(self,):
        return self.cumulative_weight(t=self.tmax)

    def weight(self, t):
        return self.r(t) / self.Z()

    def sample(self, N, quantile=None, steps=100):

        Z = self.cumulative_weight(self.tmax)        
        if quantile is None:
            quantile = torch.rand(N) * (Z - 0) + 0
        lb = torch.ones_like(quantile) * self.tmin
        ub = torch.ones_like(quantile) * self.tmax

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.0
            value = self.cumulative_weight(mid)
            lb = torch.where(value <= quantile, mid, lb)
            ub = torch.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        carry = (lb, ub)
        for i in range(steps):
            carry, _ = bisection_func(carry, i)
        (lb, ub) = carry
        return (lb + ub) / 2.0

sampler = ImpSamp()
samples = sampler.sample(N=4096)
samples = samples.numpy()
plt.hist(samples, bins=1000)
plt.show()




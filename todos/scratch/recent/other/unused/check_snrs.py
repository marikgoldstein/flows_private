import torch
import numpy as np
import matplotlib.pyplot as plt


b0 = 0.1
b1 = 20.0
bconst = 10.0
delta = 0.1




def linear_beta_fn(t):
    return b0 + (b1 - b0) * t

def const_beta_fn(t):
    return bconst

def int_linear_beta_fn(t):
    return b0 * t + (b1-b0) * 0.5 * t**2

def int_const_beta_fn(t):
    return bconst * t

def linear_meancoef(t):
    return torch.exp(-0.5 * (int_linear_beta_fn(t)))

def const_meancoef(t):
    return torch.exp(-0.5 * (int_const_beta_fn(t)))

def linear_var(t):
    return 1 - torch.exp(- int_linear_beta_fn(t))

def const_var(t):
    return 1 - torch.exp(- int_const_beta_fn(t))

def linear_meancoef_st(t, s):
    ibt = int_linear_beta_fn(t)
    ibs = int_linear_beta_fn(s)
    return torch.exp(-0.5 * (ibt - ibs))

def const_meancoef_st(t, s):
    ibt = int_const_beta_fn(t)
    ibs = int_const_beta_fn(s)
    return torch.exp(-0.5 * (ibt - ibs))

def linear_var_st(t, s):
    ibt = int_linear_beta_fn(t)
    ibs = int_linear_beta_fn(s)
    return 1 - torch.exp(-(ibt-ibs))

def const_var_st(t,s):
    ibt = int_const_beta_fn(t)
    ibs = int_const_beta_fn(s)
    return 1 - torch.exp(-(ibt-ibs))

'''
def linear_snr(t):
    return (linear_meancoef(t) ** 2) / linear_var(t)

def const_snr(t):
    return (const_meancoef(t) ** 2) / const_var(t)

def linear_log_snr(t):
    return linear_snr(t).log()

def const_log_snr(t):
    return const_snr(t).log()
'''

def linear_fancy(t):
    return linear_beta_fn(t) * linear_meancoef(t).pow(2) / linear_var(t)

def const_fancy(t):
    return const_beta_fn(t) * const_meancoef(t).pow(2) / const_var(t)

def linear_fancy_st(t):
    s = t - (delta/2)
    return linear_beta_fn(t) * linear_meancoef_st(t, s).pow(2) / linear_var_st(t, s)

def const_fancy_st(t):
    s = t - (delta/2)
    return const_beta_fn(t) * const_meancoef_st(t, s).pow(2) / const_var_st(t, s)

ts = torch.linspace(delta, 1 - 1e-5, 100)
linear_fancys = []
const_fancys = []
linear_fancys_st = []
const_fancys_st = []

for tscalar in ts:
    t = tscalar 
    one = linear_fancy(t)
    two = const_fancy(t)
    three = linear_fancy_st(t)
    four = const_fancy_st(t)
    linear_fancys.append(one.item())
    const_fancys.append(two.item())
    linear_fancys_st.append(three.item())
    const_fancys_st.append(four.item())

ts_numpy =  ts.numpy()

plt.title("g^2(t) * m^2_{t} / sigma^4(t)")
plt.plot(ts_numpy, linear_fancys, label='linear beta t|0')
plt.plot(ts_numpy, const_fancys, label='const beta t|0')
plt.plot(ts_numpy, linear_fancys_st, label='linear beta t|s')
plt.plot(ts_numpy, const_fancys_st, label='const beta t|s')

plt.legend()
plt.show()





























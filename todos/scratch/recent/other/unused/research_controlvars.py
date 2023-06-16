import time                                                                                                                                                                       
import pickle as pkl
from typing import Optional, List
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm.auto import tqdm
import math
import time 
from sklearn import linear_model as lm
from sklearn.linear_model import LinearRegression
from utils_numerical import stack
Normal = torch.distributions.Normal
Uniform = torch.distributions.Uniform

def to_np(x):
    return x.detach().cpu().numpy()

def dimwise_cov(X, Y):
    bsz = X.shape[0]
    assert X.shape == Y.shape
    return torch.einsum('bd,bd->d', X - X.mean(dim=0), Y - Y.mean(dim=0)) / (bsz - 1)

def dimwise_cov_centered(X, Y):
    bsz = X.shape[0]
    assert X.shape == Y.shape
    return torch.einsum('bd,bd->d', X, Y) / bsz

def simple_dimwise_cv(xt, mu_xt, f_per_dim):

    f = f_per_dim
    f_mean = f.mean(0)
    f_diff = f - f_mean
    f_sum = f.sum(0)
    bsz, dim = xt.shape
    g = xt
    g_mean = mu_xt
    g_var = g.var(dim=0)
    g_diff = g - g_mean
    g_sum = g.sum(0)

    assert mu_xt.shape == xt.shape
    assert f.shape == xt.shape 
    assert f_mean.shape == (dim,)
    assert f_diff.shape == xt.shape
    assert f_sum.shape == (dim,)
    assert g.shape == xt.shape
    assert g_mean.shape == xt.shape
    assert g_var.shape == (dim,)
    assert g_diff.shape == xt.shape
    assert g_sum.shape == (dim,)
    
    '''
    f_mean_loo = torch.zeros_like(xt)
    g_mean_loo = torch.zeros_like(xt)

    for i in range(bsz):
        
        f_mean_loo[i, :] = (f_sum - f[i,:]) / (bsz - 1)
        g_mean_loo[i,:] = (g_sum - g[i,:]) / (bsz - 1) 

    prod_diff_loo = (f - f_mean_loo) * (g - g_mean_loo) 
    assert prod_diff_loo.shape == xt.shape
    
    #fg = stack([f_diff, g_diff], dim=-1)
    #cov_fg = torch.cov(fg.permute(1,0))
    #assert cov_fg.shape == (dim+1, dim+1)
    #cov_fg = cov_fg[0,1:]
    
    cov_fg = dimwise_cov_centered(f_diff, g_diff)
    cov_numer = (cov_fg * bsz)[None,...] - prod_diff_loo
    cov_fg_loo = cov_numer / (bsz - 1)
    var_numer = (g_var * bsz)[None, ... ] - g_diff_loo
    g_var_loo = var_numer / (bsz - 1)
    alpha_loo = cov_fg_loo / g_var_loo
    '''

    alpha_loo = 1.0
    control = f - (alpha_loo * g_diff)
    control = control.sum(-1)
    
    #assert cov_fg.shape == (dim,) #covs of f with each g dim
    #assert cov_numer.shape == (bsz, dim)
    #assert var_numer.shape == (bsz, dim)
    #assert cov_fg_loo.shape == (bsz,dim)
    #assert g_var_loo.shape == (bsz, dim)
    #assert control.shape == (bsz, dim)
    assert control.shape == (bsz,)
    
    return control

def _regress_no_mean(xt, t, target):
    # TODO use t
    X = to_np(xt)
    y = to_np(target)
    reg = lm.Lasso(alpha=1.0)
    #reg = lm.LinearRegression()
    reg.fit(X, y)
    preds = reg.predict(X)
    return torch.tensor(preds).type_as(xt)

def _regress(u_t, u_t_mean, t, target):
    # TODO use t  
    X = to_np(u_t) #(bsz, dim)
    X_mean = to_np(u_t_mean) #(bsz, dim)
    y = to_np(target) #(bsz,)
    #reg = lm.Lasso(alpha=2.0) # hparam, which regressor
    reg = lm.LinearRegression()
    reg.fit(X, y)
    preds = reg.predict(X)
    preds_mean = reg.predict(X_mean)
    return torch.tensor(preds).to(u_t), torch.tensor(preds_mean).to(u_t)

def r_squared(y, preds):
    squared_residuals = (y - preds).pow(2)
    squared_dev = (y - y.mean()).pow(2)        
    r_sq = squared_residuals.sum() / squared_dev.sum()
    return r_sq    
       
def regression_cv(ut, ut_mean, t, f):
    bsz = ut.shape[0]
    g, g_mean = _regress(ut, ut_mean, t, f)                  

    f_mean, f_var = f.mean(), f.var().item()
    var_g = g.var().item()
    cov_fg = ((f - f_mean) * (g - g_mean)).mean()    

    #fg = torch.stack([f,g],dim=-1)
    #assert fg.shape==(bsz, 2)
    #cov_fg = torch.cov(fg.permute(1,0))
    #assert cov_fg.shape==(2,2)
    #var_g = g.var().item()
    #cov_fg = cov_fg[0,1].item()
    
    alpha =  cov_fg / var_g # TODO need to estimate alpha either with sep. samples or round robin
    #alpha = 1.0
    control = f - alpha * (g - g_mean)  
    return control

def regression_cv_no_mean(xt, t, f):
    bsz = xt.shape[0]
    g = _regress_no_mean(xt, t, f)
    assert f.shape == (bsz,)
    assert g.shape == (bsz,)
    g_mean = g.mean(0)
    g_diff = g - g_mean
    fg = torch.stack([f,g],dim=-1)
    assert fg.shape==(bsz, 2)
    cov_fg = torch.cov(fg.permute(1,0)) # because np/torch cov expect Xtranspose
    assert cov_fg.shape==(2,2)
    cov_fg = cov_fg[0,1].item()
    var_f = f.var().item() # just for debug
    var_g = g.var().item()
    alpha =  cov_fg / var_g # TODO need to estimate alpha either with sep. samples or round robin
    control = f - alpha * (g - g_mean)
    print("f mean", f.mean())
    print("g mean", g.mean())
    print("cov fg (all samples)", cov_fg)
    print("var f", f.var())
    print("var g", g.var())
    print("alpha (using all samples)", alpha)
    print("E[f - alpha(g-E[g])]", control.mean())
    print("var[f - alpha(g-E[g])]", control.var())
    assert control.shape == (bsz,)
    return control



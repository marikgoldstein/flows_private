import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from torchdiffeq import odeint_adjoint as odeint
from functorch import jacfwd, vmap
import math

# changes from michael's repo to get simple version running:
# only supporting one_sided b and s losses for now
# only brownian gamma
# not using antithetic for now

def image_sum(x):
    return x.sum(-1).sum(-1).sum(-1)

def image_sq_norm(x):
    return x.pow(2).sum(-1).sum(-1).sum(-1)

def loss_per_sample_sv(v, s, x0, x1, t, I, y, loss_fac):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = I.calc_antithetic_xts(t, x0, x1)
    dtIt        = I.dtIt(t, x0, x1)
    #dt_gam      = I.gamma_dot(t)
    vtp         = v(xtp, t, y)
    vtm         = v(xtm, t, y)
    loss_v      = 0.5*torch.sum(vtp**2) - torch.sum((dtIt) * vtp)
    loss_v      += 0.5*torch.sum(vtm**2) - torch.sum((dtIt) * vtm)
    stp         = s(xtp, t, y)
    stm         = s(xtm, t, y)
    gam = I.gamma(t)[:,None,None,None]
    loss_s = 0.5 * image_sq_norm(stp) + (1/gamma) * image_sum(stp * z)
    #loss_s      = 0.5*torch.sum(stp**2) + (1 / gam)*torch.sum(stp*z)
    loss_s += 0.5 * image_sq_norm(stm) - (1/gamma) * image_sum(stm * z)
    #loss_s      += 0.5*torch.sum(stm**2) - (1 / gam)*torch.sum(stm*z)
    # TODO rest of fixes    
    return (loss_v, loss_fac * loss_s)

def loss_per_sample_etav(v, eta, x0, x1, t, I, y, loss_fac):
    """Compute the loss on an individual sample. Denoising loss does not need antithetic trick."""
    # xtp, xtm, z = I.calc_antithetic_xts(t, x0, x1)
    xt, z = I.calc_xt(t, x0, x1)
    dtIt        = I.dtIt(t, x0, x1)
    #dt_gam      = I.gamma_dot(t)
    vt          = v(xt, t, y)
    loss_v      = 0.5*torch.sum(vt**2) - torch.sum((dtIt) * vt)
    etat         = eta(xt, t, y)
    loss_eta    = 0.5*torch.sum(etat**2) + torch.sum(etat*z)
    return (loss_v, loss_fac * loss_eta)
    
####### new stuff


def loss_per_sample_s(s, x0, x1, t, I, y):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = I.calc_antithetic_xts(t, x0, x1)
    stp         = s(xtp, t, y)
    stm         = s(xtm, t, y)
    gam = I.gamma(t)[:,None,None,None]
    #loss      = 0.5*torch.sum(stp**2) + (1 / gam)*torch.sum(stp*z)
    #loss     += 0.5*torch.sum(stm**2) - (1 / gam)*torch.sum(stm*z)    
    loss = 0.5 * image_sq_norm(stp) + image_sum((1/gam) * stp * z)
    loss += 0.5 * image_sq_norm(stm) - image_sum((1/gam) * stm * z)
    return loss

def loss_per_sample_eta(eta, x0, x1, t, I, y):
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z   = I.calc_xt(t, x0, x1)
    eta_val = eta(xt, t, y)
    return 0.5*torch.sum(eta_val**2) + torch.sum(eta_val*z) 
    
def loss_per_sample_v(v, x0, x1, t, I, y):
    """Compute the loss on an individual sample via antithetic sampling."""
    xt, z = I.calc_xt(t, x0, x1)
    dtIt  = I.dtIt(t, x0, x1)
    v_val = v(xt, t, y)
    return 0.5*torch.sum(v_val**2) - torch.sum(dtIt * v_val)

def loss_per_sample_b(b, x0, x1, t, I, y):
    """Compute the (variance-reduced) loss on an individual sample via antithetic sampling."""
    xtp, xtm, z = I.calc_antithetic_xts(t, x0, x1)
    dtIt        = I.dtIt(t, x0, x1)
    gamma_dot   = I.gamma_dot(t)[:,None,None,None]
    btp         = b(xtp, t, y)
    btm         = b(xtm, t, y)
    #loss        = 0.5*torch.sum(btp**2) - torch.sum((dtIt + gamma_dot*z) * btp)
    #loss       += 0.5*torch.sum(btm**2) - torch.sum((dtIt - gamma_dot*z) * btm)
    loss = 0.5 * image_sq_norm(btp) - image_sum((dtIt + gamma_dot*z) * btp)
    loss += 0.5 * image_sq_norm(btm) - image_sum((dtIt - gamma_dot*z) * btm)
    return loss

def loss_per_sample_one_sided(b, x0, x1, t, I, y):
    """Compute the loss on an individual sample."""
    xt  = I.calc_xt(t, x0, x1)
    dtIt        = I.dtIt(t, x0, x1)
    # gamma_dot   = I.gamma_dot(t)
    bt          = b(xt, t, y)
    #loss        = 0.5*torch.sum(bt**2) - torch.sum((dtIt) * bt)
    loss = 0.5 * image_sum((bt - dtIt).pow(2))
    return loss

def loss_per_sample_one_sided_s(s, x0, x1, t, I, y):
    """Compute the loss on an individual sample via antithetic samples for x_t = (1-t)z + t x1 where z=x0.
       Currently hardcoded for the linear one-sided interpolant.
    """
    #xtp, xtm, z = interpolant.calc_antithetic_xts(t, x0, x1)
    #stp         = s(xtp, t, y)
    #stm         = s(xtm, t, y)
    #alpha = torch.sqrt((1-t))
    #loss      = 0.5*torch.sum(stp**2) + (1 / (alpha))*torch.sum(stp*x0)
    #loss     += 0.5*torch.sum(stm**2) - (1 / (alpha))*torch.sum(stm*x0)
    xt = I.calc_xt(t, x0, x1)
    score = s(xt ,t, y)
    alpha = torch.sqrt((1 - t))[:,None,None,None]
    target = - (1/alpha) * x0
    loss = 0.5 * image_sum((score - target).pow(2))
    #loss = 0.5 * score.pow(2).sum(-1).sum(-1).sum(-1) - (target * score).sum(-1).sum(-1).sum(-1)
    return loss

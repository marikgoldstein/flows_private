import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Any, Tuple
from torchdiffeq import odeint_adjoint as odeint
from functorch import jacfwd, vmap
import math

# changes from michael's repo to get simple version running:
# - only one_sided b and one_sided s
# - only brownian gamma
# - no antithetic

class SFromEta(torch.nn.Module):
    """Class for turning a noise model into a score model."""
    def __init__(self, eta, gamma):
        super(SFromEta, self).__init__()
        self.eta = eta
        self.gamma = gamma
        
    def forward(self, x, t):
        val = (self.eta(x,t) / self.gamma(t))
        return val

    
class BFromVS(torch.nn.Module):
    """Class for turning a velocity model and a score model into a drift model."""
    def __init__(self, v, s, gg_dot):
        super(BFromVS, self).__init__()
        self.v = v
        self.s = s
        self.gg_dot = gg_dot
        
    def forward(self, x, t):
        return self.v(x, t) - self.gg_dot(t)*self.s(x, t)
    

class Interpolant(torch.nn.Module):
    """
    Class for all things interpoalnt $x_t = I_t(x_0, x_1) + \gamma(t)z.
    
    path: str,    what type of interpolant to use, e.g. 'linear' for linear interpolant. see fabrics for options
    gamma_type:   what type of gamma function to use, e.g. 'brownian' for $\gamma(t) = \sqrt{t(1-t)}
    """
    def __init__(self, path, gamma_type):
        super(Interpolant, self).__init__()
   
        assert path == 'one_sided', 'temporarily only supporting one_sided path'
        assert gamma_type == 'brownian', 'temporarily only supporting brownian gamma'

        self.gamma, self.gamma_dot, self.gg_dot = make_gamma(gamma_type=gamma_type)
        self.path = path
        self.It, self.dtIt = make_It(path, self.gamma, self.gamma_dot)

    def calc_xt(self, t, x0, x1):
        if self.path=='one_sided' or self.path == 'mirror' or self.path=='one_sided_bs':
            return self.It(t, x0, x1)
        else:
            z = torch.randn(x0.shape).to(t)
            return self.It(t, x0, x1) + self.gamma(t)[:,None,None,None]*z, z

    '''
    # TODO add antithetic and add one_sided_bs support
    def calc_antithetic_xts(self, t, x0, x1):
        if self.path=='one_sided_bs':
            #print("x1 x0 t", x1.shape, x0.shape, t.shape)
            _t = t[:, None, None, None]
            It_p = _t*x1 + torch.sqrt(1-_t)*x0 
            It_m = _t*x1 - torch.sqrt(1-_t)*x0
            return It_p, It_m, x0
        else:
            z   = torch.randn(x0.shape).to(t)
            gam = self.gamma(t)
            It  = self.It(t, x0, x1)
            return It + gam[:,None,None,None]*z, It - gam[:,None,None,None]*z, z
    '''

    def forward(self, x):
        raise NotImplementedError("No forward pass for interpolant.")


def make_gamma(gamma_type = 'brownian', a = None):
    
    assert gamma_type == 'brownian', "temporarily only supporting browninan gamma"
    
    """
    returns callable functions for gamma, gamma_dot,
    and gamma(t)*gamma_dot(t) to avoid numerical divide by 0s,
    e.g. if one is using the brownian (default) gamma.
    """
    gamma = lambda t: torch.sqrt(t*(1-t))
    gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
    gg_dot = lambda t: (1/2)*(1-2*t)
        
    '''
    if gamma_type == 'brownian':
        gamma = lambda t: torch.sqrt(t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        gg_dot = lambda t: (1/2)*(1-2*t)
        
    if gamma_type == 'a-brownian':
        gamma = lambda t: torch.sqrt(a*t*(1-t))
        gamma_dot = lambda t: (1/(2*torch.sqrt(a*t*(1-t)))) * a*(1 -2*t)
        gg_dot = lambda t: (a/2)*(1-2*t)
        
    elif gamma_type == 'zero':
        gamma = gamma_dot = gg_dot = lambda t: torch.zeros_like(t)

    elif gamma_type == 'bsquared':
        gamma = lambda t: t*(1-t)
        gamma_dot = lambda t: 1 -2*t
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sinesquared':
        gamma = lambda t: torch.sin(math.pi * t)**2
        gamma_dot = lambda t: 2*math.pi*torch.sin(math.pi * t)*torch.cos(math.pi*t)
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    elif gamma_type == 'sigmoid':
        f = torch.tensor(10.0)
        gamma = lambda t: torch.sigmoid(f*(t-(1/2)) + 1) - torch.sigmoid(f*(t-(1/2)) - 1) - torch.sigmoid((-f/2) + 1) + torch.sigmoid((-f/2) - 1)
        gamma_dot = lambda t: (-f)*( 1 - torch.sigmoid(-1 + f*(t - (1/2))) )*torch.sigmoid(-1 + f*(t - (1/2)))  + f*(1 - torch.sigmoid(1 + f*(t - (1/2)))  )*torch.sigmoid(1 + f*(t - (1/2)))
        gg_dot = lambda t: gamma(t)*gamma_dot(t)
        
    else:
        raise NotImplementedError("The gamma you specified is not implemented.")
    ''' 
    return gamma, gamma_dot, gg_dot

def make_It(path='linear', gamma = None, gamma_dot = None):
    """gamma function must be specified if using the trigonometric interpolant"""

    assert path == 'one_sided', 'temporarily only supporiting one_sided path'

    It   = lambda t, x0, x1: (1 - t[:,None,None,None])*x0 + t[:,None,None,None]*x1
    dtIt = lambda _, x0, x1: x1 - x0

    '''


    if path == 'linear':
        It   = lambda t, x0, x1: (1 - t)*x0 + t*x1
        dtIt = lambda _, x0, x1: x1 - x0
        
    elif path == 'trig':
        if gamma == None:
            raise TypeError("Gamma function must be provided for trigonometric interpolant!")
        a    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)
        b    = lambda t: torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        adot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t) \
                                - 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t)
        bdot = lambda t: -self.gg_dot(t)/torch.sqrt(1 - gamma(t)**2)*torch.sin(0.5*math.pi*t) \
                                + 0.5*math.pi*torch.sqrt(1 - gamma(t)**2)*torch.cos(0.5*math.pi*t)

        It   = lambda t, x0, x1: self.a(t)*x0 + self.b(t)*x1
        dtIt = lambda t, x0, x1: self.adot(t)*x0 + self.bdot(t)*x1
        
    elif path == 'encoding-decoding':
        def I_fn(t, x0, x1):
                if t <= torch.tensor(1/2):
                    return (torch.cos(  math.pi * t)**2)*x0
                elif t >= torch.tensor(1/2):
                    return (torch.cos(  math.pi * t)**2)*x1

        It  = I_fn

        def dtI_fn(t,x0,x1):
            if t < torch.tensor(1/2):
                return -(1/2)* torch.sin(  math.pi * t) * torch.cos(  math.pi * t)*x0
            else:
                return -(1/2)* torch.sin(  math.pi * t) * torch.cos( math.pi * t)*x1

        dtIt = dtI_fn

    elif path == 'one_sided':
        
        ### hack to override the use of z in the interpolant. If one side is a gaussian, z is redundant
        It   = lambda t, x0, x1: (1 - t[:,None,None,None])*x0 + t[:,None,None,None]*x1
        dtIt = lambda _, x0, x1: x1 - x0
    
    elif path == 'one_sided_bs':
        
        def It(t, x0, x1):
            return torch.sqrt((1 - t[:,None,None,None]))*x0 + t[:,None,None,None]*x1

        def dtIt(t, x0, x1):
            coef = .5 * (1/((1-t)**(1/2)))
            _coef = coef[:, None, None, None]
            return x1 - _coef * x0

    else:
        raise NotImplementedError("The interpolant you specified is not implemented.")
    '''

    return It, dtIt


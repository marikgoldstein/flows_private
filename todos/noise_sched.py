import torch
import torch.nn as nn
import numpy as np
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, 
    sqnorm
)
ones_like = torch.ones_like
atan = torch.atan
tan = torch.tan

# only export this function
def get_noise_schedule(config):

    if config.schedule == 'simple':
        schedule_fn = SimpleSchedule
    elif config.schedule == 'linear':
        schedule_fn = LinearSchedule
    elif config.schedule == 'cosine':
        schedule_fn = CosineSchedule
    elif config.schedule == 'cosine2':
        schedule_fn = CosineSchedule2
    elif config.schedule == 'const':
        schedule_fn = ConstSchedule
    else:
        assert False

    return schedule_fn(config)

class ConstSchedule:

    def __init__(self, config):
        self.config = config
        self.const = 10.0

    def beta_fn(self, t):
        return torch.ones_like(t) * self.const

    def int_beta_fn(self, t):
        return self.const * t

    def get_mean_coef_squared(self, t):
        return torch.exp(-self.int_beta_fn(t))

class LinearSchedule:

    def __init__(self, config):

        self.config = config
        self.b0 = .1
        self.b1 = 20.0
        self.bdiff = self.b1 - self.b0
    
    def beta_fn(self, t):
        return self.b0 + self.bdiff * t

    def int_beta_fn(self, t):
        return self.b0 * t + self.bdiff * (t.pow(2) / 2)

    def get_mean_coef_squared(self, t):
        return torch.exp(-self.int_beta_fn(t))

class SimpleSchedule:

    def __init__(self, config):
        self.config = config

    def get_mean_coef_squared(self, t):
        return 1 - t

    def beta_fn(self, t):
        return 1. / (1. - t)

    def int_beta_fn(self, t):
        assert False # not needed

class CosineSchedule2:

    def __init__(self, config):
        self.config = config
        #sigma_t = torch.sqrt(torch.sigmoid(-self.logsnr(t)))

    def get_mean_coef_squared(self, t):
        return torch.sigmoid(self.logsnr(t))

    def int_beta_fn(self, t):
        assert False # not needed

    def logsnr(self, t , logsnr_min = -15 , logsnr_max =15):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return -2 * torch.log(tan( _t))

    def beta_fn(self, t):
        t_min, t_max = self.tmin_tmax(t)
        width = t_max - t_min
        _t = t_min + t * width
        return 2 * (1 - self.logsnr(t).sigmoid()) * (1 / tan(_t)) * (1 / _t.cos().pow(2)) * width 
        
    def tmin_tmax(self, dummy, logsnr_min = -15, logsnr_max = 15):
        t1 = torch.tensor([-0.5 * logsnr_max]).type_as(dummy)
        t0 = torch.tensor([-0.5 * logsnr_min]).type_as(dummy)
        t_min = self.atanexp(t1)
        t_max = self.atanexp(t0)
        return t_min, t_max
 

    def atanexp(self, t):
        return atan(torch.exp(t))
    

class CosineSchedule:
    def __init__(self, config):
        self.config = config

    def cos_term(self, t):
        s = 0.008 # some offset for numerical safety
        numer = t+s
        denom = 1+s
        coef = np.pi / 2
        inside = (numer/denom) * coef
        return inside.cos().pow(2)

    # exp[-int beta] = mean coef squared implies that beta = -(d/dt) log (mean coef squared)
    def beta_fn(self, t):
        s = 0.008
        numer = t+s
        denom = 1+s
        coef = np.pi / 2
        inside = (numer/denom) * coef
        return inside.tan() * (np.pi / (1+s))

    def int_beta_fn(self, t):
        s = 0.008
        const = (np.pi / (1+s))
        a = (np.pi / 2) / (1+s)
        b = s
        return const * (-1 / a) * (a*(t+b)).cos().log()

    # alpha bar in openai paper, which is mean coef squared
    def get_mean_coef_squared(self, t):
        return self.cos_term(t)/self.cos_term(torch.zeros_like(t))


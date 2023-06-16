import wandb
import torch
import torch.nn as nn
import numpy as np
import math
from dit import get_dit
from encoder import Encoder
from config import ExperimentConfig
from integrators import PFlowIntegrator, SDEIntegrator
from boilerplate import make_model_and_ema, update_ema, maybe_trim_grads, get_params
from image_processing import ImageProcessing


# (bsz,) to (bsz, 1 ,1 ,1)
def wide(x):
    return x[:, None, None, None]

def triple_sum(x):
    return x.sum(-1).sum(-1).sum(-1)

def sqnorm_image(x):
    return triple_sum(x.pow(2))

class Interpolant(torch.nn.Module):

    def __init__(self, config, device):
        super(Interpolant, self).__init__()   
        self.gamma = lambda t: torch.sqrt(t*(1-t))
        self.gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        self.gg_dot = lambda t: (1/2)*(1-2*t)
        self.It   = lambda t, x0, x1: (1 - wide(t))*x0 + wide(t)*x1
        self.dtIt = lambda _, x0, x1: x1 - x0
        
 
    def forward(self, x):
        raise NotImplementedError("No forward pass for interpolant.")

    # maybe send this to each separate one eg Flow, Interpolant, Diffusion, etc
    def setup_models(self, device, rank = None):
        conf = self.config
        models = {}
        bv, bv_ema = make_model_and_ema(conf, device, rank)
        models['bv'] = bv
        models['bv_ema'] = bv_ema
        if conf.do_seta:
            seta, seta_ema = make_model_and_ema(conf, device, rank)
            models['seta'] = seta
            models['seta_ema'] = seta_ema
        return models
      
    def loss_b(self, b, x0, x1, t, y):
        xt  = self.It(t, x0, x1)
        return 0.5 * sqnorm_image(b(xt, t, y) - self.dtIt(t, x0, x1) )  

    def loss_s(self, s, x0, x1, t, y):
        xt  = self.It(t, x0, x1)
        alpha = wide(torch.sqrt(1 - t))
        target = - (1/alpha) * x0
        return 0.5 * sqnorm_image(s(xt ,t, y) - target)       

    def loss_fn(self, z, y, models, device):
        
        bv = models['bv']
        z1, z0 = z, torch.randn_like(z)
        conf = self.config
        bsz = z1.shape[0]
        t = torch.rand(bsz,).to(device)
        t = torch.clamp(t, min=self.config.T_min, max = self.config.T_max)
        loss_b, loss_s = torch.tensor([0.0]), torch.tensor([0.0])              
        loss_b = self.loss_b(bv, z0, z1, t, y)
        assert loss_b.shape == (bsz, )                                                             
        if conf.do_seta:
            seta = models['seta']
            loss_s = self.loss_s(seta, z0, z1, t, y)
            assert loss_s.shape == (bsz,)        

        total_loss = loss_b + loss_s
        return {
            'loss' : total_loss.mean(),
            'bv_loss': loss_b.mean(),
            'seta_loss': loss_s.mean(),
        }

    @torch.no_grad()                  
    def sample(self, z0, y, models, use_ema = False, cheap = False):
        model, model_ema = models['model'], models['model_ema']
        field = model_ema if use_ema else model
        return self.pflow(field, z0, y, cheap)
        
        save_fn = image_processing.save_images_for_fid
        wandb_fn = image_processing.process_images_for_wandb
        D = {}   
        ema_str = '_ema' if use_ema else '_nonema'
        print(f"sampling from model. Use ema: {use_ema}")
        z0, y = sample_prior(config.num_sampled_images, device)
         
        bv, bv_ema = self.models['bv'], self.models['bv_ema']
        zT = self.sample_ode(bv, bv_ema, use_ema, z0, y, cheap)
        x = dec_fn(zT)
        _dir = dir_fn(conf.sample_dir, 'interpolant_ode', use_ema, steps)
        save_fn(x.clone(), _dir, use_ema)
        D['interpolant_ode_samples' + ema_str] = wandb_fn(x)
                 
        bv, bv_ema = self.models['bv'], self.models['bv_ema']
        seta, seta_ema = self.models['seta'], self.models['seta_ema']
        zT = self.sample_sde(bv, bv_ema, seta, seta_ema, z0, y, cheap)
        x = dec_fn(zT)
        _dir = dir_fn(conf.sample_dir, 'interpolant_sde', use_ema, steps)
        save_fn(x.clone(), _dir, use_ema)
        D['interpolant_sde_samples' + ema_str] = wandb_fn(x)


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



def calc_xt(self, t, x0, x1):
    return self.It(t, x0, x1)
    #if self.path=='one_sided' or self.path == 'mirror' or self.path=='one_sided_bs':
    #    return self.It(t, x0, x1)
    #else:
    #    z = torch.randn(x0.shape).to(t)
    #    return self.It(t, x0, x1) + wide(self.gamma(t))*z, z


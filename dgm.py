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

AdamW = torch.optim.AdamW

# (bsz,) to (bsz, 1 ,1 ,1)
def wide(x):
    return x[:, None, None, None]

def triple_sum(x):
    return x.sum(-1).sum(-1).sum(-1)

def sqnorm_image(x):
    return triple_sum(x.pow(2))

class ContinuousTimeDGM(nn.Module):

    def __init__(self,):
        super(ContinuousTimeDGM, self).__init__()

    def setup(self, config, device, rank):
        # call this in subclass init
        self.config = config
        self.device = device
        self.rank = rank
        self.setup_models(device, rank)
        self.opt = AdamW(get_params(self.models), lr=config.base_lr, weight_decay=config.wd)
        self.encoder = Encoder(self.config, device)
        self.image_processing = ImageProcessing(self.config)
        self.wandb_fn = self.image_processing.process_images_for_wandb
        self.save_fn = self.image_processing.save_images_for_fid           
        self.pflow = PFlowIntegrator(self.config)
        self.sflow = SDEIntegrator(config, device)

    def get_ckpt_dict(self,):
        checkpoint = {k: self.models[k].state_dict() for k in self.models}
        checkpoint['opt'] = self.opt.state_dict()
        checkpoint['config'] = self.config
        return checkpoint

    def step_optimizer(self, rank, steps, log_norm = False):    

        if steps % self.config.grad_accum != 0:
            return False
        # trim each backward or just this one? 
        maybe_trim_grads(
            models = self.models,
            config = self.config,
            rank = rank, 
            steps = steps,
            log_norm = log_norm
        )
        self.opt.step()          
        self.opt.zero_grad()     
        return True

    def setup_models(self, device, rank = None):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def loss_fn(self, z, y, device):
        raise NotImplementedError

    @torch.no_grad()
    def _sample(z0, y, use_ema, cheap):
        raise NotImplementedError

    def sample_prior(self, N, device):
        conf = self.config
        C, H, W = conf.C_dgm, conf.H_dgm, conf.W_dgm
        y = torch.randint(0, conf.num_classes, (N,)).to(device)
        z0 = torch.randn(N, C, H, W).to(device)
        return z0, y

    @torch.no_grad()                  
    def sample(self, config, directory, device, use_ema = False, cheap = False):
        z0, y = self.sample_prior(config.num_sampled_images, device)
        zT = self._sample(z0, y, use_ema, cheap)
        x = self.encoder.decode(zT)
        self.save_fn(x.clone(), directory)
        return self.wandb_fn(x)

class Flow(ContinuousTimeDGM):

    def __init__(self, config, device, rank, **kwargs):
        super(Flow, self).__init__()
        self.setup(config, device, rank) # see superclass
        
    def forward(self, x):
        raise NotImplementedError("No forward pass for flow")

    # maybe send this to each separate one eg Flow, Interpolant, Diffusion, etc
    def setup_models(self, device, rank = None):
        model, model_ema = make_model_and_ema(self.config, device, rank)
        self.models = {'model': model, 'model_ema': model_ema}

    def loss_fn(self, z1, y, device):
        model = self.models['model']
        bsz = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(bsz,).to(device)
        t = t.clamp(min=self.config.T_min, max = self.config.T_max)
        zt = wide(t) * z1 + wide(1-t) * z0
        vtheta = model(zt, t, y)
        loss = sqnorm_image(vtheta - (z1 - z0))
        return {'loss': loss.mean()}

    @torch.no_grad()                  
    def _sample(self, z0, y, use_ema = False, cheap = False):
        model, model_ema = self.models['model'], self.models['model_ema']
        field = model_ema if use_ema else model
        field.eval()
        x = self.pflow(field, z0, y, cheap)
        if not use_ema:
            field.train()
        return x


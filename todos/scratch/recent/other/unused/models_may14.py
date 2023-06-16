import os, sys
import pickle
import torch
import torch.nn as nn
import sys
from functorch import vmap
import numpy as np
import math
from math import pi
import time
import torch.distributions as D
from ema_pytorch import EMA

# local
from utils_numerical import (
    randn, cat, ones, ones_like, eye, linspace
)
from models_openai import get_openai_mnist_unet
from models_bigunet import get_big_unet
from models_dhariwal import get_dhariwal_unet
from models_tiny import get_tiny_net
from models_lucid import get_lucid_unet
from models_lucid2 import get_lucid2_unet
from models_scoreunet import get_scoreunet
from models_dit import get_dit
import copy
Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                        

def get_unet_fn(arch):

    if arch == 'tiny':
        return get_tiny_net
    elif arch == 'scoreunet':
        return get_scoreunet
    elif arch == 'lucid':
        return get_lucid_unet
    elif arch == 'lucid2':
        return get_lucid2_unet
    elif arch == 'openai':
        return get_openai_mnist_unet
    elif arch == 'bigunet': 
        return get_big_unet
    elif arch == 'dhariwal':
        return get_dhariwal_unet
    elif arch == 'dit':
        return get_dit
    else:
        assert False

class EMAObj(nn.Module):

    def __init__(self, trainer, model_obj):
        super().__init__()
        self.model_obj = model_obj
        self.config = trainer.config
        device = self.config.device
        # self.ema.ema_model will contain the actual model,
        # but better save whole thing
        self.model = EMA(
            self.model_obj.model,
            beta = self.config.ema_decay,
            update_after_step = self.config.num_warmup_steps,
            update_every = self.config.update_ema_model_every_n_steps,
        )

    def __call__(self, x, t):
        return self.model(x, t)

    def update(self,):
        self.model.update()

    def ema_model_to_eval_mode(self,):
        self.model.eval()

class ModelObj:
    
    def __init__(self, trainer):
        
        self.trainer = trainer
        self.config = trainer.config
        unet_fn = get_unet_fn(self.config.arch)
        self.model = unet_fn(self.config)
        self.model.to(self.config.device)
        self.train_mode()

        self.ema_obj = None
        if self.config.use_ema:
            self.ema_obj = EMAObj(trainer = self.trainer, model_obj = self)

        self.opt = AdamW(
            self.model.parameters(), 
            lr=self.config.original_lr, 
            weight_decay=self.config.wd
        )
        self.total_params = self.count_params()
        print("total params", self.total_params)

        if self.config.lr_sched:
            sched_args = {
                'total_steps': self.config.total_steps,
                'num_warmup_steps': self.config.num_warmup_steps,
                'start_value': self.config.original_lr,
                'end_value': self.config.min_lr,
                'decay': 'cosine'
            }
            self.sched = Scheduler(**sched_args)

        print("testing model saving")
        self.dump()
        print("success")


    def count_params(self,):
        total = 0
        for n, p in self.model.named_parameters():
            total += p.numel()
        return total


    def __call__(self, u, t, with_ema_model):

        if with_ema_model:
            assert self.config.use_ema
            m = self.ema_obj
        else:
            m = self.model
        if self.config.arch == 'dit':
            return m(u, t, torch.zeros_like(t).long())
        else:
            return m(u, t)

    def ema_model_to_eval_mode(self,):
        return self.ema_obj.ema_model_to_eval_mode()

    def train_mode(self,):
        self.model.train()
    
    def eval_mode(self,):
        self.model.eval()

    def get_step(self,):
        return self.trainer.step

    def get_current_lr(self,):
        if self.config.lr_sched:
            return self.sched.get_value(self.get_step())
        else:
            return self.config.original_lr

    def update_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr

    def maybe_handle_nan_grads(self,):
        if not self.config.handle_nan_grads:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, 
                    nan=0, 
                    posinf=1e5, 
                    neginf=-1e5, 
                    out=param.grad
                )

    def maybe_clip_grads(self):
        gcn = self.config.grad_clip_norm
        if gcn is None:
            gcn = np.inf
        norm = clip_grad_norm_(self.model.parameters(), max_norm = gcn)
        return norm

    def compute_grads_no_step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.maybe_handle_nan_grads()
        self.maybe_clip_grads()
    
    def step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.maybe_handle_nan_grads()
        grad_norm = self.maybe_clip_grads()
        if self.config.lr_sched:
            current_lr = self.get_current_lr()
            self.update_lr(current_lr)
        self.opt.step()
        if self.config.use_ema:
            self.ema_obj.update()
        return grad_norm
    
    def dump(self,):
        step = self.get_step()
        data = {
            'optimizer_state': self.opt.state_dict(),
            'step': step,
            'ema_model_state': self.ema_obj.state_dict() if self.config.use_ema else None, 
            'model_state': self.model.state_dict() 
        }
        fname = os.path.join(self.config.save_ckpt_path, f'training_state_{step}.pkl')
        torch.save(data, fname)
        del data # conserve memory
        print(f"model checkpoint written in: {fname}")


    def restore(self, model_state, opt_state, ema_model_state):
        self.model.load_state_dict(model_state)
        self.model.to(self.config.device)

        if self.config.use_ema:
            self.ema_obj.load_state_dict(ema_model_state)
            self.ema_obj.to(self.config.device)

        print("models restored")
        self.opt.load_state_dict(opt_state)
        print("optimizer state dict loaded")

    def possibly_restore(self,):

        if self.config.resume_path is None:
            print("Starting from blank model")
        else:
            path = self.config.resume_path
            data = torch.load(path, map_location="cpu")
            self.trainer.step = data['step']
            opt_state = data['optimizer_state']
            model_state = data['model_state']
            ema_model_state = data['ema_model_state']
            self.restore(
                model_state = model_state, 
                opt_state = opt_state, 
                ema_model_state = ema_model_state
            )
            del data # conserve memory
            print("Current step is :", self.trainer.step)

    

def cosine_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 0.5 * (1 + math.cos(math.pi * x))
    return decay_value

def linear_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 1 - x
    return decay_value

def square_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = math.pow((x - 1), 2)
    return decay_value

def sqrt_decay(step, max_step):
    step = min(step, max_step)
    x = step / max_step
    decay_value = 1 - math.sqrt(1 - math.pow((x - 1), 2))
    return decay_value

class Scheduler:
    
    def __init__(self, total_steps, num_warmup_steps, start_value, end_value, decay):
       
        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.start_value = start_value
        self.end_value = end_value
        print("LR Scheduler initialized with:")
        print("Total steps", total_steps)
        print("Num warmup steps", num_warmup_steps)
        print("start value", start_value)
        print("end value", end_value)

        if decay == "cosine":
            self.decay = cosine_decay
        elif decay == "linear":
            self.decay = linear_decay
        elif decay == "square":
            self.decay = square_decay
        elif decay == "sqrt":
            self.decay = sqrt_decay

    def get_value(self, step):
        if step < self.num_warmup_steps:
            value = self.start_value * step / self.num_warmup_steps
        else:
            lr_range = (self.start_value - self.end_value)
            dec = self.decay(step - self.num_warmup_steps, self.total_steps - self.num_warmup_steps)
            value = self.end_value + lr_range * dec
        
        self.temp_value = value
        return value

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
import copy

Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                              

def get_unet_fn(arch):

    if arch == 'tiny':
        return get_tiny_net
    elif arch == 'openai':
        return get_openai_mnist_unet
    elif arch == 'bigunet':                                                                                                
        return get_big_unet
    elif arch == 'dhariwal':
        return get_dhariwal_unet
    else:
        assert False

class EMA:

    def __init__(self, trainer):

        self.trainer = trainer
        self.config = trainer.config
        self.num_warmup = self.conf.num_warmup_steps
        self.every_n_steps = self.config.update_ema_model_every_n_steps
        self.decay = self.config.ema_dea        
        device = self.trainer.device 
        self.n_averaged = 0
        #self.ema_model = TODO #ExponentialMovingAverage(self.model, device=device, decay=1.0 - alpha)
        #def copy_state(self, other_model_wrapper):
        #    self.model.load_state_dict(copy.deepcopy(other_model_wrapper.model.state_dict()))

    def model_params(self,):
        return self.trainer.model_obj.model.named_parameters()

    def ema_params(self,):
        return self.ema_model.named_parameters()

    def zipped_params(self,):
        return zip(self.model_params(), self.ema_params()

    def overwrite(self,):
        for (nmodel, pmodel), (nema, pema) in self.zipped_params()
            assert nmodel == nema
            pema.data = pmodel.data.clone()
        self.n_averaged = 0

    def mix(self,):
        for (nmodel, pmodel), (nema, pema) in self.zipped_params()
            assert nmodel == nema
            torch.mul_(pema, self.decay)
            torch.add_(pema, pmodel, alpha=(1.0 - self.decay))
        self.n_averaged += 1

    def update(self,):

        step = self.trainer.get_step()
        
        if step < self.num_warmup:
            self.overwite()
            return

        if step % self.every_n_steps == 0:
            self.mix()                               



class ModelObj:
    
    def __init__(self, trainer):
        
        self.trainer = trainer
        self.conf = trainer.config
        self.C = self.conf.C
        self.W = self.conf.W
        self.H = self.conf.H
        self.model_parameterization = self.conf.model_parameterization
        conf = self.conf
        unet_fn = get_unet_fn(conf.arch)
        self.model = unet_fn(conf.d)
        self.model.to(self.trainer.device)
        self.train_mode()

        self.ema_model = None
        if conf.use_ema:
            self.ema_model = EMA(trainer = self.trainer)

        self.wd = conf.wd
        self.grad_clip_norm = conf.grad_clip_norm
        self.opt = AdamW(self.model.parameters(), lr=conf.original_lr, weight_decay=conf.wd)
        self.bsz = conf.bsz
        self.total_params = self.count_params()
        print("total params", self.total_params)

        if conf.lr_sched:
            sched_args = {
                'total_steps': conf.total_steps,
                'num_warmup_steps': conf.num_warmup_steps,
                'start_value': conf.original_lr,
                'end_value': conf.min_lr,
                'decay': 'cosine'
            }
            self.sched = Scheduler(**sched_args)

    def count_params(self,):
        total = 0
        for n, p in self.model.named_parameters():
            total += p.numel()
        return total


    def __call__(self, u, t, with_ema_model):

        if with_ema_model:
            m = self.ema_model
        else:
            m = self.model
        return m(u, t)

    def train_mode(self,):
        self.model.train()
    
    def eval_mode(self,):
        self.model.eval()

    def get_step(self,):
        return self.trainer.step

    def get_current_lr(self,):
        if self.conf.lr_sched:
            return self.sched.get_value(self.get_step())
        else:
            return self.conf.original_lr

    def update_lr(self, lr):
        for g in self.opt.param_groups:
            g['lr'] = lr

    def maybe_handle_nan_grads(self,):
        if not self.conf.handle_nan_grads:
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
        gcn = self.conf.grad_clip_norm
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
        if self.conf.lr_sched:
            current_lr = self.get_current_lr()
            self.update_lr(current_lr)
        self.opt.step()
        self.custom_update_ema_model()
        return grad_norm
    
    def dump(self,):
        
        step = self.get_step()
        # these are models and not state_dict()'s
        data = {
            'optimizer_state': self.opt.state_dict(),
            'step': step,
            'ema_model_state': self.ema_model.state_dict() if self.conf.use_ema else None, 
            'model_state': self.model.state_dict() 
        }
        suffix = f'{self.conf.ckpt_name}_' + f'index_{self.conf.index}_training_state_{step}.pkl'
        fname = os.path.join(self.conf.ckpt_dir, suffix)
        torch.save(data, fname)
        del data # conserve memory
        print(f"model checkpoint written in: {fname}")


    def restore(self, model_state, opt_state, ema_model_state):
        # these are models and not state_dict's
        
        #self.copy_params_and_buffers(src=model, dst=self.model, require_all=True)
        self.model.load_state_dict(model_state)
        self.model.to(self.trainer.device)

        if self.conf.use_ema:

            #self.copy_params_and_buffers(src=ema_model, dst=self.ema_model, require_all=True)
            self.ema_model.load_state_dict(ema_model_state)
            self.ema_model.to(self.trainer.device)

        print("models restored")
        self.opt.load_state_dict(opt_state)
        print("optimizer state dict loaded")

    def possibly_restore(self,):

        if self.conf.resume_path is None:
            print("Starting from blank model")
        else:
            path = self.conf.resume_path
            #with open(path, 'rb') as f:
            #    data = pickle.load(f)
            
            data = torch.load(path, map_location="cpu")
            self.trainer.step = data['step']
            opt_state = data['optimizer_state']
            
            model_state = data['model_state']
            ema_model_state = data['ema_model_state']
            
            self.restore(model_state = model_state, opt_state = opt_state, ema_model_state = ema_model_state)
            del data # conserve memory
            print("Current step is :", self.trainer.step)

    
    @torch.no_grad()
    def copy_params_and_buffers(self, src, dst, require_all=False):
        assert isinstance(src, torch.nn.Module)
        assert isinstance(dst, torch.nn.Module)
        src_tensors = dict(named_params_and_buffers(src))
        for name, tensor in named_params_and_buffers(dst):
            assert (name in src_tensors) or (not require_all)
            if name in src_tensors:
                tensor.copy_(src_tensors[name])






class EMA(t



class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())



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

'''

# TODO CONSIDER USING MORE CONVENTIONAL SCHEDULER
# THIS IS JUST WARMUP, CONSIDER USING WARMUP + COSINE DECAY
def get_lr(self, cur_nimg):
    warmup_factor = min(cur_nimg / max(self.rampup_img, 1e-8), 1)
    return self.conf.original_lr * warmup_factor


def update_ema_model(self,):
    if self.ema_rampup_ratio is not None:
        ema_hl_nimg = min(self.ema_halflife_nimg, cur_nimg * self.ema_rampup_raito)
        power = (self.bsz / max(ema_hl_nimg, 1e-8))
        ema_beta = 0.5 ** power
        cur_params = self.model.parameters()
        for p_ema, p_net in zip(self.ema_model.parameters(), cur_params):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

'''

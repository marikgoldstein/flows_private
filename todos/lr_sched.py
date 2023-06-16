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
# outdated
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
       
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]
       
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
'''

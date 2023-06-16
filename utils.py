import torch
import torch.nn as nn
import sys
import numpy as np
from math import pi
import time
import logging
Uniform = torch.distributions.Uniform

def gradi(fval, x, i):  
    return torch.autograd.grad(fval[:,i].sum(), x, create_graph=True)[0][:,i]   
                  
def compute_div(f, x, t):   
    """Compute the divergence of f(x,t) with respect to x, assuming that x is batched."""   
    bs = x.shape[0] 
    with torch.set_grad_enabled(True):  
        x.requires_grad_(True)  
        t.requires_grad_(True)  
        f_val = f(x, t) 
        divergence = 0.0
        for i in range(x.shape[1]):     
            divergence += gradi(f_val, x, i)
                  
    return divergence.view(bs)


def is_type_for_logging(x):
    if isinstance(x, int):
        return True
    elif isinstance(x, float):
        return True
    elif isinstance(x, bool):
        return True
    elif isinstance(x, str):
        return True
    elif isinstance(x, list):
        return True
    elif isinstance(x, set):
        return True
    else:
        return False

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
                   
@torch.no_grad()   
def create_logger(logging_dir):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )          
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger  


class TimeSampler:            
                              
    def __init__(self, mint, maxt):                                 
                              
        self.mint = mint      
        self.maxt = maxt      
        self.t_width = self.maxt - self.mint                        
        self.U = Uniform(low=self.mint, high=self.maxt)
        print("tmin tmax width", self.mint, self.maxt, self.t_width)
                              
    def __call__(self, bsz, device):  
        t, w = self.sample_U(bsz)                                   
        return t.to(device), w.to(device)
                              
    def sample_U(self, bsz):                                        
        time = self.U.sample(sample_shape = (bsz,)).squeeze(-1)     
        weight = self.t_width * torch.ones(bsz,)
        return time, weight

#def test_distributed(self,):
    #print('dist initialized:', dist.is_initialized())
    #print("nccl avail", torch.distributed.is_nccl_available())
    #print('dist initialized:', dist.is_initialized())                                           
    #print("nccl avail", torch.distributed.is_nccl_available())
    #print("elastic launched", torch.distributed.is_torchelastic_launched())
#    pass
    

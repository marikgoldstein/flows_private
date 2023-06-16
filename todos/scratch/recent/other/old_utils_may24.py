import torch
import torch.nn as nn
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
from functorch import vmap
import numpy as np
from math import pi
import time
import torch.distributions as D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import wandb
from utils_numerical import (
    randn, cat, ones, ones_like, eye, linspace, stack
)
Uniform = torch.distributions.Uniform

#self.soboleng = torch.quasirandom.SobolEngine(dimension=784)
#def draw_quasirandom(bsz,):
#    return self.soboleng.draw
#x = soboleng.draw(1)
#x = x.squeeze(0)
#print(x.shape)

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


#def merge1d(h, high, low):
#    return h*high + (~h)*low

#def merge2d(h, high, low):
#    _h = h[:, None]
#    return _h*high + (~_h)*low

#def check(self, x,s):
#    if torch.any(torch.isnan(x)):
#        print(s,"is nan")
#        assert False
#    if torch.any(torch.isinf(x)):
#        print(s,"is inf")
#        assert False
#    print(s, "stats: min mean max", x.min(), x.mean(), x.max())


#def str_to_bool(s):                                                                                                            
#    if s=='false':
#        return False
#    else:
#        return True


#def log_results(D, stage, use_wandb, step, prefix=''): 
#
#    new_D = {}
#    for key in D:
#        if D[key] is not None:
#            new_D[key] = D[key]
#
#    if use_wandb:
#
#        if prefix == '':
#            prefix += f'{stage}_'
#        else:
#            prefix += f'_{stage}_'
#        wandb.log({prefix + k :new_D[k] for k in new_D},
#        step = step
#    )


#def merge_model_dicts(model1_dict, model2_dict):
#
#    D = {}
#    for key in model1_dict:
#        #D['model1_' + key] = model1_dict[key]
#        D[key] = model1_dict[key]
#    for key in model2_dict:
#        #D['model2_' + key] = model2_dict[key]
#        D[key] = model2_dict[key]
#    return D

#def param_no_grad(tensor):
#    return nn.Parameter(tensor, requires_grad=False)

#def isbad(x):
#    return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

#def check(msg, x):
#    if isbad(x):
#        print(msg, "is bad")
#        assert False


class Metrics:                                                         
        
    def __init__(self, keys, num_mc):                                  
        
        self.keys = keys                                               
        self.num_mc = num_mc                                           
        self.mode = 'across_mc' if num_mc > 0 else 'across_batch'
 
        for key in keys:
            empty = []
            setattr(self, key + '_list', empty)
 
    def append_metrics(self, loss_dict):
               
        for key in loss_dict:
            getattr(self, key + '_list').append(loss_dict[key])
 
    def aggregate(self, stacked, stat_type):
 
        return stacked.mean() if stat_type == 'mean' else stacked.std()
 
    def is_empty(self, lst):
        if lst is None or lst == []:
            return True
        if lst[0] is None:
            return True
        return False                                                                                                                                                                                                          
        
    def get_stats(self, key, stat_type):                               
        
        lst = getattr(self, key + '_list')                             
        if self.is_empty(lst):                                         
            return None                                                
        
        if self.mode == 'across_mc':                                   
            stacked = stack(lst, dim=1)                                
            assert len(stacked.shape) == 2                             
            assert stacked.shape[1] == self.num_mc                     
        else: # across batch                                           
            stacked = stack(lst, dim=0)                                
            assert len(stacked.shape) == 1                             
        
        return self.aggregate(stacked, stat_type) 



def wandb_alert(use_wandb, step):
    if use_wandb and step % 100000 == 0:
        wandb.alert(
            title='100k steps',
            text=f'Step is {step}',
            wait_duration = 300
        )



def print_total_params(trainer):
    total = 0
    for n, p in trainer.model1.named_parameters():
        size = p.numel()
        #print(n, size)
        total += size
    print("Total Params: {}".format(total))


def set_devices():

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return use_cuda, device


def monitoring(trainer,):
    monitor_grads(trainer.score_model)
    monitor_weights(trainer.score_model)

def monitor_weights(model):
    for n, p in model.named_parameters():
        print(n, p.mean(), p.std())

def monitor_grads(model):
    for n,p in model.named_parameters():
        if p.grad is not None:
            print("p grad", p.grad.abs().mean())


class GMM_Config:


    def __init__(self, layout, N0, N1):

        self.N0 = N0
        self.N1 = N1
        self.d = 2 # some code assumes 2 for now
        self.layout = layout
        # note if changing these, change 
        # ranges and limits for plots
        self.min_x = -10.0
        self.max_x = 10.0
        self.min_y = -20
        self.max_y = 20

class GMM:

    def __init__(self, config):


        self.d = config.d
        self.layout = config.layout 

        self.N0 = config.N0
        self.N1 = config.N1
        
        self.p0s = ones(self.N0) / self.N0
        self.p1s = ones(self.N1) / self.N1

        self.C0s = eye(self.d).unsqueeze(0).repeat(self.N0, 1, 1)
        self.C1s = eye(self.d).unsqueeze(0).repeat(self.N1, 1, 1)
   
        self.min_x = config.min_x
        self.max_x = config.max_x
        self.min_y = config.min_y
        self.max_y = config.max_y

        self.mu0s = self.make_means(xval=self.min_x, K=self.N0).float()
        self.mu1s = self.make_means(xval=self.max_x, K=self.N1).float()

        self.q0 = D.MixtureSameFamily(
            D.Categorical(self.p0s),
            D.MultivariateNormal(self.mu0s, self.C0s)
        )

        self.q1 = D.MixtureSameFamily(
            D.Categorical(self.p1s),
            D.MultivariateNormal(self.mu1s, self.C1s)
        )
   
    def make_means(self, xval, K):

        means = []

        if self.layout=='flow':

            for yval in linspace(self.min_y, self.max_y, K+2).numpy()[1:-1]:
                means.append([xval, yval])

        elif self.layout=='diffusion':

            for k in range(K):

                xval = D.Uniform(low=self.min_x, high=self.max_x).sample()
                yval = D.Uniform(low=self.min_y, high=self.max_y).sample()
                means.append([xval, yval])
        else:
            assert False
        
        means = torch.tensor(means)
        return means



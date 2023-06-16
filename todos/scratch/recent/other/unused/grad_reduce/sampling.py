import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from models import TinyNet
from data import DataModule
from cv import ControlRegression
from torch.func import functional_call, vmap, grad
import wandb
import matplotlib.pyplot as plt


def to_np(x):
    return x.cpu().detach().numpy()

class Sampler:
    def __init__(self, trainer):

        self.trainer = trainer
        
    def prior_samples(self, N):
        return self.trainer.datamodule.prior_samples(N)
    
    def data_samples(self, N):
        return self.trainer.datamodule.q1.sample(sample_shape=(N,))

    def plot_tensor(self, a, label):
        a = to_np(a)
        x, y = a[:,0], a[:,1]
        plt.scatter(x, y, label=label)

    def sample_from_model(self, N, which_model):

        if which_model == 'model1':
            model = self.trainer.model1
        
        elif which_model == 'model2':
            model = self.trainer.model2
        
        else:
            assert False

        x0 = self.prior_samples(N)
        xt = x0
        ts = torch.linspace(0, 1, self.trainer.sampling_steps).type_as(xt)
        dt = ts[1] - ts[0]
        for t in ts:
            tvec = torch.ones(N,1).type_as(xt) * t
            xt = xt + model(xt, tvec) * dt
        return x0, xt

    def plot_samples(self, N, step, which_model, use_wandb):
        x1 = self.data_samples(N)
        self.plot_tensor(x1, 'real')
        prior, samples = self.sample_from_model(N, which_model)
        self.plot_tensor(prior, 'prior')
        self.plot_tensor(samples, 'model samples')
        plt.legend()
        #plt.show()
        if use_wandb:
            wandb.log({"samples_{}".format(which_model): wandb.Image(plt)}, step=step)
        plt.clf()



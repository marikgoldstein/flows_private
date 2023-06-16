import torch
import torch.nn as nn
import sys
from functorch import vmap
import seaborn as sns
import numpy as np
from math import pi
import time
import torch.distributions as D
import numpy as np
from models import TinyNet
from numerical_lib import (
    randn, cat, ones, ones_like, eye, linspace, stack
)
import torch.distributions as D

Uniform = torch.distributions.Uniform

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
        self.config = config
        self.N1 = config.N1
        self.p1s = ones(self.config.N1) / self.config.N1
        self.C1s = eye(self.d).unsqueeze(0).repeat(self.config.N1, 1, 1)
        self.mu1s = self.make_means(xval=self.config.max_x, K=self.config.N1).float()
        self.q1 = D.MixtureSameFamily(
            D.Categorical(self.p1s),
            D.MultivariateNormal(self.mu1s, self.C1s)
        )

    def make_means(self, xval, K):
        means = []
        for k in range(K):
            xval = D.Uniform(low=self.config.min_x, high=self.config.max_x).sample()
            yval = D.Uniform(low=self.config.min_y, high=self.config.max_y).sample()
            means.append([xval, yval])
        means = torch.tensor(means)
        return means





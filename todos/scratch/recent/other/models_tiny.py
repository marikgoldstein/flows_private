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

from utils_numerical import (
    randn, cat, ones, ones_like, eye, linspace
)
import copy

Linear = nn.Linear
Parameter = nn.Parameter
ModuleList = nn.ModuleList
Sequential = nn.Sequential

def get_tiny_net(d):
    return TinyNet(d)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = Parameter(randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]

class TinyNet(nn.Module):
    
    def __init__(self, d):

        super().__init__()
        self.d = d
        self.time_embed_dim = 16
        self.space_embed_dim = 16
        self.layers = 4
        self.hidden_size = self.space_embed_dim + self.time_embed_dim
        self.time_embed = Sequential(
            GaussianFourierProjection(embed_dim=self.time_embed_dim),
            Linear(self.time_embed_dim, self.time_embed_dim),
        )
        self.act = nn.SiLU()
        self.linear_in = Linear(d, self.space_embed_dim)
        linears = []
        for l in range(self.layers):
            linears.append(Linear(self.hidden_size, self.hidden_size))
        self.linears = ModuleList(linears)
        self.linear_out = Linear(self.hidden_size, d)

    def forward(self, x, t):
        batch_size = x.shape[0]
        assert x.shape[1] == self.d
        ht = self.act(self.time_embed(t))
        hx = self.act(self.linear_in(x))
        h = cat([hx, ht], dim=-1)
        for fc in self.linears:
            h = self.act(fc(h))
        h = self.linear_out(h)
        return h


import torch
import torch.nn as nn
import sys
from functorch import vmap
import numpy as np
from math import pi
import time
import torch.distributions as D
import numpy as np
import numpy as np
import wandb
from diffusers.models import AutoencoderKL
Uniform = torch.distributions.Uniform

# from malbergo
def logit_eps(x, eps=1e-4):
    logit_x = torch.special.logit(eps + (1 - 2*eps)*x)
    return logit_x
 
# from malbergo
def inv_logit_eps(logit_x, eps=1e-4):
    return (torch.sigmoid(logit_x) - eps) / (1-2*eps)

# from Chin-Wei Huang, Variational perpsective on Diffusions
class CenterTransform(nn.Module):
    
    def __init__(self):
        super(CenterTransform, self).__init__()
    
    def forward_transform(self, x, logpx=None):
        # Rescale from [0, 1] to [-1, 1]
        y = x * 2.0 - 1.0
        if logpx is None:
            return y
        return y, logpx + self._logdetgrad(x).view(x.size(0), -1).sum(1)
 
    def reverse(self, y, logpy=None, **kwargs):
        # Rescale from [-1, 1] to [0, 1]
        x = (y + 1.0) / 2.0
        if logpy is None:
            return x
        return x, logpy - self._logdetgrad(x).view(x.size(0), -1).sum(1)
 
    def _logdetgrad(self, x):
        return (torch.ones_like(x) * 2).log()
 
    def __repr__(self):
        return "{name}({alpha})".format(name=self.__class__.__name__, **self.__dict__)


class Encoder(nn.Module):
      
    def __init__(self, config, device):
        super(Encoder, self).__init__()
        self.config = config
        self.dataset = config.dataset
        self.device = device
        self.use_vae = self.config.use_vae
        self.transform = CenterTransform()
        self.dequantize = config.dequantize
        if self.use_vae:
            assert self.config.dataset == 'imagenet256'
            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
            self.vae_const = 0.18215
        # main two methods are encode / decode
        # either stable diffusion vae, or simple data scaling

    def xdata_okay(self, x):
        assert torch.all(x >= 0.0) and torch.all(x <= 1.0)

    def encode(self, x):

        self.xdata_okay(x)

        if self.use_vae:

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                z = self.vae.encode(x).latent_dist.sample().mul_(self.vae_const)
            ldj = None

        else:
        
            if self.dequantize:
                x = self.uniform_dequantization(x) 

            z, ldj = self.scaler(x) # [0,1] to [-1, 1]

        return z, ldj

    def decode(self, z):

        if self.use_vae:                                                                                                                            
            x = self.vae.decode(z / self.vae_const).sample
        else:
            x = self.inverse_scaler(z) # approx [-1,1] to approx [0,1]

        return x

    # needs to be adjusted if data transform is changed!!!
    def nelbo_to_bpd(self, nelbo, ldj):
        assert False, 'needs to be changed to fit whatever the code does re transforms'
        elbo = -nelbo
        elbo += ldj
        elbo_per_dim = elbo / (self.config.C * self.config.W * self.config.H)
        nelbo_per_dim = -elbo_per_dim
        nelbo_per_dim_log2 = nelbo_per_dim / np.log(2)
        offset = 8.0 if self.dequantize else 0.0
        bpd = nelbo_per_dim_log2 + offset
        return bpd
      
    def forward(self, x):
        assert False
      
    def uniform_dequantization(self, batch):
        return (batch * 255 + torch.rand_like(batch)) / 256
      
    def scaler(self, x):
        assert x.min() >= 0.
        assert x.max() <= 1.
        return self.transform.forward_transform(x, 0)
      
    def inverse_scaler(self, x):
        y = self.transform.reverse(x)
        return y

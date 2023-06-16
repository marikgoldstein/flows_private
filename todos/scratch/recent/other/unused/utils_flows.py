import torch
import torch.nn as nn

# local
Normal = torch.distributions.Normal
from utils_numerical import (
    cat, stack, zeros, zeros_like, ones, ones_like, randn, randn_like, rand, rand_like,
    flip, sqrt, mat_square_root, matrix_exp,
    trace_fn,
    batch_transpose,
    eye,
    linspace,
    sqnorm
)


class FlowPrior(nn.Module):
   
    def __init__(self, config, trainer):
        super().__init__()
        self.config = config
        self.trainer = trainer

        if self.config.flow_prior_type == 'gaussian':
            zero = torch.zeros(1).to(self.config.device) 
            one = torch.ones(1).to(self.config.device)
            self.q0 = Normal(zero, one)
                                                                                                                                                    
        elif self.config.flow_prior_type == 'mnist':
            
            print("Using MNIST as prior")

        else:

            assert False

    def sample(self, bsz):

        if self.config.flow_prior_type == 'gaussian':

            x0 = self.q0.sample(sample_shape=(bsz, self.config.d)).squeeze(-1)

        elif self.config.flow_prior_type == 'mnist':

            x0 = self.trainer.get_batch(bsz = bsz)
            x0 = self.trainer.encoder.preprocess(x0)
        
        else:   
            assert False

        return x0



import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import math
from utils import gradi, compute_div

class PFlowRHS(nn.Module):
    def __init__(self, field, likelihood = False):
        super(PFlowRHS, self).__init__()
        self.field = field
        self.likelihood = likelihood
        self.rhs = lambda x, t, y: self.field(x, t, y)
        
    def forward(self, t, states):
        (x, y, _ ) = states
        y = y.long()
        t_arr = torch.ones(x.shape[0]).to(x) * t
        if not self.likelihood:

            return (self.rhs(x, t_arr, y), torch.zeros_like(y), torch.zeros(x.shape[0]).to(x))
        else:
            return (self.rhs(x, t_arr, y), torch.zeros_like(y), -compute_div(self.rhs, x, t_arr, y))
        
    #def reverse(self, t, states):
    #    (x, y, _) = states
    #    y = y.long()
    #    t_arr = torch.ones(x.shape[0]).to(x) * t
    #    if not self.likelihood:
    #        return (-self.rhs(x, t_arr, y), torch.zeros_like(y), torch.zeros(x.shape[0]).to(x))
    #    else:
    #        return (-self.rhs(x, t_arr, y), torch.zeros_like(y), compute_div(self.rhs, x, t_arr, y))
        
class PFlowIntegrator:
        
    def __init__(self, config):
        
        self.config = config

    def __call__(self, field, initial_state, y, likelihood = False, cheap = False, reverse=False):
        # rollout 
        conf = self.config
        rhs = PFlowRHS(field = field, likelihood = likelihood)
        atol = conf.integration_atol
        rtol = conf.integration_rtol
        start = conf.T_min
        end = conf.T_max
        
        if cheap:
            n_step = int(conf.n_sample_steps / 10)
            print(f"Doing a cheap sampling run with 10% of steps: {n_step}")
        else:
            n_step = conf.n_sample_steps
            print(f"Doing a regular sampling run: {n_step}")
        
        z0 = initial_state
        
        if reverse:
            t = torch.linspace(end, start, n_step).to(z0)
        else:
            t = torch.linspace(start, end, n_step).to(z0)
        
        dlogp = torch.zeros(z0.shape[0]).to(z0)
        x, _y, dlogp = odeint(
            rhs,
            (z0, y, dlogp),
            t,
            method=conf.integration_method,
            atol=[atol, atol, atol],
            rtol=[rtol, rtol, rtol],
        )
        return x, dlogp
                                                  

class SDEIntegrator:

    def __init__(self, config):

        self.config = config
        self.start = config.T_min
        self.end = config.T_max
        self.eps = config.epsilon.to(config.device)
        self.n_save = 1
        self.n_likelihood = 10
        self.n_step = config.n_sample_steps
        self.bf = lambda b,s,x,t,y: b(x,t,y) + self.eps * s(x,t,y)
        self.br = lambda b,s,x,t,y: b(x,t,y) - self.eps * s(x,t,y)
        # d/dt logp. assumes int from 1 to 0
        #self.dt_logp = lambda b,s,x,t:  -(compute_div(self.bf, x, t) + self.eps * image_sum(s(x,t).pow(2)))

    def step_forward_heun(self, b, s, x, t, y, dt):
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(dt)*torch.randn_like(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.bf(b, s, xhat, t + dt, y)
        xp   = xhat + dt*K1
        K2   = self.bf(b, s, xp, t + dt, y)
        return xhat + 0.5*dt*(K1 + K2)

    def step_forward(self, b, s, x, t, y, dt):
        """Euler-Maruyama."""
        dW = torch.sqrt(dt)*torch.randn_like(x)
        return x + self.bf(b, s, x, t, y)*dt + torch.sqrt(2*self.eps)*dW

    def step_reverse(self, b, s, x, t, y, dt):
        """Euler-Maruyama."""
        dW = torch.sqrt(dt)*torch.randn_like(x)
        return x - self.br(b, s, x, t, y)*dt + torch.sqrt(2*self.eps)*dW
        
    def step_reverse_heun(self, b, s, x, t, y, dt):
        """Heun Step -- see https://arxiv.org/pdf/2206.00364.pdf, Alg. 2"""
        dW   = torch.sqrt(dt)*torch.randn_like(x)
        xhat = x + torch.sqrt(2*self.eps)*dW
        K1   = self.br(b, s, xhat, t - dt, y)
        xp   = xhat - dt*K1
        K2   = self.br(b, s, xp, t - dt, y)
        return xhat - 0.5*dt*(K1 + K2)

    def __call__(self, b, s, initial_state, y, method='heun', cheap = False):
        # rollout forward, modified not to save intermediate states
        """Solve the forward-time SDE to generate a batch of samples."""
        n_step = self.n_step
        
        if cheap:
            print("Doing a cheap sampling run with 10% of steps")
            n_step = int(n_step / 10)
        
        z0 = initial_state
        #zs = torch.zeros_like(z0)[None,...].repeat(self.n_save, 1, 1, 1, 1)

        ts = torch.linspace(self.start, self.end, n_step).to(z0)
        dt = (ts[1] - ts[0])
        zt = z0
        
        for i, t in enumerate(ts):
            tarr = torch.ones_like(y) * t
            if method == 'heun':
                zt = self.step_forward_heun(b, s, zt, tarr, y, dt)
            else:
                zt = self.step_forward(b, s, zt, tarr, y, dt)

        return zt

    '''
    def step_likelihood(self, like, b, s, x, t):
        """Forward-Euler."""
        return like - self.dt_logp(b, s, x, t)*self.dt

    def rollout_likelihood(self, b, s, init):
        # TODO mg
        """Solve the reverse-time SDE to generate a likelihood estimate."""
        # n_step = int(torch.ceil(1.0/self.dt))
        bsz, d  = init.shape
        likes  = torch.zeros((self.n_likelihood, bsz)).to(init)
        xs     = torch.zeros((self.n_likelihood, bsz, d)).to(init)

        # TODO: for more general dimensions, need to replace these 1's by something else.
        x    = init.repeat((self.n_likelihood, 1, 1)).reshape((self.n_likelihood*bsz, d))
        like = torch.zeros(self.n_likelihood*bsz).to(x)
        save_counter = 0

        for ii,t in enumerate(self.ts):
            t = self.end - t.to(x)
            x    = self.step_reverse_heun(b, s, x, t)
            like = self.step_likelihood(like, b, s, x, t-self.dt) # semi-implicit discretization?
                             
        xs, likes = x.reshape((self.n_likelihood, bsz, d)), like.reshape((self.n_likelihood, bsz))

        # only output mean
        return xs, torch.mean(likes, axis=0)
    '''



import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.optim import Adam
from collections import defaultdict
from torch.func import functional_call, vmap, grad


class LinearModel(nn.Module):
    
    def __init__(self, d):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(d, 1)
    
    def forward(self, x):
        return self.fc(x).squeeze(-1)

def loss_fn(y, yhat):
    return (y - yhat).pow(2).mean()

def compute_loss_stateless_model(params, buffers, model, x, y):
    x, y = x[None,...], y[None,...]
    yhat = functional_call(model, (params, buffers), (x,))
    return loss_fn(y, yhat)

class Trainer:

    def __init__(self, use_functorch):
    
        # data
        self.bsz = 128
        self.d = 2
        self.real_W = torch.rand(self.d)
        
        # stateful model stuff
        self.model = LinearModel(d = self.d)
        self.optimizer = Adam(self.model.parameters(), lr=1e-2)

        self.use_functorch = use_functorch
        if use_functorch:
            self.setup_functorch()
        self.step_fn = self.fstep_fn if self.use_functorch else self.usual_step_fn

        # training
        self.current_step = 0
        self.total_steps = 1000
        self.monitor_every_n_steps = 100

    def setup_functorch(self,):
        self.params = {k: v.detach() for k, v in self.model.named_parameters()}
        self.buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        self.loss_grad_fn = grad(compute_loss_stateless_model)
        self.batched_loss_grad_fn = vmap(self.loss_grad_fn, in_dims=(None, None, None, 0, 0))

    def get_loss_functorch(self, x, y):
        return compute_loss_stateless_model(self.params, self.buffers, self.model, x, y)
    
    def get_grads_functorch(self, x, y):
        return self.batched_loss_grad_fn(self.params, self.buffers, self.model, x, y)

    def make_batch(self,):
        x = torch.randn(self.bsz, self.d)
        y = (x * self.real_W).sum(-1)
        return x, y
       
    def get_loss_stateful(self, x, y):
        return (y - self.model(x)).pow(2).mean()

    def train_loop(self,):
        for step in range(self.total_steps):
            monitor = step % self.monitor_every_n_steps == 0
            self.optimizer.zero_grad()
            x, y = self.make_batch()
            loss = self.step_fn(x, y, compute_loss = monitor)
            self.optimizer.step()
            self.current_step += 1
            if monitor:
                print("loss:{}".format(loss))

    # okay to change self.model.params because it changes detached ones too
    def copy_grads(self, per_sample_grads):
        for n, p in self.model.named_parameters():
            if n in per_sample_grads:
                p.grad = per_sample_grads[n].mean(0)

    def fstep_fn(self, x, y, compute_loss):
        bsz = x.shape[0]
        per_sample_grads = self.get_grads_functorch(x, y)
        self.copy_grads(per_sample_grads)
        return self.get_loss_functorch(x, y).item() if compute_loss else None

    def usual_step_fn(self, x, y, compute_loss):
        loss = self.get_loss_stateful(x, y)
        loss.backward()
        return loss.item()

if __name__=='__main__':

    trainer = Trainer(use_functorch = True)
    trainer.train_loop()

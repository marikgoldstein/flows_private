import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad, vjp
from torch.distributions import Uniform, Normal

def batched_dot(a, b):
    bsz = a.shape[0]
    assert b.shape[0] == bsz
    dim = a.shape[1]
    return torch.bmm(a[:, None, :], b[:,:,None]).squeeze(-1).squeeze(-1)

def f(x):
    return 2*x

def div_f(x):
    return 4.0 * torch.ones(x.shape[0])

if __name__ == '__main__':

    bsz = 1000
    D = 2
    lhs_list = []
    rhs_list = []
    mu = torch.ones(bsz, 2)
    std = torch.ones(bsz, 2) * 100.0
    eps = torch.randn_like(mu)
    x = mu + std*eps
    score = -eps/std
    lhs = batched_dot(score, f(x)).mean(0)
    rhs = div_f(x).mean(0)
    print(lhs)
    print(rhs)

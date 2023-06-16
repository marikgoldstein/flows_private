import torch
import torch.distributions as D

Mix = D.MixtureSameFamily
Cat = D.Categorical
N = D.Normal

bsz = 128
C = 3
H = 32
W = 32

# bsz C H W ->  C H W bsz
def batch_last(x):
    return x.permute(1, 2, 3, 0)

# C H W bsz -> bsz C H W
def batch_first(x):
    return x.permute(3, 0, 1, 2)

pis = torch.ones(bsz, C, H ,W)
mus = torch.randn(bsz, C, H ,W)
stds = torch.ones(bsz, C, H, W)
mus = batch_last(mus)
stds = batch_last(stds)
pis = batch_last(pis)
assert mus.shape == (C, H ,W ,bsz)
gmm = Mix(Cat(pis), N(mus,stds))
x=gmm.sample()
assert x.shape == (C, H, W)


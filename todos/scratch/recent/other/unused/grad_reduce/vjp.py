import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad, vjp
from torch.distributions import Uniform, Normal
torch.manual_seed(0)

bsz = 10000#1024 #512
D = 2


def make_t():
    t = Uniform(low=0.0,high=1.0).sample(sample_shape=(bsz,))
    t = t.squeeze(-1)
    t = t[:, None]
    return t

def make_x1():
    return torch.randn(bsz, D)

def field_from_x1_xt(x1, xt, t):
    return x1 + (t*x1)/(1-t) -  xt/(1-t)

def field_from_x1_x0(x1, x0, t):
    return x1 - x0

def xt_given_x1(x1, t):
    return Normal(loc = t*x1, scale=(1-t).repeat(1,2))

def sample_xt_given_x1(x1, t):
    q = xt_given_x1(x1, t)
    xt = q.sample()
    xt_mean = q.loc
    return xt, xt_mean

def x0_from_x1_xt_t(x1, xt, t):
    return xt/(1-t) - (t*x1)/(1-t)

def make_batch():
    x1 = make_x1()
    t = make_t()
    xt, xt_mean = sample_xt_given_x1(x1, t)
    x0 = x0_from_x1_xt_t(x1, xt, t)
    ut = field_from_x1_x0(x1, x0, t)
    ut2 = field_from_x1_xt(x1, xt, t)
    for tensor in [x1, t, xt, x0, ut, ut2]:
        assert len(tensor.shape) == 2
    bsz = x1.shape[0]
    assert t.shape==(bsz, 1)
    return x1, x0, xt, t, ut

def batched_dot(a, b):
    bsz = a.shape[0]
    assert b.shape[0] == bsz
    dim = a.shape[1]
    ret1 = torch.bmm(a[:, None, :], b[:,:,None]).squeeze(-1).squeeze(-1)
    ret2 = (a*b).sum(-1)
    assert torch.allclose(ret1, ret2)
    return ret1


'''
def model_to_params_and_buffers(model):
    params = {k : v.detach() for k, v in model.named_parameters()}
    buffers = {k : v.detach() for k, v in model.named_buffers()}
    return params, buffers

def compute_loss_stateless_model(params, buffers, model, xt, t, ut):
    xt, t, ut = xt[None,...], t[None,...], ut[None,...]
    vt = functional_call(model, (params, buffers), (xt, t))
    return (vt - ut).pow(2).sum(-1).mean()

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.fc_x = nn.Linear(D, D)
        self.fc_t = nn.Linear(1, D)
    def forward(self, xt, t):
        return self.fc_x(xt) + self.fc_t(t)

# returns func that calls model(space, time)
def get_model_fn(model, params, buffers):
    model_fn = lambda space, time : functional_call(model, (params, buffers), (space, time))
    return model_fn

# model fn should come from get model fn
def get_div_fn(model_fn):
    def div_fn(xt, t, eps):
        curried = lambda arg: model_fn(arg, t) # curry t 
        (_, vjp_fn) = torch.func.vjp(curried, xt) # use xt
        return vjp_fn(eps) # use eps
    return div_fn

def get_div_model_fn(model, params, buffers):

    def div_model_fn(xt, t, eps):
        model_fn = get_model_fn(model, params, buffers)
        div_fn = get_div_fn(model_fn)
        eps_J = div_fn(xt, t, eps)[0]
        return batched_dot(eps_J, eps)
    
    return div_model_fn


model = Model()
params, buffers = model_to_params_and_buffers(model)
loss_grad_fn = grad(compute_loss_stateless_model)
batched_loss_grad_fn = vmap(loss_grad_fn, in_dims = (None, None, None, 0, 0, 0))
div_model_fn = get_div_model_fn(model, params, buffers)

#grad = batched_loss_grad_fn(params, buffers, model, xt, t, ut)
mxt = functional_call(model, (params, buffers), (xt, t))
rhs = div_model_fn(xt, t, eps_div)
'''

def get_simple_batch():
    x1, x0, xt, t, ut = make_batch()
    q = xt_given_x1(x1, t)
    mu, std = q.loc, q.scale
    return x1, t, mu, std

def sample_and_score(mu, std):
    eps = torch.randn_like(mu)
    xt = mu + eps * std
    score = -eps / std
    return xt, score

def f(x):
    return 2*x

def div_f(x):
    return 4 * torch.ones(x.shape[0])

#x1, t, mu, std = get_simple_batch()
#xt, score = sample_and_score(mu, std)

mu = torch.zeros(bsz, 2)
std = torch.ones(bsz, 2)
xt = torch.randn(bsz, 2)
score = -xt
lhs = batched_dot(score, f(xt))
rhs = div_f(xt)
print("lhs rhs means", lhs.mean(), rhs.mean())

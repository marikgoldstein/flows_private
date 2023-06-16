import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
CAT = D.Categorical
MVN = D.MultivariateNormal
MIX = D.MixtureSameFamily
UNIF = D.Uniform
torch.manual_seed(0)
Linear = nn.Linear
Parameter = nn.Parameter
ModuleList = nn.ModuleList
Sequential = nn.Sequential

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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
        h = torch.cat([hx, ht], dim=-1)
        for fc in self.linears:
            h = self.act(fc(h))
        h = self.linear_out(h)
        return h



def make_functional(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        return torch.func.functional_call(stateless_mod, new_params_dict, args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values

def make_functional_with_buffers(mod, disable_autograd_tracking=False):
    params_dict = dict(mod.named_parameters())
    params_names = params_dict.keys()
    params_values = tuple(params_dict.values())

    buffers_dict = dict(mod.named_buffers())
    buffers_names = buffers_dict.keys()
    buffers_values = tuple(buffers_dict.values())
    
    stateless_mod = copy.deepcopy(mod)
    stateless_mod.to('meta')

    def fmodel(new_params_values, new_buffers_values, *args, **kwargs):
        new_params_dict = {name: value for name, value in zip(params_names, new_params_values)}
        new_buffers_dict = {name: value for name, value in zip(buffers_names, new_buffers_values)}
        return torch.func.functional_call(stateless_mod, (new_params_dict, new_buffers_dict), args, kwargs)
  
    if disable_autograd_tracking:
        params_values = torch.utils._pytree.tree_map(torch.Tensor.detach, params_values)
    return fmodel, params_values, buffers_values


def print_params(model):
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        print(n)
    print("Total Params:", total)

def make_param_name_dict(model):
    param_names = {}
    for i, (n, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            param_names[i] = n 
    return param_names



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
        cov = torch.eye(self.d)[None,...].repeat(self.config.N1, 1, 1)
        mu = []
        lowx, highx, lowy, highy = config.min_x, config.max_x, config.min_y, config.max_y
        mu = torch.tensor([
            [UNIF(lowx, highx).sample(),UNIF(lowy, highy).sample()]
            for k in range(config.N1)
        ]).float()
        self.q1 = MIX(
            CAT(torch.ones(self.config.N1) / self.config.N1),
            MVN(mu, covariance_matrix=cov)
        )

class Experiment:

    def __init__(self,):

        self.bsz = 4
        self.d = 2
        self.q0 = D.Normal(torch.zeros(0), torch.ones(1))
        self.q1 = GMM(GMM_Config(layout='diffusion', N0=1, N1=2)).q1
        self.model = TinyNet(d=self.d)
        print_params(self.model)
        self.param_names = make_param_name_dict(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.current_step = 0
        self.total_steps = 100
        self.U = UNIF(low=0.0, high=1.0)


    def get_per_datapoint_grads(self,):

        model = self.model
        x0, x1, xt, t, ut = self.get_x0_x1_xt_t_ut()
        fmodel, params, buffers = make_functional_with_buffers(model)
        loss_grad = torch.func.grad(compute_loss_stateless_model)
        batched_loss_grad = torch.func.vmap(loss_grad, in_dims=(None, None, None, 0, 0, 0))
        psg = batched_loss_grad(params, buffers, fmodel, xt, t, ut)
        per_sample_grads = {
            self.param_names[k] : psg[k] for k in self.param_names
        }
        for k in per_sample_grads:
            print(k, per_sample_grads[k].shape)

    def get_x1_batch(self,):
        return self.q1.sample(sample_shape=(self.bsz,))
    
    def get_unif_times(self,):
        return self.U.sample(sample_shape=(self.bsz,))[:,None]

    def phi(self, x0, x1, t):
        return t*x1 + (1-t)*x0

    def dphi(self, x0, x1, t):
        return x1 - x0

    def q_xt_given_x1(self, x1, t):
        return D.Normal(loc = t*x1, scale = (1 - t).repeat(1,2))

    def sample_xt_given_x1(self, x1, t):
        return self.q_xt_given_x1(x1, t).sample()

    def x0_from_x1_xt_t(self, x1, xt, t):
        return xt/(1-t) - (t*x1)/(1-t)

    def cond_field(self, x1, xt, t):
        return x1 + (t*x1)/(1-t) -  xt/(1-t)

    def get_x0_x1_xt_t_ut(self,):
        x1 = self.get_x1_batch()
        t = self.get_unif_times()
        xt = self.sample_xt_given_x1(x1, t)
        ut = self.cond_field(x1, xt, t)
        x0 = self.x0_from_x1_xt_t(x1, xt, t)
        return x0, x1, xt, t, ut

    def norm(self, x):
        return x.pow(2).sum(-1)

    def loss_fn(self, vt, ut): #control):
        return self.norm(vt - ut) #, self.norm(vt - ut - control)
    
    def do_one_step(self,):
        self.optimizer.zero_grad()
        x0, x1, xt, t, ut = self.get_x0_x1_xt_t_ut()
        vt = self.model(xt, t.squeeze())
        loss = self.loss_fn(vt, ut)
        loss_to_print = loss
        loss_for_backprop = loss
        loss_for_backprop.backward()
        if self.current_step % 100 == 0:
            print("loss:{}".format(loss_to_print.item()))
        self.optimizer.step()
        self.current_step += 1

    def train_loop(self,):

        for step in range(self.total_steps):
            self.do_one_step()


def loss_fn(vt, ut):
    return (vt - ut).pow(2).sum(-1).mean()

def compute_loss_stateless_model(params, buffers, fmodel, xt, t, ut):
    xt, t, ut = xt[None,...], t[None,...], ut[None,...]
    vt = fmodel(params, buffers, xt, t.squeeze(-1))
    return loss_fn(vt, ut)



if __name__=='__main__':

    exp = Experiment()
    #exp.train_loop()
    exp.get_per_datapoint_grads()


#def test(self,):
#    x0, x1, xt, t, ut = self.geti_x0_x1_xt_t_ut()
#    x0_reconstructed = self.x0_from_x1_xt_t(x1, xt, t)
#    ut2 = x1 - x0_reconstructed
#    print("diff", (ut-ut2).pow(2).mean())

#def density_xt_given_x1(self, x1, xt, t):
#    q = self.q_xt_given_x1(x1, t)
#    lp = q.log_prob(xt)
#    lp = lp.sum(-1)
#    p = lp.exp()
#    return p

#control = self.compute_control(x1, xt, t)
#loss, loss_control = self.loss_fn(vt, ut, control)
#loss, loss_control = loss.mean(), loss_control.mean()
#loss_for_backprop = loss_control


#def estimate_mean_g_func(self, xt, t):
#    bsz = xt.shape[0]
#    x1 = self.get_x1_batch()
#    numer = self.density_xt_given_x1(x1, xt, t)
#    denom = torch.zeros(x1.shape[0])
#    MC = 1000
#    for mc in range(MC):
#        x1p = self.get_x1_batch()
#        denom += self.density_xt_given_x1(x1p, xt, t)
#    denom += numer
#    denom = denom / (MC+1)
#    weight = torch.where(numer > 0, numer / denom, torch.zeros_like(numer))
#    g = self.g_func(x1, t)
#    mu = weight[:, None] * g
#    return mu
#
#def g_func(self, x1, t):
#    return x1
#
#def compute_control(self, x1, xt, t):
#    alpha = 1.0 
#    g = self.g_func(x1, t)
#    mu_g = self.estimate_mean_g_func(xt, t)
#    g_diff = g - mu_g
#    return alpha * g_diff



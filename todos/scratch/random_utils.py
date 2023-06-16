import torch


def gradi(fval ,x):

    return torch.autograd.grad(
        f_val[:,i].sum(), x, create_graph=True
    )[0][:,i]


def compute_div(f, x, t):
    bsz = x.shape[0]
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        t.requires_grad_(True)
        f_val = f(x, t)
        div = 0.0
        for i in range(x.shape[1]):
            div += gradi(f_val, x)
    return div.view(bsz,)



import torch
import numpy as np
import scipy

def sqnorm(x):
    return x.pow(2).sum(-1)

def expand1(t, dim):
    return t.unsqueeze(dim)

def expand2(t, dim):
    t = expand1(t, dim)
    return expand1(t, dim)

def linspace(start, end, steps):
    return torch.linspace(start, end, steps)

def eye(D):
    return torch.eye(D)

def cat(x, dim):
    return torch.cat(x, dim=dim)

def chunk(x, chunks, dim):
    return torch.chunk(x, chunks, dim=dim)

def stack(x, dim):
    return torch.stack(x, dim=dim)

def zeros(shape):
    return torch.zeros(shape)

def zeros_like(x):
    return torch.zeros_like(x)

def ones(shape):
    return torch.ones(shape)

def ones_like(x):
    return torch.ones_like(x)

def randn(shape):
    return torch.randn(shape)

def randn_like(shape):
    return torch.randn_like(shape)

def rand(shape):
    return torch.rand(shape)

def rand_like(x):
    return torch.rand_like(x)

def flip(x, dims):
    return torch.flip(x, dims=dims)

def sqrt(x):
    return torch.sqrt(x)

def matrix_exp(A):
    return torch.matrix_exp(A)

#def matrix_exp(t, A):
#    t = t.unsqueeze(-1).unsqueeze(-1)
#    mexp = torch.matrix_exp(t * A)
#    return mexp

def mat_square_root(A):
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    s = torch.sqrt(S)
    rootA = U @ torch.diag_embed(s) @ V
    return rootA

''' 
def batch_transpose(A):
    if len(A.shape) == 3:
        return A.permute(0, 2, 1)
    elif len(A.shape) == 4:
        return A.permute(0, 1, 3, 2)
    elif len(A.shape) == 2:
        return A.permute(1, 0)
    else:
        raise NotImplementedError
'''

def batch_transpose(A):
    if len(A.shape) == 3:
        return A.permute(0, 2, 1)
    elif len(A.shape) == 4:
        return A.permute(0, 1, 3, 2)
    elif len(A.shape) == 2:
        return A.permute(1, 0)
    else:
        raise NotImplementedError

def inv(A):
    return torch.linalg.inv(A)

def trace_fn(A):
    """returns the trace of a matrix"""
    if len(A.shape) == 3:
        return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    elif len(A.shape) == 4:
        return A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).sum(-1)
    else:
        raise NotImplementedError


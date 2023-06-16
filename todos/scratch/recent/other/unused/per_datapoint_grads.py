import torch
from functorch import make_functional_with_buffers, vmap, grad

batch_size, dim = 4, 2
x = torch.randn(batch_size, 2)
W = torch.randn(1, 2)
y = (W @ x.permute(1,0)).squeeze(0)
model = torch.nn.Linear(2, 1)
fmodel, params, buffers = make_functional_with_buffers(model)

param_names = {}
for i, (name, param) in enumerate(model.named_parameters()):
    if param.requires_grad:
        param_names[i] = name
print("param names", param_names)

def loss_fn(preds, target):
    return (preds - target).mean()

def compute_loss(params, buffers, x, y):
    x, y, = x[None,...], y[None,...]
    preds = fmodel(params, buffers, x)
    loss = loss_fn(preds, y)
    return loss

loss_grad = grad(compute_loss)
batched_loss_grad = vmap(loss_grad, in_dims=(None, None, 0, 0))
per_sample_grads = batched_loss_grad(params, buffers, x, y)

# per sample grads is a tuple
# ith element is a tensor containg loss grads for ith parameter
# each grad is num datapoints by parameter dims
for k in param_names:
    print(param_names[k])
    print(per_sample_grads[k])

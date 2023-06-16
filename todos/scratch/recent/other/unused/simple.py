import torch


batch_size = 128
dim = 784

X = torch.randn(batch_size, dim)
Y = X + 0.001 * torch.randn(batch_size, dim)


def dimwise_cov(X, Y):
    return torch.einsum('bd,bd->d', X - X.mean(dim=0), Y - Y.mean(dim=0)) / (batch_size - 1)

print(dimwise_cov(X,Y).shape)

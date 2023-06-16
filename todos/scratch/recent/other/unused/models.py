import torch
import torch.nn as nn
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


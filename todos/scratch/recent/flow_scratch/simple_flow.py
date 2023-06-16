import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
import math

class Config:

    def __init__(self,):
        
        self.T_min = 1e-5
        self.T_max = 1 - 1e-5
        self.batch_size = 1024
        self.training_steps = 10000 
        self.print_loss_every = 500 # for logging
        self.sample_every = 1000 
        self.n_sample_steps = 500 # for generation
        self.num_samples = 1000 # for generation

# checkboard
class Dataset:

    def sample(self, N):

        x1a = torch.rand(N,) * 4 - 2
        x1b = torch.rand(N,) - torch.randint(0, 2, (N,)).float() * 2 + (x1a.floor() % 2)
        return torch.stack([x1a,x1b],  dim=-1)

# used in most time-dependent models model(xt, t)
# can ignore this if not interested in architectural details
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# very simple time dependent model(xt, t)
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.in_x = nn.Linear(2, 64)
        self.in_t = TimestepEmbedder(64)
        modules = [nn.Linear(128, 128), nn.SiLU()] * 5 + [nn.Linear(128, 2)]
        self.net = nn.Sequential(*modules)

    def forward(self, x, t):
        
        hx = self.in_x(x)
        ht = self.in_t(t)
        h = torch.cat([hx, ht], dim=-1)
        return self.net(h)


# q0 gaussian
# q1 data
# xt = phi(x0, t | x1) = tx1 + (1-t)x0
# dphi/dt = x1 - x0
class Flow:

    def __init__(self, config):
        
        self.config = config
    
    def get_times(self, x1):
        
        t = torch.rand(x1.shape[0],).type_as(x1)
        return t.clamp(min = self.config.T_min, max = self.config.T_max)

    def wide(self, t):
        return t[:, None]

    def loss_fn(self, x1, model, device):
        
        x0 = torch.randn_like(x1)
        t = self.get_times(x1)
        xt = self.wide(t) * x1 + self.wide(1-t) * x0
        return (model(xt, t) - (x1 - x0)).pow(2) # loss per sample

    @torch.no_grad()
    def sample(self, N, model, device):
        
        T_min, T_max = self.config.T_min, self.config.T_max
        ts = torch.linspace(T_min, T_max, self.config.num_samples).to(device)
        dt = 1 / self.config.n_sample_steps
        xt = torch.randn(N, 2)
        for t in ts:
            xt += dt * model(xt, torch.ones(N,).to(device) * t)
        return xt


class FlowTrainer:


    def __init__(self,):

        self.config = Config()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = Model()
        self.flow = Flow(self.config)
        self.data = Dataset()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def plot(self, x, title):
        
        plt.scatter(x[:,0].numpy(), x[:,1].numpy())
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.title(title)
        plt.show()

    def main_loop(self,):

        # check data
        self.plot(self.data.sample(self.config.batch_size), 'real data')

        for step in range(self.config.training_steps):

            x1 = self.data.sample(self.config.batch_size)
                      
            loss = self.flow.loss_fn(x1, self.model, self.device).mean()

            self.optimizer.zero_grad()
            
            loss.backward()
            
            self.optimizer.step()
            
            if step % self.config.print_loss_every == 0:
                
                print(f"Loss: {loss.item()}")
            
            if step % self.config.sample_every == 0:

                samples = self.flow.sample(self.config.num_samples, self.model, self.device)
                self.plot(samples, f'samples, step {step}')

if __name__ == '__main__':

    trainer = FlowTrainer()
    trainer.main_loop()








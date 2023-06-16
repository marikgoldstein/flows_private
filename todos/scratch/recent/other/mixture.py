import torch
import numpy as np
import torch.distributions as D
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.animation as animation
import seaborn as sns
sns.set_theme()

Param = nn.Parameter
Normal = D.Normal
Cat = D.Categorical
Mix = D.MixtureSameFamily

class RealData:
    def __init__(self, N):
        
        self.N = N
        self.mu_q = torch.zeros(1)
        self.sig_q = torch.ones(1)
        self.q = Normal(loc = self.mu_q, scale = self.sig_q)

    def get_batch(self,):         
        xs  = self.q.sample(sample_shape=(self.N,)).squeeze(-1)
        ys = torch.zeros_like(xs)
        return xs, ys

    def compute_data_score(self, xs):
        return -(xs - self.mu_q) / self.sig_q.pow(2)

class GMM(nn.Module):
    def __init__(self, K, std_lowerbound, freeze_means, freeze_pi):
        super().__init__()
        self.K = K
        self.freeze_means = freeze_means
        self.freeze_pi = freeze_pi
        self.logit_pi = Param(torch.randn(self.K) / self.K, requires_grad=True)
        self._mu = Param(torch.randn(self.K), requires_grad=True)
        self.scale = Param(torch.zeros(self.K), requires_grad=True)
        self.std_lowerbound = std_lowerbound
    def forward(self, xs):
        gmm = self.make_gmm()
        return gmm.log_prob(xs)

    def pi(self,):
        if self.freeze_pi:
            return torch.ones(self.K) / self.K
        else:
            return nn.Softmax(dim=-1)(self.logit_pi)

    def mu(self,):
        m = self._mu
        if self.freeze_means:
            return m.detach()
        else:
            return m

    def std(self,):
        #return nn.Softplus()(self.scale) + torch.tensor([self.std_lowerbound])
        return torch.sigmoid(self.scale) + torch.tensor([self.std_lowerbound])

    def make_gmm(self): 
        return Mix(Cat(self.pi()), Normal(self.mu(), self.std()))

    def f(self, x, k):
        pi, mu, std = self.pi(), self.mu(), self.std()
        return -(x - mu[k]) / std[k].pow(2)

    def g(self, x, k):
        summ = torch.ones_like(x)
        pi, mu, std = self.pi(), self.mu(), self.std()
        for j in range(self.K):
            if j != k:
                term1 = pi[j] / pi[k]
                term2 = std[k] / std[j]
                term3a = -.5 * (x - mu[j]).pow(2)/std[j].pow(2)
                term3b =  .5 * (x - mu[k]).pow(2)/std[k].pow(2)
                term3 = torch.exp(term3a + term3b)
                summ += term1 * term2 * term3
        return summ

    def score(self, xs,):
        gmm = self.make_gmm()
        f = gmm.log_prob(xs).sum()
        diff = torch.autograd.grad(f, xs, create_graph=True, retain_graph=True)[0]   
        return diff

    def nablaxx(self, xs):
        f = self.score(xs).sum()
        tr = torch.autograd.grad(f, xs, create_graph = True, retain_graph=True)[0]   
        return tr
    '''
    def nablaxx(self, xs):
        bsz = xs.shape[0]
        pi, mu, std = self.pi(), self.mu(), self.std()
        summ = 0.0
        # term 1 
        for k in range(self.K):
            sig2 = std[k].pow(2)
            sig4 = std[k].pow(4)
            diff_sq= (xs - mu[k]).pow(2)
            numer = - (sig2 - diff_sq) / sig4
            denom = self.g(xs, k)
            assert numer.shape == (bsz,)
            assert denom.shape == (bsz,)
            summ += numer / denom
        # term 2
        summ += self.score(xs).pow(2)
        assert summ.shape == (bsz,)
        return summ
    
    def score(self, xs,):
        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        pi, mu, std = self.pi(), self.mu(), self.std()
        #print("pi",pi.data)
        #print("std", std.data)
        #print("mu", mu.data)
        #print("xs",xs.data)       
        summ = torch.zeros_like(xs)
        for k in range(self.K):
            numer = self.f(xs,k)
            denom = self.g(xs,k)
            summ += numer / denom
        return summ
    '''
class Trainer:

    def __init__(self, N, K, mode='mle', std_lowerbound = 1.0, init_means = False, freeze_means=False, freeze_pi = False):
        self.freeze_means = freeze_means
        self.freeze_pi = freeze_pi
        assert mode in ['mle','esm', 'ism']
        self.mode = mode
        self.init_means = init_means
        # setting N = K
        self.N = N
        self.K = K
        self.std_lowerbound = std_lowerbound
        self.data = RealData(N=self.N)
        self.model = GMM(K=self.K, std_lowerbound = std_lowerbound, freeze_means = freeze_means, freeze_pi = freeze_pi)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=.9)
        self.num_steps = 0   
        self.xs, self.ys = self.data.get_batch()

        if self.mode == 'mle':
            self.step = self.step_mle
        elif mode == 'esm':
            self.step = self.step_esm
        elif mode == 'ism':
            self.step = self.step_ism
        else:
            assert False

    def step_mle(self, xs, ys):
        lp = self.model(xs)
        loss = -lp.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.num_steps +=1
        self.loss = loss.item()

    def step_esm(self, xs, ys):
        #with torch.autograd.detect_anomaly():
        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        ds = self.data.compute_data_score(xs)
        if torch.any(torch.isnan(ds)):
            assert False, 'data score is nan'
        xclone = xs.clone()
        xclone.requires_grad=True
        ms = self.model.score(xclone)
        if self.K == 1:
            print("generic score is", ms)
            print("specific score is ", -(xs - self.model.mu()) / self.model.std().pow(2))
        if torch.any(torch.isnan(ms)):
            assert False, 'model score is nan'
        assert ds.shape == (bsz,)
        assert ms.shape == (bsz,)
        loss_per_datapoint = .5 * (ds-ms).pow(2)
        loss = loss_per_datapoint.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.debug()
        self.opt.step()
        self.num_steps +=1
        self.loss = loss.item()

    def step_ism(self, xs, ys):
        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        xclone = xs.clone()
        xclone.requires_grad=True
        ms = self.model.score(xclone)
        assert ms.shape == (bsz,)
        norm_term = ms.pow(2)
        xclone2 = xs.clone()
        xclone2.requires_grad=True
        trace_term = self.model.nablaxx(xclone2)
        #if self.K == 1:
        #    print("generic trace is ", trace_term)
        #    print("specific gauss is", -1 / self.model.std().pow(2))
        loss_per_datapoint = trace_term + .5 * norm_term
        assert loss_per_datapoint.shape == (bsz,)
        loss = loss_per_datapoint.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.debug()
        self.opt.step()
        self.num_steps += 1
        self.loss = loss.item()

    def debug(self,): 
        pi, mu, std = self.model.pi(), self.model.mu(), self.model.std()
        #print("pi",pi.data)
        #print("std", std.data)
        #print("mu", mu.data)
        #print("xs",self.xs.data)       
        #print("mu grad", self.model._mu.grad)
        if self.model._mu.grad is not None:
            if torch.any(torch.isnan(self.model._mu.grad)):
                assert False, 'mu grad is nan'

    '''
    plots the 1D data xs at the line ys=1
    then displays the model density on the same plot
    '''
    def plot(self, xs ,ys):
        self.plot_width = 3
        self.ax.clear()
        self.ax.scatter(xs.numpy(), ys.numpy(), color='blue')
        grid = torch.linspace(-5, 5, 1000)
        density = self.model(grid).exp()
        self.ax.plot(grid.numpy(), density.detach().numpy(), label='density', color='purple')
        #if self.num_steps >= self.total_steps:
        #    plt.title("done")
        #else:
        plt.title(f"Step:{self.num_steps}, loss: {self.loss}")
        self.ax.set_xlim(-self.plot_width, self.plot_width)
        #self.ax.set_ylim(0, 1.0)
        plt.legend()

    def training_loop(self, step):

        if self.init_means:
            self.model._mu.data = self.xs.clone()
        for rep in range(self.steps_per_frame):
            self.step(self.xs, self.ys)
        self.plot(self.xs, self.ys)

    # total steps is 1000 frames by 100 grad steps per frame 
    def main(self):
        self.fig, self.ax = plt.subplots()
        #self.ax2.set_yscale('log')
        self.steps_per_frame = 100
        self.frames = 100
        self.total_steps = self.steps_per_frame * self.frames
        ani = animation.FuncAnimation(self.fig, self.training_loop, repeat=False, frames=self.frames)
        plt.show()
if __name__ == '__main__':
    #trainer = Trainer(K = 2, mode='esm')
    both = 2
    #trainer = Trainer(N = both, K = both, mode='ism', std_lowerbound = .01, init_means = True, freeze_means = True, freeze_pi = True)
    #trainer = Trainer(N = both, K = both, mode='esm', std_lowerbound = .01, init_means = False, freeze_means = False, freeze_pi = False)
    trainer = Trainer(N = both, K = both, mode='mle', std_lowerbound = .01, init_means = False, freeze_means = False, freeze_pi = False)
    trainer.main()

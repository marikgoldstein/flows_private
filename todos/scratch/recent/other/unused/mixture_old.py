import torch
import numpy as np
import torch.distributions as D
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.animation as animation

Normal = D.Normal
Cat = D.Categorical
Mix = D.MixtureSameFamily

class RealData:
    def __init__(self, N):
        
        self.N = N
        self.mu_q = torch.zeros(1)
        self.sig_q = torch.ones(1) * 10.0
        self.q = Normal(loc = self.mu_q, scale = self.sig_q)

    def get_batch(self,):         
        xs  = self.q.sample(sample_shape=(self.N,)).squeeze(-1)
        ys = torch.ones_like(xs)
        return xs, ys

    def compute_data_score(self, xs):
        return -(xs - self.mu_q) / self.sig_q.pow(2)

class GMM(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.logit_pi = nn.Parameter(torch.ones(self.K) / self.K, requires_grad=True)
        self._mu = nn.Parameter(torch.rand(self.K), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(self.K), requires_grad=True)

    @property
    def pi(self,):
        return nn.Softmax(dim=-1)(self.logit_pi)
    
    @propety
    def mu(self,):
        return self._mu
    
    @propety
    def std(self,):
        return nn.Softplus()(self.scale)

    def make_gmm(self): 
        return Mix(Cat(self.pi), Normal(self.mu, self.std))

    def forward(self, xs):
        gmm = self.make_gmm()
        return gmm.log_prob(xs)

    def score(self, xs,):

        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        out = {}

        pi, mu, std = self.pi, self.mu, self.std
        if self.K == 1:
            for k in range(self.K):
                muk, pik, stdk, vark = mu[k], pi[k], std[k], std[k].pow(2)
                manual  = -(xs - muk) / vark
        else:
            manual = None
        out['manual']=manual

        # unstable
        numer = 0.0
        root2pi = torch.sqrt(torch.tensor([2 * np.pi]))
        for k in range(self.K):
            muk, pik, stdk, vark, sig3k = mu[k], pi[k], std[k], std[k].pow(2), std[k].pow(3)
            numer1 = pik
            numer2 = (1/ (root2pi * stdk))
            numer3 = ((xs - muk) / vark )
            numer4 = torch.exp(-.5 * (xs - muk).pow(2) / vark)
            numer = numer1 * numer2 * numer3 * numer4
        denom = self.make_gmm().log_prob(xs)
        unstable = -numer / denom
        out['unstable'] = unstable
        
        numer =0
        denom = 0
        for k in range(self.K):
            
            muk, pik, stdk, vark = mu[k], pi[k], std[k], std[k].pow(2)

            numer += -(xs - muk)/vark
            
            for j in range(self.K):

                denom += 1
                if j != k:
                    
                    muj, pij, stdj, varj = mu[j], pi[j], std[j], std[j].pow(2)
                    one = pij/pik
                    two = torch.sqrt(vark/varj)
                    three = torch.exp(
                        (-1/2)*(xs-muj).pow(2)/varj
                        +
                        (1/2)*(xs-muk).pow(2)/vark
                    )
                    denom += (one*two*three)
        stable = numer / denom
        out['stable'] = stable

        for key in out:
            if out[key] is not None:
                assert out[key].shape==(bsz,)
        return out



class Trainer:

    def __init__(self, K, mode='mle'):
        
        assert mode in ['mle','esm', 'ism']
        self.mode = mode
        # setting N = K
        self.K = K
        self.data = RealData(N=self.K)
        self.model = GMM(K=self.K)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2)
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

    def step_esm(self, xs, ys):
        #with torch.autograd.detect_anomaly():
        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        ds = self.data.compute_data_score(xs)
        out = self.score(xs)
        stable_ms = out['stable']
        unstable_ms = out['unstable']
        manual_ms = out['manual']
        ms = stable_ms
        assert ds.shape == (bsz,)
        assert ms.shape == (bsz,)
        loss_per_datapoint = .5 * (ds-ms).pow(2)
        loss = loss_per_datapoint.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.num_steps +=1

    def step_ism(self, xs, ys):
        #assert False
        #with torch.autograd.detect_anomaly():
        bsz = xs.shape[0]
        assert xs.shape == (bsz,)
        ms = self.model.score(xs)['manual']
        assert ms.shape == (bsz,)
        norm_term = ms.pow(2)
        variance = nn.Softplus()(self.model.scale).pow(2)
        trace_term = - 1 / variance
        loss_per_datapoint = .5 * trace_term + norm_term
        assert loss_per_datapoint.shape == (bsz,)
        loss = loss_per_datapoint.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.num_steps += 1

    '''
    plots the 1D data xs at the line ys=1
    then displays the model density on the same plot
    '''
    def plot(self, xs ,ys):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.scatter(xs.numpy(), ys.numpy(), label='true', color='blue')
        grid = torch.linspace(-20, 20, 1000)
        density = self.model(grid).exp()
        self.ax2.plot(grid.numpy(), density.detach().numpy(), label='density', color='orange')
        if self.num_steps >= self.total_steps:
            plt.title("done")
        else:
            plt.title(f"Step:{self.num_steps}")
        self.ax1.set_xlim(-20, 20)
        self.ax2.set_xlim(-20, 20)
        plt.legend()

    def training_loop(self, step):
        
        for rep in range(self.steps_per_frame):
            self.step(self.xs, self.ys)
                
        self.plot(self.xs, self.ys)

    # total steps is 1000 frames by 100 grad steps per frame 
    def main(self):
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx() # plot data and density on same plot
        self.steps_per_frame = 100
        self.frames = 1000
        self.total_steps = self.steps_per_frame * self.frames
        ani = animation.FuncAnimation(self.fig, self.training_loop, repeat=False, frames=self.frames)
        plt.show()

if __name__ == '__main__':
    #trainer = Trainer(K = 2, mode='esm')
    trainer = Trainer(K = 1, mode='ism')
    trainer.main()

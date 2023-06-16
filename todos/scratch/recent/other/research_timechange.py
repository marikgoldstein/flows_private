import torch
import torch.nn as nn
import numpy as np
Uniform = torch.distributions.Uniform 
Normal = torch.distributions.Normal

ones_like = torch.ones_like


class TimeSampler(nn.Module):

    def __init__(self, mint, time_change, optimize, maxt):
        super(TimeSampler, self).__init__()
        self.mint = mint
        self.maxt = maxt
        self.time_change = time_change

        self.t_width = self.maxt - self.mint
        self.t_weight = 1.0
        print("tmin tmax width", self.mint, self.maxt, self.t_width)
        self.optimize = optimize

        #self.z = self.zeros(1)
        #self.o = self.ones(1)

        if self.optimize:
            self.mu = nn.Parameter(self.z.clone(), requires_grad=True)
            self.log_std = nn.Parameter(self.o.clone(), requires_grad=True)
        else:
            self.mu = 0.0 #torch.zeros(1) #self.z
            self.log_std = 1.0 #torch.ones(1self.o
        
        self.U = Uniform(low=self.mint, high=self.maxt)


    def forward(self, bsz):
        return self.sample(bsz)

    #def zeros(self, bsz):
    #    return torch.zeros(bsz).to(self.device)
 
    #def ones(self, bsz):
    #    return torch.ones(bsz).to(self.device)

    def sample(self, bsz):
        
        if self.time_change:
            t, w = self.sample_changed(bsz)
        else:
            t, w = self.sample_U(bsz)
        
        return t, w

    def sample_U(self, bsz):
        time = self.U.sample(sample_shape = (bsz,)).squeeze(-1)
        width = self.t_width
        weight = self.t_weight * torch.ones(bsz,) #self.ones(bsz,)
        return time, width * weight
 
    def sample_changed(self, bsz):
        
        beta_inv_low = self.beta_inv(self.o * self.mint)
        beta_inv_high = self.beta_inv(self.o * self.maxt)
        low = self.ones(bsz) * beta_inv_low
        high = self.ones(bsz,) * beta_inv_high
        s = self.batched_unif_sample(low, high)
        time = self.beta(s)
        width = beta_inv_high - beta_inv_low 
        weight = self.beta_prime(s)
        return time, width * weight         

    def make_N(self,):
        mu = self.mu
        #std = nn.Softplus()(self.log_std)
        std = self.log_std.exp()
        return Normal(mu, std)

    def pdf(self, x):
        N = self.make_N()
        return N.log_prob(x).exp()

    def cdf(self, x):
        N = self.make_N()
        return N.cdf(x)

    def icdf(self, x):
        N = self.make_N()
        return N.icdf(x)

    def shrink(self, x):
        return .5 * x + .25

    def stretch(self, x):
        return 2 * (x-.25)

    def beta(self, s):
        return self.stretch(self.cdf(s))

    def beta_inv(self, t):
        return self.icdf(self.shrink(t))

    def beta_prime(self, s):
        return 2 * self.pdf(s)

    def batched_unif_sample(self, low, high):
        return Uniform(low, high).sample()


if __name__ == '__main__':

    OPTIMIZE_TIMECHANGE = False

    def f(t):
        return t.pow(2)

    def f(t):
        return t.sin()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    time_sampler = TimeSampler(mint=1e-5, time_change=False, optimize = False, device = device)
    bsz = 1024 # 256 # 1_000_000
    t, width, weight = time_sampler.sample(bsz)
    est = f(t) * width * weight 
    print("original unif(0,1) expectation sampler")
    print('mu',est.mean())
    print('std',est.std())
    print("------")
    print("time changes:")
    print("------")
    time_sampler = TimeSampler(mint=1e-5, time_change=True, optimize = OPTIMIZE_TIMECHANGE, device = device)
    for n,p in time_sampler.named_parameters():
        print(n, p )
    
    if OPTIMIZE_TIMECHANGE:
        opt = torch.optim.Adam(time_sampler.parameters(), lr=1e-3)
    iters = 1_000_000
    for i in range(iters):
        t, width, weight = time_sampler.sample(bsz)
        est = f(t) * width * weight 
        if i % 100 == 0:
            print('mu',round(est.mean().item(),4))
            print('std',round(est.std().item(),4))
            print("normal mean is ", time_sampler.mu.data.item())
        loss = est.pow(2).mean() 
       
        if OPTIMIZE_TIMECHANGE:
            opt.zero_grad()
            loss.backward()
            opt.step()


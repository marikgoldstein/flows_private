import torch
import numpy as np
import matplotlib.pyplot as plt

class NoiseSchedule:

    def __init__(self, beta_obj):
        self.beta_obj = beta_obj
        self.b = self.beta_obj.b
        self.B = self.beta_obj.B

    def m(self, t):
        return torch.exp(-0.5 * self.B(t))

    def m2(self, t):
        return torch.exp(-self.B(t))

    def v(self, t):
        return 1 - self.m2(t)

    def snr(self, t):
        return self.m2(t) / self.v(t)

    def log_snr(self, t):
        return self.snr(t).log()

class CosSchedule:

    def __init__(self,):
        self.s = 0.008
        #mean coef equals f(t)/f(0) where f(t) = cos^2 of (t/T+s)/(1+s) * pi/2

    def f(self, t):
        const1 = (np.pi/2)
        const2 = (1+self.s)
        const = const1 / const2
        inside = const * (t+self.s)
        return inside.cos().pow(2)

    def neg_log_m2(self, t):
        z = torch.zeros_like(t)
        return self.f(z).log() - self.f(t).log() 

    def b(self, t):
        const = (np.pi/2) / (1+self.s)
        inside = const * (t+self.s)
        return (2.0  / self.f(t)) * const * inside.sin() * inside.cos()

    def B(self, t):
        return self.neg_log_m2(t)

class LinearMeanCoef:

    def __init__(self,):
        self.hi = 'hi'

    def b(self, t):
        return t / (1-t)

    def B(self, t):
        return -(1-t).log()


class ConstBeta:

    def __init__(self, bconst):    
        self.bconst = bconst

    def b(self, t):
        return self.bconst 

    def B(self, t):
        return self.bconst * t


class LinearBeta:

    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b0

    def b(self, t):
        return self.b0 + (self.b1-self.b0) * t

    def B(self, t):
        return self.b0*t + (self.b1-self.b0)*0.5*t.pow(2)

    def m(self, t):
        return torch.exp(-0.5 * self.B(t))
 
    def m2(self, t):                                                                                                   
        return torch.exp(-self.B(t))
 
    def v(self, t):
        return 1 - self.m2(t)
 
    def snr(self, t):
        return self.m2(t) / self.v(t)
 
    def log_snr(self, t):
        return self.snr(t).log()

linear = NoiseSchedule(LinearBeta(.1, 20))
t = torch.tensor([0.5], requires_grad=True)
snr_t = linear.log_snr(t)
snr_t.backward()
print(t.grad)

t = torch.tensor([0.5])
g2 = linear.b(t)
sig2 = linear.v(t)
print(g2/sig2)




'''
linear = NoiseSchedule(LinearBeta(.1, 20))
simple = NoiseSchedule(LinearMeanCoef())
cos = NoiseSchedule(CosSchedule())
const = NoiseSchedule(ConstBeta(10.0))
delta = 0.1
ts = torch.linspace(delta, 1 - 1e-5, 100)
ts_numpy = ts.numpy()

linear_log_snrs = [linear.log_snr(t).item() for t in ts]
simple_log_snrs = [simple.log_snr(t).item() for t in ts]
cos_log_snrs = [cos.log_snr(t).item() for t in ts]
const_log_snrs = [const.log_snr(t).item() for t in ts]

plt.plot(ts_numpy, linear_log_snrs, label='linear')
plt.plot(ts_numpy, simple_log_snrs, label='simple')
plt.plot(ts_numpy, cos_log_snrs, label='cos')
plt.plot(ts_numpy, const_log_snrs, label='const')
plt.legend()
plt.show()
'''
    




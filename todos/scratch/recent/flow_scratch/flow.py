import torch



# requires
#config.T_min
#config.T_max
#config.n_sample_steps
#config.num_classes
#config.C
#config.H
#config.W

class Flow:

    def __init__(self, config):
        self.config = config

    def triple_sum(self, x):
        return x.sum(-1).sum(-1).sum(-1)
    
    def sqnorm_image(self, x):
        return self.triple_sum(x.pow(2))

    def wide(self, x):
        return x[:, None, None, None]

    def path_fn(self, z1, z0, t):
    
        # mean coef = t, std = 1 - t
        # d/dt(tz1 + (1-t)z0) = d/dt(tz1 + z0 - tz0) = z1 - z0
        if self.config.path_type == 'straight':
            zt = self.wide(t) * z1 + self.wide(1-t) * z0
            target = z1 - z0

        # mean coef squared = t
        # mean coef = root t
        # variance = 1 - t
        # std = root(1-t)
        # d/dt (root(t) z1 + root(1-t)z0)
        # 1 / (2 root(t)) * z1  + d/dt root(1-t)z0
        # 1 / (2root(t)) * z1 - 1 / (2root(1-t))z0
        elif self.config.path_type == 'vp':
            zt = self.wide(t.sqrt()) * z1 + self.wide((1-t).sqrt()) * z0
            target = self.wide(1 / (2 * t.sqrt())) * z1 -  self.wide(1 / (2 * (1-t).sqrt())) * z0
        else:
            assert False

        return zt, target

    def loss_fn(self, z1, y, model, device, loss_type=None):
        bsz = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(bsz,).to(device)
        t = torch.clamp(t, min = self.config.T_min, max = self.config.T_max)
        zt, target = self.path_fn(z1, z0, t)
        vtheta = model(zt, t, y)
        loss = self.sqnorm_image(vtheta - target)
        return {'loss':loss,'zt':zt,'z0':z0,'z1':z1,'t':t,'y':y}
    
    def sample(self, n_samples, model, device):
        steps = self.config.n_sample_steps
        ts = torch.linspace(self.config.T_min, self.config.T_max, steps).to(device)
        y = torch.randint(0, self.config.num_classes, (n_samples,)).to(device)
        zt = torch.randn(n_samples, self.config.C_dgm, self.config.H_dgm, self.config.W_dgm).to(device)
        for t in ts:
            zt += (1/steps) * model(zt, torch.ones_like(y) * t, y)
        return zt

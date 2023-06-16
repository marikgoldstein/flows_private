import wandb
import torch
import numpy as np
import torch
import math
from dit import get_dit
from encoder import Encoder
from config import ExperimentConfig
from integrators import PFlowIntegrator, SDEIntegrator
from boilerplate import make_model_and_ema, update_ema, sample_dir_this_step, trim_grads
from image_processing import ImageProcessing                                                                                                                                                  
# bsz: batch size
#return self.seta if self.config.seta_param == 's' else SFromEta(self.seta, self.I.gamma)
#return self.bv if self.config.bv_param == 'b' else BFromVS(self.bv, self.get_s(), self.I.gg_dot)

class ContinuousTimeDGM:

    def __init__(self, config):

        self.config = config
        self.pflow = PFlowIntegrator(self.config)
        self.sflow = SDEIntegrator(self.config)    
        if config.dgm_type == 'flow':
            self.dgm = Flow(self.config)

        else:
            self.dgm = Interpolant(self.config)

        self.setup_models()
        self.opt = torch.optim.AdamW(self.get_params(), lr=config.base_lr, weight_decay=config.wd)
        self.encoder = Encoder(self.config, self.config.device)
         # for processing model sample for fid/wandb/etc
        self.image_processing = ImageProcessing(self.config)

    def get_ckpt_dict(self,):
        # TODO update for interpolants     
        checkpoint = {
            "model": self.model.state_dict(),
            "ema": self.model_ema.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.config,
        }
        return checkpoint

    def trim_grads(self, train_steps, log_norm = False):
        D = {} 
        conf = self.config
        dgm_type = conf.dgm_type
        if dgm_type == 'flow':
            D['grad_norm'] = trim_grads(conf, self.model)
        else:
            D['bv_grad_norm'] = trim_grads(conf, self.bv)
            if conf.do_seta:  
                D['seta_grad_norm'] = trim_grads(conf, self.seta)
       
        # does this slow down training? logging to wandb every grad step?
        if log_norm and conf.use_wandb: 
            wandb.log(D, step = train_steps)  

    def step_optimizer(self, train_steps, log_norm = False):    
        conf = self.config       
        stepped = False                                 
        if train_steps % conf.grad_accum == 0:
            self.trim_grads(train_steps, log_norm = log_norm)
            self.opt.step()          
            self.opt.zero_grad()     
            stepped = True          
        return stepped

    def setup_models(self,):
             
        conf = self.config
        dgm_type = conf.dgm_type
        self.models = {} 
        if dgm_type == 'flow':
            self.model, self.model_ema = make_model_and_ema(conf)
            self.models['model'] = self.model
            self.models['model_ema'] = self.model_ema
            #self.bv, self.bv_ema = None, None
            #self.seta, self.seta_ema = None, None
        else:
            self.bv, self.bv_ema = make_model_and_ema(conf)
            self.models['bv'] = self.bv
            self.models['bv_ema'] = self.bv_ema
            if conf.do_seta:
                self.seta, self.seta_ema = make_model_and_ema(conf)
                self.models['seta'] = self.seta
                self.models['seta_ema'] = self.seta_ema
            #else:
            #    self.seta, self.seta_ema = None, None
            #self.model, self.model_ema = None, NOne

        self.update_emas(decay = 0.) # Ensure EMA is initialized with synced weights
        print("Done setting up models")

    def get_params(self,):
        dgm_type = self.config.dgm_type
        if dgm_type == 'flow':
            params = self.model.parameters()
             
        else:
            params = list(self.bv.parameters())
            if self.do_seta:
                params += list(self.seta.parameters())
        return params

    def maybe_update_emas(self, train_steps): 
        conf = self.config
        cond1 = train_steps > conf.update_ema_after
        cond2 = train_steps % conf.update_ema_every == 0
        if cond1 and cond2:
            self.update_emas(conf.ema_decay)

    def update_emas(self, decay):
        dgm_type = self.config.dgm_type
        if dgm_type == 'flow':
            update_ema(self.model_ema, self.model, decay)
        else:
            update_ema(self.bv_ema, self.bv, decay)
            if self.config.do_seta:
                update_ema(self.seta_ema, self.seta, decay)
 
    def loss_fn(self, z, y):

        if self.config.dgm_type == 'flow':
            return self.dgm.loss_fn(z, y, self.model)
        else:
            return self.dgm.loss_fn(z, y, self.bv, self.seta)

    def sample_ode(self, model, model_ema, use_ema, z0, y, cheap):
        field = model if not use_ema else model_ema
        field.eval()                    
        samples, _ = self.pflow(field, z0, y, cheap=False)
        zT = samples[-1] # samples is whole trajectory, take last
        if not use_ema:
            field.train()               
        return zT 

    def sample_sde(bv, bv_ema, seta, seta_ema, z0, y, cheap):
        b = bv_ema if use_ema else bv                                
        s = seta_ema if use_ema else seta                            
        b.eval()                                                     
        s.eval()                                                     
        zT = self.sflow(b, s, z0, y, cheap = cheap)
        if not use_ema:                                              
            b.train()                                                
            s.train()                                                
        return zT

    def sample_prior(self, N):
        conf = self.config
        device = conf.device
        C, H, W = conf.C_flow, conf.H_flow, conf.W_flow
        y = torch.randint(0, conf.num_classes, (N,)).to(device)
        z0 = torch.randn(N, C, H, W).to(device)
        return z0, y

    @torch.no_grad()                  
    def sample(self, use_ema = False, cheap = False, train_steps = -1):
                 
        conf = self.config
        dgm_type = conf.dgm_type
        save_fn = self.image_processing.save_images_for_fid
        wandb_fn = self.image_processing.process_images_for_wandb
        dec_fn = self.encoder.decode
        device = conf.device
        dir_fn = sample_dir_this_step 
        D = {}   
        ema_str = '_ema' if use_ema else '_nonema'
        print(f"sampling from model. Use ema: {use_ema}")
        
        z0, y = sample_prior(conf.num_sampled_images)
        
        if dgm_type == 'flow':
            model, model_ema = self.models['model'], self.models['model_ema']
            zT = self.sample_ode(model, model_ema, use_ema, z0, y, cheap)
            x = dec_fn(zT)
            _dir = dir_fn(conf.sample_dir, 'flow_ode', use_ema, train_steps)
            save_fn(x.clone(), _dir, use_ema)
            D['flow_ode_samples' + ema_str] = wandb_fn(x)
                 
        if dgm_type == 'interpolant':
            bv, bv_ema = self.models['bv'], self.models['bv_ema']
            zT = self.sample_ode(bv, bv_ema, use_ema, z0, y, cheap)
            x = dec_fn(zT)
            _dir = dir_fn(conf.sample_dir, 'interpolant_ode', use_ema, train_steps)
            save_fn(x.clone(), _dir, use_ema)
            D['interpolant_ode_samples' + ema_str] = wandb_fn(x)
                 
        if dgm_type == 'interpolant' and conf.do_seta:
            bv, bv_ema = self.models['bv'], self.models['bv_ema']
            seta, seta_ema = self.models['seta'], self.models['seta_ema']
            zT = self.sample_sde(bv, bv_ema, seta, seta_ema, z0, y, cheap)
            x = dec_fn(zT)
            _dir = dir_fn(conf.sample_dir, 'interpolant_sde', use_ema, train_steps)
            save_fn(x.clone(), _dir, use_ema)
            D['interpolant_sde_samples' + ema_str] = wandb_fn(x)
                 
        return D    

class Flow:

    def __init__(self, config):
        self.config = config

    def triple_sum(self, x):
        return x.sum(-1).sum(-1).sum(-1)
    
    def sqnorm_image(self, x):
        return self.triple_sum(x.pow(2))

    # (bsz,) to (bsz, 1 ,1 ,1)
    def wide(self, x):
        return x[:, None, None, None]

    def path_fn(self, z1, z0, t):
        zt = self.wide(t) * z1 + self.wide(1-t) * z0
        target = z1 - z0
        return zt, target

    def get_times(self, bsz):
        t = torch.rand(bsz,).to(self.config.device)
        t = torch.clamp(t, min=self.config.T_min, max = self.config.T_max)
        return t

    def loss_fn(self, z1, y, model):
        bsz = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = self.get_times(bsz,)
        zt, target = self.path_fn(z1, z0, t)
        vtheta = model(zt, t, y)
        loss = self.sqnorm_image(vtheta - target)
        #return {'loss':loss.mean(),'zt':zt,'z0':z0,'z1':z1,'t':t,'y':y} 
        return {'loss': loss.mean()}

# changes from michael's repo to get simple version running:
# - only one_sided b and one_sided s
# - only brownian gamma
# - no antithetic

class SFromEta(torch.nn.Module):
    """Class for turning a noise model into a score model."""
    def __init__(self, eta, gamma):
        super(SFromEta, self).__init__()
        self.eta = eta
        self.gamma = gamma
        
    def forward(self, x, t):
        val = (self.eta(x,t) / self.gamma(t))
        return val


class BFromVS(torch.nn.Module):
    """Class for turning a velocity model and a score model into a drift model."""
    def __init__(self, v, s, gg_dot):
        super(BFromVS, self).__init__()
        self.v = v
        self.s = s
        self.gg_dot = gg_dot
        
    def forward(self, x, t):
        return self.v(x, t) - self.gg_dot(t)*self.s(x, t)
    
class Interpolant(torch.nn.Module):

    def __init__(self, path, gamma_type):
        super(Interpolant, self).__init__()
   
        assert path == 'one_sided', 'temporarily only supporting one_sided path'
        assert gamma_type == 'brownian', 'temporarily only supporting brownian gamma'
        self.make_gamma(gamma_type=gamma_type)
        self.path = path
        self.make_It(path, self.gamma, self.gamma_dot)
        self.bv_loss = self.loss_per_sample_one_sided
        self.seta_loss = self.loss_per_sample_one_sided_s

    def calc_xt(self, t, x0, x1):
        if self.path=='one_sided' or self.path == 'mirror' or self.path=='one_sided_bs':
            return self.It(t, x0, x1)
        else:
            z = torch.randn(x0.shape).to(t)
            return self.It(t, x0, x1) + self.wide(self.gamma(t))*z, z

    def forward(self, x):
        raise NotImplementedError("No forward pass for interpolant.")

    def wide(self, x):
        return x[:, None, None, None]

    def image_sum(self, x):
        return x.sum(-1).sum(-1).sum(-1)

    def image_sq_norm(self, x):
        return self.image_sum(x.pow(2))

    def make_gamma(self, gamma_type = 'brownian', a = None):
        self.gamma = lambda t: torch.sqrt(t*(1-t))
        self.gamma_dot = lambda t: (1/(2*torch.sqrt(t*(1-t)))) * (1 -2*t)
        self.gg_dot = lambda t: (1/2)*(1-2*t)

    def make_It(self, path='linear', gamma = None, gamma_dot = None):
        self.It   = lambda t, x0, x1: (1 - self.wide(t))*x0 + self.wide(t)*x1
        self.dtIt = lambda _, x0, x1: x1 - x0

    def get_times(self, bsz):
        t = torch.rand(bsz,).to(self.config.device)
        t = torch.clamp(t, min=self.config.T_min, max = self.config.T_max)
        return t

    def loss_per_sample_one_sided(self, b, x0, x1, t, y):
        xt  = self.calc_xt(t, x0, x1)      
        return 0.5 * self.image_sq_norm(                                                                                                                                                              
            b(xt, t, y) - self.dtIt(t, x0, x1) 
        )  

    def loss_per_sample_one_sided_s(self, s, x0, x1, t, y):
        xt = self.calc_xt(t, x0, x1)
        alpha = self.wide(torch.sqrt(1 - t))
        target = - (1/alpha) * x0
        return 0.5 * self.image_sq_norm(
            s(xt ,t, y) - target
        )       

    def loss_fn(self, z, y, bv, seta):

        z1, z0 = z, torch.randn_like(z)

        conf = self.config

        bsz = z1.shape[0]
        t = self.get_times(bsz,)

        loss_bv, loss_seta = torch.tensor([0.0]), torch.tensor([0.0])              
                                                        
        if conf.do_bv:                              
            loss_bv = self.bv_loss(bv, z0, z1, t, y)
            assert loss_bv.shape == (bsz, )         
                                                    
        if conf.do_seta:                            
            loss_seta = self.seta_loss(seta, z0, z1, t, y)
            assert loss_seta.shape == (bsz,)        

        total_loss = loss_bv + loss_seta

        return {
            'loss' : total_loss.mean(),
            'bv_loss': loss_bv.mean(),
            'seta_loss': seta_loss.mean(),
        }

        return loss_dict

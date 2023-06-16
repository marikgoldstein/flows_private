import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torch.distributions as D
from models import TinyNet
from data import DataModule
from cv import ControlModule #,control_scalar, #ControlRegression
from torch.func import functional_call, vmap, grad, vjp
import wandb
from sampling import Sampler
from collections import defaultdict

def set_device():
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def sqnorm(x):
    return x.pow(2).sum(-1)

def loss_fn_stateless(vt, ut, weight):
    return (sqnorm(vt - ut) * weight).mean()

def loss_fn_stateless_control(vt, ut, c, weight):
    return (sqnorm(vt - ut - c) * weight).mean()

def compute_loss_stateless_model(params, buffers, model, xt, t, ut, weight):
    xt, t, ut = xt[None,...], t[None,...], ut[None,...]
    vt = functional_call(model, (params, buffers), (xt, t))
    return loss_fn_stateless(vt, ut, weight)

def compute_loss_stateless_model_control(params, buffers, model, xt, t, ut, c, weight):
    xt, t, ut, c = xt[None,...], t[None,...], ut[None,...], c[None,...]
    vt = functional_call(model, (params, buffers), (xt, t))
    return loss_fn_stateless_control(vt, ut, c, weight)


class Trainer:

    def __init__(self, use_functorch, use_wandb, train_both):
    
        self.device = set_device()

        # wandb
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(
                project='var_reduce',
                entity='marikgoldstein',
            )
     
        # data
        self.train_both = train_both
        self.bsz = 256
        self.d = 2
        self.datamodule = DataModule(self.bsz, self.d, device=self.device)

        self.model1 = TinyNet(d=self.d)
        self.model1.to(self.device)
        self.optimizer1 = Adam(self.model1.parameters(), lr=1e-2)
    
        if self.train_both:
            self.model2 = TinyNet(d=self.d)
            self.model2.to(self.device)
            self.model2.load_state_dict(copy.deepcopy(self.model1.state_dict()))
            self.optimizer2 = Adam(self.model2.parameters(), lr=1e-2)
 
        self.sampler = Sampler(trainer = self)
        self.cm = ControlModule(datamodule = self.datamodule)
          
        self.use_functorch = use_functorch
        if use_functorch:
            self.setup_functorch()
        self.step_fn = self.fstep_fn if self.use_functorch else self.usual_step_fn
       
        # training
        self.current_step = 0
        self.total_steps = 1000
        self.monitor_every_n_steps = 1
        self.sampling_steps = 1000
        self.sample_every_n_steps = 1

    def get_stuff(self, which_model):
        if which_model == 'model1':
            model, params, buffers = self.model1, self.params1, self.buffers1
        elif which_model == 'model2':
            model, params, buffers = self.model2, self.params2, self.buffers2
        else:
            assert False
        return model, params, buffers


    def setup_functorch(self,):
        self.params1 = {k: v.detach() for k, v in self.model1.named_parameters()}
        self.buffers1 = {k: v.detach() for k, v in self.model1.named_buffers()}

        if self.train_both:
            self.params2 = {k: v.detach() for k, v in self.model2.named_parameters()}
            self.buffers2 = {k: v.detach() for k, v in self.model2.named_buffers()}
    
        self.loss_grad_fn = grad(compute_loss_stateless_model)
        self.batched_loss_grad_fn = vmap(self.loss_grad_fn, in_dims=(None, None, None, 0, 0, 0, 0))
  
        self.loss_grad_fn_control = grad(compute_loss_stateless_model_control)
        self.batched_loss_grad_fn_control = vmap(self.loss_grad_fn_control, in_dims=(None, None, None, 0, 0, 0, 0, 0))


    def get_loss_functorch(self, batch, which_model):
        x0, x1, xt, xt_mean, t, ut, c, weight = batch
        model, params, buffers = self.get_stuff(which_model)
        vt = functional_call(model, (params, buffers), (xt, t))
        return loss_fn_stateless(vt, ut, weight)
    
    def get_grads_functorch(self, batch, which_model):
        x0, x1, xt, xt_mean, t, ut, c, weight = batch
        model, params, buffers = self.get_stuff(which_model)
        return self.batched_loss_grad_fn(params, buffers, model, xt, t, ut, weight)

    def get_loss_functorch_control(self, batch, which_model):
        x0, x1, xt, xt_mean, t, ut, c, weight = batch
        model, params, buffers = self.get_stuff(which_model)
        vt = functional_call(model, (params, buffers), (xt, t))
        return loss_fn_stateless_control(vt, ut, c, weight)
    
    def get_grads_functorch_control(self, batch, which_model):
        x0, x1, xt, xt_mean, t, ut, c, weight = batch
        model, params, buffers = self.get_stuff(which_model)
        return self.batched_loss_grad_fn_control(params, buffers, model, xt, t, ut, c, weight)

    def copy_grads_into_model(self, per_sample_grads, which_model):

        if which_model == 'model1':
            m = self.model1
        else:
            m = self.model2

        for n, p in m.named_parameters():
            if n in per_sample_grads:
                p.grad = per_sample_grads[n].mean(0)

    def get_var_dict(self, per_sample_grads, which_model):

        D = {}
        for param_name in per_sample_grads:
            grad = per_sample_grads[param_name]
            if 'weight' in param_name:
                m = grad.mean(0)[0,0]
            else:
                m = grad.mean(0).mean()
            v = grad.var(0).mean()
            D['mean_' + param_name + '_' + which_model] = m
            D['var_' + param_name + "_" + which_model] = v
        return D  

    def fstep_fn(self, batch1, batch2, compute_loss):

        grad1_D, grad2_D = None, None
        loss1, loss2 = None, None
        model1_batch = batch1
        model2_batch = batch1 # batch2

        grads1 = self.get_grads_functorch(model1_batch, 'model1')
        grad1_D = self.get_var_dict(grads1, 'model1')
        self.copy_grads_into_model(grads1, 'model1')
        if self.train_both:
            grads2 = self.get_grads_functorch_control(model2_batch, 'model2')
            grad2_D = self.get_var_dict(grads2, 'model2')
            #grads2_cv = self.cm.control_grads(model2_batch, grads2)
            #grad2_D = self.get_var_dict(grads2_cv, 'model2')
            #self.copy_grads_into_model(grads2_cv, 'model2')
            self.copy_grads_into_model(grads1, 'model2')
        if compute_loss:
            loss1 = self.get_loss_functorch(model1_batch, 'model1').item()
            if self.train_both:
                # NOT CONTROL!
                loss2 = self.get_loss_functorch(model2_batch, 'model2').item()
                loss2_control = self.get_loss_functorch_control(model2_batch, 'model2').item()
                #loss2_reg = self.get_loss_functorch(model1_batch, 'model2').item()
                #loss2_change = self.get_loss_functorch(model2_batch, 'model2').item()
            else:
                #loss2_reg = loss1
                #loss2_change = loss1
                pass

        return loss1, loss2, loss2_control, grad1_D, grad2_D # loss2_change, grad1_D, grad2_D

    def usual_step_fn(self, batch, compute_loss):
        assert False, "1 vs 2 batches"
        x0, x1, xt, xt_mean, t, ut, c, weight = batch
        vt = self.model(xt, t)
        loss = loss_fn_stateless(vt, ut, weight)
        loss.backward()
        return loss.item()

    def check_means(self):

        print("checking means")
        D = {'model1':defaultdict(list), 'model2':defaultdict(list)}

        B = 10

        for b in range(B):

            model1_batch, _ = self.datamodule.get_batch()
            model2_batch = model1_batch 
            
            grads1 = self.get_grads_functorch(model1_batch, 'model1')

            grads2 = self.get_grads_functorch(model2_batch, 'model2')
            grads2_cv = self.cm.control_grads(model2_batch, grads2)

            for g in grads1:
                
                D['model1'][g].append(grads1[g].mean(0))
                D['model2'][g].append(grads2_cv[g].mean(0))
        
        import pdb
        pdb.set_trace()
    
    def train_loop(self,):
        
        #self.check_means()
        
        for step in range(self.total_steps):
            monitor = step % self.monitor_every_n_steps == 0
            self.optimizer1.zero_grad()

            if self.train_both:
                self.optimizer2.zero_grad()
            
            batch1, batch2_dont_use = self.datamodule.get_batch()
            
            #loss1, loss2_reg, loss2_change, grad1_D, grad2_D = self.step_fn(batch1, batch2, compute_loss = monitor)
            #loss1, loss2, grad1_D, grad2_D = self.step_fn(batch1, batch2_dont_use, compute_loss = monitor)

            loss1, loss2, loss2_control, grad1_D, grad2_D = self.step_fn(batch1, batch2_dont_use, compute_loss = monitor)
            
            self.optimizer1.step()
            
            if self.train_both:
                self.optimizer2.step()
            self.current_step += 1
            
            if monitor:
               
                #print("loss 1: {} loss 2 (reg): {} loss 2 (change): {}".format(round(loss1,4), round(loss2_reg, 4), round(loss2_change, 4)))
                #print("loss 1: {} loss 2: {} ".format(round(loss1,4), round(loss2, 4)))
                print("loss 1 : {} loss 2: {} loss 2 (whats being optimized): {}".format(round(loss1,4), round(loss2, 4), round(loss2_control, 4)))
                if self.use_wandb:
                    #wandb.log({'loss1': loss1, 'loss2_reg': loss2_reg, 'loss2_change': loss2_change}, step=step)
                    wandb.log({'loss1': loss1, 'loss2': loss2, 'loss2_control': loss2_control}, step=step)
                    wandb.log(grad1_D, step=step)
                    if self.train_both:
                        wandb.log(grad2_D, step=step)
            
            if step % self.sample_every_n_steps == 0:
                N = 64
                self.sampler.plot_samples(N, step = step, which_model = 'model1', use_wandb = self.use_wandb)
                if self.train_both:
                    self.sampler.plot_samples(N, step = step, which_model = 'model2', use_wandb = self.use_wandb)


if __name__ == '__main__':

    trainer = Trainer(use_functorch = True, use_wandb = True, train_both = True)
    trainer.train_loop()

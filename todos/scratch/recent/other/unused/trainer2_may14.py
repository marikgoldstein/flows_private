import pathlib
import os
import pickle
import time
from typing import Optional, List
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import math
import torchvision
import matplotlib.pyplot as plt
import wandb
import copy
import uuid
from pathlib import Path                              
# local
from utils import (
    is_type_for_logging,
    set_devices, 
    print_total_params,
    Metrics,
    log_results
)
from encoder import Encoder
from utils_data import make_dataloaders
from utils import wandb_alert
from torchvision.utils import save_image
from cleanfid import fid 

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os                                                           
import copy
from diffusion import Diffusion
from models_library import get_unet_fn

Adam = torch.optim.Adam
AdamW = torch.optim.AdamW
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                        

def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def create_logger(logging_dir):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class Trainer(nn.Module):
    def __init__(self, config): 
        super(Trainer, self).__init__()
        self.config = config

    def training_loop(self,):
        n_gpus = torch.cuda.device_count()
        print('dist initialized:', dist.is_initialized())
        print("nccl avail", torch.distributed.is_nccl_available())
        dist.init_process_group("nccl")
        print('dist initialized:', dist.is_initialized())
        print("nccl avail", torch.distributed.is_nccl_available())
        print("elastic launched", torch.distributed.is_torchelastic_launched())
        assert self.config.bsz % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        local_seed = self.config.global_seed * dist.get_world_size() + rank
        torch.manual_seed(local_seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={local_seed}, world_size={dist.get_world_size()}.")
        # Setup an experiment folder:
        if rank == 0:
            os.makedirs(self.config.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{self.config.results_dir}/*"))
            model_string_name = '--------'
            experiment_dir = f"{self.config.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
            self.checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger = create_logger(experiment_dir)
            self.logger.info(f"Experiment directory created at {experiment_dir}")
        else
            self.logger = create_logger(None)
        self.model = get_unet_fn(self.config.arch)(self.config)
        self.ema = deepcopy(self.model).to(device)  
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(device), device_ids=[rank])
        self.logger.info(f"DiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.opt = AdamW(
            self.model.parameters(), 
            lr=self.config.original_lr, # saining uses 1e-4
            weight_decay=self.config.wd
        )

        self.config.C = 3
        self.config.H = 32
        self.config.W = 32
        self.config.d = 32 * 32 * 3
        self.config.flat_d = self.config.d

        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR10(
                './data',
                train=True,
                download=True,
                transform = transform
        )
        self.sampler = DistributedSampler(
                        self.dataset,
                        num_replicas=dist.get_world_size(),
                        rank=rank,
                        shuffle=True,
                        seed=self.config.global_seed
        )
        self.loader = DataLoader(
                self.dataset,
                batch_size=int(self.config.bsz // dist.get_world_size()),
                shuffle=False,
                sampler=self.sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True
        )
  
        self.diffusion = Diffusion(self.config)
        self.encoder = Encoder(self.config)

        # Prepare models for training:
        update_ema(self.ema, self.model.module, decay=0)  # Ensure EMA is initialized with synced weights
        self.model.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()  # EMA model should always be in eval mode

        # Variables for monitoring/logging purposes:
        self.train_steps = 0
        self.epoch = -1
        self.log_steps = 0
        self.running_loss = 0

        self.start_time = time()
        self.logger.info(f"Training!")
        while self.train_steps < 1_000:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.logger.info(f"Beginning epoch {self.epoch}...")
            for batch_idx, (batch, _) in enumerate(self.loader):
                batch = batch.to(device)
                bsz = batch.shape[0]
                u_0, ldj = self.encoder.preprocess(batch)
                loss_dict = self.diffusion.nelbo_fn(batch, self.model)
                loss = loss_dict['nelbo'].mean()
                bpd = self.encoder.nelbo_to_bpd(loss_dict['nelbo'], ldj)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                update_ema(self.ema, self.model.module)
                self.running_loss += loss.item()
                self.log_steps += 1
                self.train_steps += 1
                '''
                if self.train_steps % self.config.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = self.log_steps / (end_time - self.start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(self.running_loss / self.log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    self.logger.info(f"(step={self.train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    self.running_loss = 0
                    self.log_steps = 0
                    self.start_time = time()
                # Save DiT checkpoint:
                if self.train_steps % self.config.ckpt_every == 0 and self.train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": self.model.module.state_dict(),
                            "ema": self.ema.state_dict(),
                            "opt": self.opt.state_dict(),
                            "config": self.config,
                        }
                        checkpoint_path = f"{self.checkpoint_dir}/{self.train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()
                '''
            self.model.eval()  # important! This disables randomized embedding dropout
            # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...             
            self.logger.info("Done!")
            dist.destroy_process_group()
             
'''

 
    def old_data_setup(self,):
            dset = trainer.config.dataset

            # sizes
            if dset == 'mnist':
                    img_size, channels = 28, 1
            elif dset == 'cifar':
                    img_size, channels = 32, 3
            elif dset == 'flowers':
                    img_size, channels = 128, 3
            else:
                    assert False
            flat_d = img_size * img_size * channels

            if dset == 'mnist':
                    train_tf = transforms.Compose([transforms.ToTensor(),])
                    test_tf = train_tf
            elif dset == 'cifar':
                    train_tf = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(), # TODO CHECK
                                    # transforms.Normalize([0.5], [0.5]),
                    ])
                    test_tf = transforms.Compose([transforms.ToTensor()])
            elif dset == 'flowers':
                    assert False
            else:
                    assert False

            # datasets  
            if dset == 'mnist':
                    dset_fn = datasets.MNIST
            elif dset == 'cifar':
                    dset_fn = datasets.CIFAR10
            elif dset == 'flowers':
                    assert False
            else:
                    assert False

            trainset = dset_fn('./data', train=True, download=True, transform=train_tf)
            testset = dset_fn('./data', train=False, download=True, transform=test_tf)

            # loaders
            use_cuda = trainer.config.use_cuda
            num_workers = trainer.config.num_workers
            train_kwargs = {'batch_size': trainer.config.bsz, 'shuffle': True, 'drop_last': True}
            test_kwargs = {'batch_size': trainer.config.bsz, 'shuffle': False}
            if use_cuda:
                    cuda_kwargs = {
                            'num_workers': num_workers,
                            'pin_memory': True,
                            'persistent_workers': True
                    }
                    train_kwargs.update(cuda_kwargs)
                    test_kwargs.update(cuda_kwargs)

             
            trainer.train_loader = DataLoader(trainset, **train_kwargs)
            trainer.test_loader = DataLoader(testset, **test_kwargs)
            



def main_setup(self,):

        self.config = config                                                                                                                                      
        if self.config.overfit:
            self.overfit_batch = torch.load('overfit_batch.pt')

        if self.config.use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name = self.config.wandb_name,
            )
            self.config.wandb_id = self.wandb_run.id
            for key in vars(self.config):
                item = getattr(self.config, key)
                if is_type_for_logging(item):
                    setattr(wandb.config, key, getattr(self.config, key))

        config = generate_ckpt_path(config)
        if self.config.use_wandb:
            wandb.config.save_ckpt_path = self.config.save_ckpt_path
            wandb.config.save_img_path = self.config.save_img_path

        self.config.use_cuda, self.config.device = set_devices()
        make_dataloaders(trainer = self)
        self.encoder = Encoder(trainer = self, dataset=self.config.dataset)
        #self.model_obj = ModelObj(trainer = self)
        self.model_obj.possibly_restore()

def generate_new_folder(parent_dir):

    while True:
        rstr = str(uuid.uuid4())
        rstr = rstr[:int(len(rstr)/4)]
        rstr = rstr.replace('-','')
        rstr = rstr.replace('_','')
        fname = os.path.join(parent_dir, rstr)
        if not os.path.exists(fname):
            break
    return rstr


def generate_ckpt_path(config):

    parent_directory = config.save_ckpt_base_dir
    assert config.save_ckpt_path is None
    if config.use_wandb:
        wandb_id = config.wandb_id
        new_dir_name = 'wb_' + wandb_id
    else:
        new_dir_name = generate_new_folder(parent_directory)
    directory = f'{config.save_ckpt_group_name}/{new_dir_name}'     
    config.save_ckpt_path = os.path.join(parent_directory, directory)
    config.save_img_path = os.path.join(config.save_ckpt_path, 'images/')
    print("making path:{}".format(config.save_ckpt_path))                                                                                                                     
    pathlib.Path(config.save_ckpt_path).mkdir(parents=True, exist_ok=True)
    print("made path:{}".format(config.save_ckpt_path))
    print("making path:{}".format(config.save_img_path))
    pathlib.Path(config.save_img_path).mkdir(parents=True, exist_ok=True)
    print("made path:{}".format(config.save_img_path))
    return config


  @torch.no_grad() 
    def sampling(self, with_ema_model, additional_message = None, compute_fid = False):
        print("sampling. with ema model: {}".format(with_ema_model))
        print("Computing FID:{}".format(compute_fid))
        self.model_obj.eval_mode()

        num_images = self.config.num_sampled_images
        
        prefix = 'ema' if with_ema_model else 'nonema'

        # make samples

        D = self.generate_samples(
            n_samples = num_images, 
            with_ema_model = with_ema_model
        )
        
        model_samples_for_wandb = D['for_wandb']
        model_samples_for_fid = D['for_fid']
    
        # save samples

        pth = self.config.save_img_path
        pth_this_step = os.path.join(pth, 'step_' + str(self.step))
        pathlib.Path(pth_this_step).mkdir(parents=True, exist_ok=True)
        print("saving images to", pth_this_step)
        bsz = model_samples_for_fid.shape[0]
        for i in range(bsz):
            sample_i = model_samples_for_fid[i]
            sample_i = sample_i.cpu()
            sample_i = sample_i.float()
            step = self.step
            save_image(sample_i, os.path.join(pth_this_step, f'{prefix}_img_{i}.png'))
        print("images saved")

        # possibly compute fid
        score = -1.0
        if compute_fid:
            dataset_split="train"
            dataset_split='test'
            score = fid.compute_fid(pth_this_step, 
                        dataset_name='cifar10', 
                        dataset_res=32, 
                        dataset_split = dataset_split, 
                        mode='clean'
                    )
            print("fid:{}".format(score))
        
        # log samples to wandb

        model_samples_to_wandb(
            model_samples_for_wandb, 
            prefix=prefix, 
            use_wandb = self.config.use_wandb, 
            step = self.step, 
            additional_message = additional_message
        )
        return score


class ModelObj:                                                                                                                                                                                                                                                                                                                                                     
    def __init__(self, trainer):

        self.dump()

    def train_mode(self,):
        self.model.train()
    
    def eval_mode(self,):
        self.model.eval()

    def compute_grads_no_step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.maybe_handle_nan_grads()
        self.maybe_clip_grads()

    def dump(self,):
        step = self.get_step()
        data = {
            'optimizer_state': self.opt.state_dict(),
            'step': step,
            'model_state': self.model.state_dict()
        }
        fname = os.path.join(self.config.save_ckpt_path, f'training_state_{step}.pkl')
        torch.save(data, fname)
        del data # conserve memory
        print(f"model checkpoint written in: {fname}")


    def restore(self, model_state, opt_state):
        self.model.load_state_dict(model_state)
        self.model.to(self.config.device)
        print("models restored")
        self.opt.load_state_dict(opt_state)
        print("optimizer state dict loaded")

    def possibly_restore(self,):

        if self.config.resume_path is None:
            print("Starting from blank model")
        else:
            path = self.config.resume_path
            data = torch.load(path, map_location="cpu")
            self.trainer.step = data['step']
            opt_state = data['optimizer_state']
            model_state = data['model_state']
            self.restore(
                model_state = model_state,
                opt_state = opt_state,
            )
            del data # conserve memory
            print("Current step is :", self.trainer.step)



    def maybe_handle_nan_grads(self,):
        if not self.config.handle_nan_grads:
            return
        for param in self.model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, 
                    nan=0, 
                    posinf=1e5, 
                    neginf=-1e5,
                    out=param.grad
                )

    def maybe_clip_grads(self):
        gcn = self.config.grad_clip_norm
        if gcn is None:
            gcn = np.inf
        norm = clip_grad_norm_(self.model.parameters(), max_norm = gcn)
        return norm



        @torch.no_grad()
    def evaluate(self, with_ema_model):

        print("evaluating. with ema model: {}".format(with_ema_model))
        prefix = 'ema' if with_ema_model else 'nonema'

        self.model_obj.eval_mode()
        keys = [key + '_mean' for key in self.config.metric_keys]
        keys += [key + '_std' for key in self.config.metric_keys]
        
        metrics = Metrics(keys, num_mc = -1)
        for batch_idx, batch in enumerate(self.test_loader):
            batch, _ = batch
            if self.config.overfit:
                batch = self.overfit_batch
            batch = batch.to(self.config.device)
            batch_stats = self._step(batch, 'eval', with_ema_model = with_ema_model)
            metrics.append_metrics(batch_stats)
        # take mean over mean/std stats for each batch, over the whole eval set 
        eval_set_mean = {}
        for key in keys:
            mean = metrics.get_stats(key, 'mean')
            eval_set_mean[key + '_mean'] = mean
        log_results(eval_set_mean, 'eval', self.config.use_wandb, self.step, prefix=prefix)
    
        def maybe_aux_metrics(self,):
        no_skip = not self.config.skip_aux_metrics
        is_time = self.epoch % self.config.aux_metrics_every_n_epochs == 0
        if no_skip and is_time:
            self.aux_metrics()
    
        def possibly_plot_real_samples(self,):
        no_skip = not self.config.skip_plot_real_samples
        is_time = self.epoch % self.config.plot_real_samples_every_n_epochs == 0
        if no_skip and is_time:
            self.plot_real_samples()

    def plot_real_samples(self,):
        batch_idx, (batch, _ ) = next(enumerate(self.train_loader))
        if self.config.overfit:        
            batch = self.overfit_batch 
        real_samples_to_wandb(batch, self.config.use_wandb, self.step)

    def possibly_sample(self,):
        no_skip = not self.config.skip_sampling
        is_time = self.epoch % self.config.sample_every_n_epochs == 0
        compute_fid = not self.config.skip_fid and self.epoch % self.config.fid_every_n_epochs == 0
        if no_skip and is_time:
            nonema_fid = self.sampling(with_ema_model=False, compute_fid = compute_fid)
            ema_fid = -1.0

            if self.config.use_ema:
                self.model_obj.ema_model_to_eval_mode()
                ema_fid = self.sampling(with_ema_model=True, compute_fid = compute_fid)

            if compute_fid and self.config.use_wandb:
                print("logging fids, which are", nonema_fid, ema_fid)
                fidD = {'nonema_fid': nonema_fid, 'ema_fid': ema_fid}
                wandb.log(fidD, step=self.step)
    def possibly_evaluate(self,):
        no_skip = not self.config.skip_eval
        is_time = self.epoch % self.config.eval_every_n_epochs == 0
        if no_skip and is_time:
            self.evaluate(with_ema_model=False)
            if self.config.use_ema:
                self.model_obj.ema_model_to_eval_mode()
                self.evaluate(with_ema_model=True)

    def possibly_print_loss(self, loss):
        if self.step % self.config.print_loss_every_n_steps == 0:
            print("current loss is:", loss.item())

    def possibly_checkpoint(self,):
        no_skip = not self.config.skip_save_model
        is_time = self.epoch % self.config.save_every_n_epochs == 0
        if no_skip and is_time:
            self.model_obj.dump()

    def preamble(self,):
       
        print("preamble")
        if self.config.write_overfit:
            assert False

        if self.config.metrics_during_preamble:

            if not self.config.skip_aux_metrics:
                print("initial aux metrics e.g. grad variance")
                self.aux_metrics()

            #print("initial eval")
            #self.evaluate(with_ema_model=False)

            print('make sure data looks okay!')
            self.plot_real_samples()

            print("initial sample test run")
            compute_fid = not self.config.skip_fid
            nonema_fid, ema_fid = -1, -1
            nonema_fid = self.sampling(with_ema_model=False, compute_fid = compute_fid)
            if self.config.use_ema:
                ema_fid = self.sampling(with_ema_model=True, compute_fid = compute_fid)
            print("nonema fid", nonema_fid, "ema_fid", ema_fid)

        print("starting training")

    def sampling_only(self,):
        print("sampling only from this ckpt", self.config.resume_path)
        print("self beta func is", self.config.which_beta)
        print("self ts is" ,self.config.backprop_ts)
        self.sampling(with_ema_model=False)

        '''


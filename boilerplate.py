from time import time 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from copy import deepcopy
from glob import glob
from collections import OrderedDict
import wandb                                                                                                                                                  
import logging
import torch
from utils import is_type_for_logging, requires_grad 
from dit import get_dit
clip_grad_norm_ = torch.nn.utils.clip_grad_norm_                                                                                                                  
#self.scheduler = torch.optim.lr_scheduler.StepLR(
# optimizer, step_size = 1000, gamma = .999)
# N = 10_000, lr_m = 1.0)

class TrainState:
              
    def __init__(self):
        self.train_steps = 0
        self.grad_steps = 0 
        self.log_steps = 0
        self.epoch = 0
        self.running_loss = 0
        self.start_time = time() 

def maybe_log(state, config, logger, rank, device, **kwargs):
    if state.train_steps % config.log_every == 0:
        avg_loss, steps_per_sec = sync_loss(state, device)
        logger.info(f"(step={state.train_steps:07d})")
        logger.info(f"Train Loss: {avg_loss:.4f}")
        logger.info(f"Train Steps/Sec: {steps_per_sec:.2f}")
        D = {'train_loss': avg_loss, 'train_steps_per_sec': steps_per_sec}
        maybe_log_wandb(config, D, state.train_steps, rank)

def sync_loss(state, device):                                                                                                                                                                     
    torch.cuda.synchronize()
    end_time = time()
    steps_per_sec = state.log_steps / (end_time - state.start_time)
    avg_loss = torch.tensor(state.running_loss / state.log_steps, device = device)
    dist.all_reduce(avg_loss, op = dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    # Reset monitoring variables:                                               
    state.running_loss = 0  
    state.log_steps = 0
    state.start_time = time()
    return avg_loss, steps_per_sec

def setup_torch_backends(conf):
    np.random.seed(conf.global_seed)  
    torch.manual_seed(conf.global_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def setup_ddp(config,):                      
    rank = dist.get_rank()                   
    device = rank % torch.cuda.device_count()
    n_gpus = torch.cuda.device_count()       
    assert config.bsz % dist.get_world_size() == 0
    local_seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(local_seed)            
    torch.cuda.set_device(device)            
    print(f"Starting rank={rank}, seed={local_seed}, world_size={dist.get_world_size()}.")
    return rank, device   

def spmd_boilerplate():
    keys = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    env_dict = {key: os.environ[key] for key in keys}
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(            
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )   

def maybe_label_drop(conf, y):
    if conf.label_drop:
        probs = torch.ones_like(y) * conf.label_keep_prob
        keep = torch.bernoulli(probs).bool()
        null_label = torch.ones_like(y) * conf.null_token
        y = torch.where(keep, y, null_label)    
    return y

def do_update_ema(steps, config):
    cond1 = steps > config.update_ema_after
    cond2 = steps % config.update_ema_every == 0   
    return cond1 and cond2

def maybe_update_emas(config, models, state, **kwargs):
    steps = state.train_steps
    if do_update_ema(steps, config):
        keys = [k for k in models if 'ema' not in k]
        for key in keys:
            assert key in models
            assert key + '_ema' in models
            update_ema(
                model_ema = models[key + '_ema'],
                model = models[key],
                decay = config.ema_decay
            )
 
# copied from https://github.com/facebookresearch/DiT
def update_ema(model, model_ema, decay):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    ema_params = OrderedDict(model_ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad 
        # to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    print("updated ema")

def setup_overfit(config, loader, mode = 'datapoint', **kwargs):  

    #print("In Debug Mode. Overfitting on one datapoint!!!!!!")
    x, y = next(iter(loader)) # useful for debug

    if mode == 'datapoint':
        index = 42
        x0 = x[index][None, :, :, :]
        assert x0.shape == (1, config.C, config.W, config.H)
        x0 = x0.repeat(x.shape[0], 1, 1, 1)
        overfit_x = x0
        overfit_y = (torch.ones_like(y) * y[index]).long()
    
    elif mode == 'batch':

        overfit_x, overfit_y = x, y

    else:

        assert False
    return overfit_x, overfit_y

def make_model_and_ema(config, device, rank):
    model = get_dit(config)
    model_ema = deepcopy(model).to(device)
    requires_grad(model_ema, False)
    if rank is not None:
        print("using ddp in model init")
        model = DDP(model.to(device), device_ids=[rank])
    else:
        model.to(device)
    # Ensure EMA is initialized with synced weights
    update_ema(model = model, model_ema = model_ema, decay=0.)
    model.train()
    model_ema.eval() # EMA model should always be in eval mode        
    return model, model_ema

def _trim_grads(conf, model):
    if conf.handle_nan_grads:
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
    # if set grad clip norm to e.g. 10000000, this just simply returns norm
    gcn = conf.grad_clip_norm if conf.grad_clip_norm > 0. else 1000000.
    norm = clip_grad_norm_(model.parameters(), max_norm = gcn)
    return norm

        
def maybe_trim_grads(models, config, rank, steps, log_norm = False):
    D = {key + '_grad_norm' : _trim_grads(config, models[key]) for key in models}
    # does this slow down training? logging to wandb every grad step?
    if rank == 0 and log_norm and config.use_wandb: 
        wandb.log(D, step = steps)    

                 
def get_params(models):                        
    params = [] 
    for key in models:                         
        params += list(models[key].parameters())                                                                                                                              
    return params


# WANDB SETUP

def setup_wandb(config):
    print("setting up wandb")
    if config.use_wandb:
        config.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                #name = self.config.wandb_name,
        )        
        #self.config.wandb_id = self.wandb_run.id
        # copy config args to wandb
        for key in vars(config):
                item = getattr(config, key)
                if is_type_for_logging(item):
                        setattr(wandb.config, key, item)

def setup_wandb_distributed(config, rank, **kwargs):
    if rank == 0:
        setup_wandb(config)


# LOGGING and directory making

def create_logger(logging_dir):
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )          
    logger = logging.getLogger(__name__)
    return logger  

def make_directories(config):

    # Make results folder (holds all experiment subfolders)
    os.makedirs(config.results_dir, exist_ok=True)  
    experiment_index = len(glob(f"{config.results_dir}/*"))
    experiment_name = config.experiment_name

    # Create an experiment folder
    experiment_dir = f"{config.results_dir}/{experiment_index:03d}-{experiment_name}"
    
    # Stores saved model checkpoints
    config.checkpoint_dir = f"{experiment_dir}/checkpoints"  
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    print("results dir", config.results_dir)
    print("experiment dir", experiment_dir)
    print("ckpt dir", config.checkpoint_dir)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    # make sample dir for all steps/epochs
    config.sample_dir = os.path.join(config.checkpoint_dir, 'samples')
    os.makedirs(config.sample_dir, exist_ok=True)
    print(f"Will be Saving png samples at {config.sample_dir}")

    return config, logger

def make_directories_distributed(config, rank, **kwargs):

    if rank == 0:
        config, logger = make_directories(config)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    dist.barrier()
    return config, logger

# wandb logging

def _log(D, steps):
    wandb.log(D, step = steps)

def log_distributed(D, steps, rank):
    if rank == 0:
        _log(D, steps)

def maybe_log_wandb(conf, D, steps, rank):
    if steps % conf.log_every == 0:
        if conf.use_wandb:
            log_distributed(D, steps, rank)

# CHECKPOINTING 
def save_ckpt(config, steps, checkpoint):
    print("saving ckpt")                   
    checkpoint_path = f"{config.checkpoint_dir}/{steps:09d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def save_ckpt_distributed(config, steps, checkpoint, rank): 
    if rank == 0:
        save_ckpt(config, steps, checkpoint)
    dist.barrier()

def maybe_ckpt(dgm, config, state, rank, **kwargs):                   
    steps = state.train_steps
    ckpt_every = config.ckpt_every                               
    if steps % ckpt_every == 0 and steps > 0:                    
        D = dgm.get_ckpt_dict()                                  
        save_ckpt_distributed(config, steps, D, rank)   


def do_fid(steps, config):
    skip_fid = config.skip_fid                                
    fid_every = config.fid_every                              
    cond1 = not skip_fid                                      
    cond2 = steps % fid_every == 0                            
    return cond1 and cond2


# FID
def maybe_fid(config, name, rank, state, **kwargs):                 
    steps = state.train_steps                                 
    if rank == 0 and do_fid(steps, config):
        for use_ema in [False, True]:                         
            compute_fid(                                      
                config = config,                              
                name = name,
                use_ema = use_ema,                            
                steps = steps                                 
            )                                                                                                                            
        dist.barrier()
 
def compute_fid(config, name, use_ema, steps):
    print("Computing FID")  
    directory = sample_dir_this_step(config.sample_dir, name, use_ema, steps)
    score = compute_one_cifar_fid(directory)
    print("name", name, "use_ema" , use_ema, "score", score)
 
def compute_one_cifar_fid(directory):
    return fid.compute_fid(
            directory,
            dataset_name = 'cifar10',
            dataset_res = 32,
            model_name = 'inception_v3',
            dataset_split = 'train',
            mode = 'clean'
    )

# SAMPLE
def sample_dir_this_step(sample_dir, name, use_ema, steps):
    subdir_name = f"{name}_{steps:09d}"               
    subdir_name += '_ema' if use_ema else '_nonema'       
    directory = os.path.join(sample_dir, subdir_name)         
    os.makedirs(directory, exist_ok = True)                   
    return directory                             

def do_sample_ema(steps, config):
    cond1 = config.sample_with_ema                                      
    cond2 = steps > config.update_ema_after                             
    cond3 = steps % config.sample_ema_every == 0
    return cond1 and cond2 and cond3

def maybe_sample(dgm, name, config, state, rank, device, cheap, **kwargs):    
    if rank == 0:
        D = {}
        steps = state.train_steps
        args = {
            'dgm': dgm,
            'config': config,
            'state': state,
            'rank': rank,
            'device': device,
            'cheap': cheap,
            'steps': steps,
            'name': name,
        }
        if steps % config.sample_every == 0:                                
            x = sample(use_ema = False, **args)
            D[name + '_nonema'] = x

        if do_sample_ema(steps, config):
            x_ema = sample(use_ema = True, **args)
            D[name + '_ema'] = x_ema
        if config.use_wandb:
            wandb.log(D, step = steps)
    dist.barrier()

@torch.no_grad()
def sample(config, name, dgm, steps, rank, device, use_ema = False, cheap = False, **kwargs):
    return dgm.sample(
        config = config,
        directory = sample_dir_this_step(config.sample_dir, name, use_ema, steps),
        device = device,
        use_ema = use_ema,
        cheap = cheap,
    )

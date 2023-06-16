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


def maybe_label_drop(conf, y):

    if conf.label_drop:

        probs = torch.ones_like(y) * conf.label_keep_prob
        keep = torch.bernoulli(probs).bool()
        null_label = torch.ones_like(y) * conf.null_token
        y = torch.where(keep, y, null_label)
    
    return y


# copied from https://github.com/facebookresearch/DiT
def update_ema(model_ema, model, decay):
    # TODO if using DDP, need to unwrap model but not model_ema
    print("updating ema model")
    ema_params = OrderedDict(model_ema.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():                                                                                                                  
            # TODO: Consider applying only to params that require_grad 
            # to avoid small numerical changes of pos_embed
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    print("updated ema")



def setup_overfit(config, loader, mode = 'datapoint'):  

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

def make_model_and_ema(config):
    device = config.device
    model = get_dit(config)
    model_ema = deepcopy(model).to(device)
    requires_grad(model_ema, False)
    model.to(device)
    model.train()
    model_ema.eval() # EMA model should always be in eval mode        
    return model, model_ema


def trim_grads(conf, model):

    if conf.handle_nan_grads:
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

    # if set grad clip norm to e.g. 10000000, this just simply returns norm
    gcn = conf.grad_clip_norm if conf.grad_clip_norm > 0. else 1000000.
    norm = clip_grad_norm_(model.parameters(), max_norm = gcn)

    return norm



def setup_wandb(config):
    print("setting up wandb")
    if config.use_wandb:
        config.wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
        )               
        # copy config args to wandb
        for key in vars(config):
                item = getattr(config, key)
                if is_type_for_logging(item):
                        setattr(wandb.config, key, item)


def create_logger(logging_dir):
    logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )          
    logger = logging.getLogger(__name__)
    return logger  

def make_directories(config, rank = None):

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


def sample_dir_this_step(sample_dir, which, ema, step):
    subdir_name = f"{which}_{step:09d}"   
    subdir_name += '_ema' if ema else '_nonema'
    d = os.path.join(sample_dir, subdir_name)
    os.makedirs(d, exist_ok = True)       
    return d  


def save_ckpt(config, steps, checkpoint):           
    print("saving ckpt")                   
    checkpoint_path = f"{config.checkpoint_dir}/{steps:09d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")



import torch
from cleanfid import fid
from boilerplate import sample_dir_this_step




def compute_one_cifar_fid(directory):
    return fid.compute_fid(
            directory,
            dataset_name = 'cifar10',
            dataset_res = 32,
            model_name = 'inception_v3',
            dataset_split = 'train',
            mode = 'clean'
    )

def maybe_compute_fid(trainer, which, use_ema, train_steps):                                                                                                                                                                                                     
    conf = trainer.config    
    skip_fid = conf.skip_fid
    fid_every = conf.fid_every
    if not skip_fid and trainer.train_steps % fid_every == 0:
        compute_fid(trainer, which, use_ema, train_steps)

def compute_fid(trainer, which, use_ema, train_steps):
    print("Computing FID")  
    conf = trainer.config
    directory = sample_dir_this_step(conf.sample_dir, which, use_ema, train_steps)
    score = compute_one_cifar_fid(directory)
    print("which", which, "use_ema" , use_ema, "score", score)

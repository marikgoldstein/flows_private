import torch

class ExperimentConfig:
    def __init__(self, dataset, use_wandb, results_dir, debug):


        self.dgm_type = 'flow'
        self.global_seed = 1
        self.use_wandb = use_wandb
        self.results_dir = results_dir 
        self.debug = debug # this can mean anything you choose it to mean in trainer

        # subdir of results_dir, eg for "example"
        # project will live in results_dir/example-001, .../example-002
        # numbers appended automatically.
        self.experiment_name = 'example' 
       
        # data
        self.dataset = dataset
        self.num_workers = 4
        self.bsz = 128 if self.dataset != 'imagenet256' else 32

        # optimization 
        self.total_train_steps = 1_000_000 
        self.base_lr = 2e-4 
        self.wd = 0.0 
        self.grad_clip_norm = 2 
        self.handle_nan_grads = False
        self.grad_accum = 1

        # encoding/decoding 
        self.use_vae = False if self.dataset != 'imagenet256' else True
        self.dequantize = False

        # conditioning
        self.label_drop = 1
        self.label_keep_prob = .99999
     
        # specific to flows/interpolants
        self.T_min = 1e-5 # both
        self.T_max = 1 - 1e-5 # both
        self.epsilon = torch.tensor([.1])
        self.do_bv = True # interpolant
        self.do_seta = True # False # interpolant
        assert self.do_bv or self.do_seta, 'must train at least 1 model' #interpolant

        # ema
        self.ema_decay = 0.9999
        self.update_ema_after = 10_000 
        self.update_ema_every = 100 
        self.fid_with_ema = False
        self.sample_with_ema = True
        self.eval_with_ema = False

        # image saving
        self.saining_postprocessing = True # if true, uses PIL and a specific method for mapping [0,1] to [0,255]
        self.use_pil = True # if true, save images with pil, else with torchvision

        # sampling / integration
        self.n_sample_steps = 500 
        self.num_sampled_images = 64 if self.dataset != 'imagenet256' else 16
        self.integration_atol = 1e-5                                                                                                     
        self.integration_rtol = 1e-5
        self.integration_method = 'dopri5'

        # periodic logging/eval
        if self.dataset != 'imagenet256':
            # training is slow, so do stuff more frequently
            self.log_every = 100
            self.ckpt_every = 500
            self.sample_every = 500
            self.sample_ema_every = self.sample_every * 2
            self.fid_every = 1000
            #self.eval_every = 1000

        else:            
            self.log_every = 10
            self.ckpt_every = 100
            self.sample_every = 50
            self.sample_ema_every = 100

        # model
        self.arch = 'dit'
        self.arch_size = 'small'
       
        # wandb
        self.wandb_project = 'flow'
        self.wandb_entity = 'marikgoldstein'

        # quicker / more frequent logging
        if self.debug:
            self.num_sampled_images = 128
            self.n_sample_steps = 500
            self.sample_every = 500
            self.ckpt_every = 500
            self.log_every = 100
            self.sample_ema_every = 500
            self.update_ema_after = 10_000
            self.update_ema_every = 500
            self.fid_every = 1000


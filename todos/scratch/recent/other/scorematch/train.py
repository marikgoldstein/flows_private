import glob
import tqdm
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
import wandb

# custom / local
from ncsn import NCSN, NCSNdeeper
from ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from get_sigmas import get_sigmas
from utils import get_optimizer, EMAHelper,  make_cifar_data, dsm, ald , is_type_for_logging 

# consider score matching (no times) for gaussian data as a simple example versus 
# maximum likelihood (use a normalizing flow and really try to overfit the training objective)


class Trainer():

    def __init__(self, config):
        self.config = config
            
        # set device
        self.config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info("Using device: {}".format(self.config.device))

        # set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = True
        
        # data
        make_cifar_data(trainer = self)
        #self.make_gaussian_data()

        self.step = 0
        self.epoch = 0
        
        # model init
        self.score_model = NCSNv2(self.config)
        self.score_model.to(self.config.device)

        total_params = 0
        total_params_grad = 0
        for n, p in self.score_model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                total_params_grad += p.numel()
        print(f"Total params: {total_params}")
        print(f"Total params (requires grad): {total_params_grad}")

        self.optimizer = get_optimizer(self.config, self.score_model.parameters())

        if self.config.use_ema:
            self.ema_helper = EMAHelper(mu=self.config.ema_rate)
            self.ema_helper.register(self.score_model)

        self.possibly_resume()

        # noise schedule
        self.sigmas = get_sigmas(self.config)
    
        self.setup_wandb()

        self.sample_dir = os.path.join(self.config.ckpt_dir, 'samples')                                                                                                                            
        print("overall sample dir will be", self.sample_dir)
        os.makedirs(self.sample_dir, exist_ok=True)

    def setup_wandb(self):
        if self.config.use_wandb:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
            )               
            self.config.wandb_id = self.wandb_run.id
            for key in vars(self.config):
                item = getattr(self.config, key)
                if is_type_for_logging(item):
                    setattr(wandb.config, key, getattr(self.config, key))


    def maybe_log_wandb(self, D, step):
        if self.config.use_wandb:
            wandb.log(D, step = step)

    def possibly_resume(self,):
        if self.config.resume_training:
            assert False
            states = torch.load(os.path.join(self.config.ckpt_dir, 'checkpoint.pth'))
            self.score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim_eps
            self.optimizer.load_state_dict(states[1])
            self.start_epoch = states[2]
            self.step = states[3]
            if self.config.use_ema:
                self.ema_helper.load_state_dict(states[4])


    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def data_transform(self, X):
        if self.config.uniform_dequantization:
            X = X / 256. * 255. + torch.rand_like(X) / 256.
        if self.config.gaussian_dequantization:
            X = X + torch.randn_like(X) * 0.01

        if self.config.rescaled:
            X = 2 * X - 1.
        elif self.config.logit_transform:
            X = logit_transform(X)

        if hasattr(self.config, 'image_mean'):
            return X - self.config.image_mean.to(X.device)[None, ...]

        return X

    def inverse_data_transform(self, X):
        if hasattr(self.config, 'image_mean'):
            X = X + self.config.image_mean.to(X.device)[None, ...]

        if self.config.logit_transform:
            X = torch.sigmoid(X)
        elif self.config.rescaled:
            X = (X + 1.) / 2.

        return torch.clamp(X, 0.0, 1.0)

    def step_loss(self, X):
        loss = dsm(self.score_model, X, self.sigmas, labels=None, anneal_power=self.config.training_anneal_power)

        logging.info("step: {}, loss: {}".format(self.step, loss.item()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.config.use_ema:
            self.ema_helper.update(self.score_model)

        return loss.item()


    def prepare_batch(self, X):
        X = X.to(self.config.device)
        X = self.data_transform(X)
        return X

    def train_loop(self,):
        
        print("starting train loop")
        for epoch in range(self.config.start_epoch, self.config.num_epochs):

            print(f"starting epoch:{epoch}")

            for i, (X, y) in enumerate(self.train_loader):
                
                self.score_model.train()
                self.step += 1

                X = self.prepare_batch(X)

                loss_item = self.step_loss(X)

                self.maybe_log_wandb({'loss:':loss_item}, step = self.step)

                if self.step >= self.config.total_steps:
                    return 0

                if self.step % self.config.print_every == 0:
                    print(f"Step:{self.step}")

                if self.step % self.config.eval_every == 0:
                    print("evaluating")
                    self.evaluate(self.step)

                if self.step % self.config.ckpt_every == 0:
                    print("writing ckpt")
                    self.write_ckpt(epoch, self.self.step)
                
                if self.step % self.config.sample_every == 0:
                    print("sampling during training")
                    self.sample_during_training(self.step)

    @torch.no_grad()
    def evaluate(self, step):

        if self.config.use_ema:
            test_score_model = self.ema_helper.ema_copy(self.score_model)
        else:
            test_score_model = score_model

        test_score_model.eval()
        try:
            test_X, test_y = next(self.test_iter)
        except StopIteration:
            test_iter = iter(self.test_loader)
            test_X, test_y = next(self.test_iter)

        test_X = self.prepare_batch(test_X)

        test_dsm_loss = dsm(test_score_model, test_X, self.sigmas, labels=None, anneal_power=self.config.training_anneal_power)

        logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

        self.maybe_log_wandb({'test_loss:': test_dsm_loss.item()}, step = self.step)  

        del test_score_model

    @torch.no_grad()
    def write_ckpt(self, epoch, step):

        states = [
            self.score.state_dict(),
            self.optimizer.state_dict(),
            epoch,
            self.step,
        ]
        if self.config.use_ema:
            states.append(self.ema_helper.state_dict())

        torch.save(states, os.path.join(self.config.ckpt_dir, 'checkpoint_{}.pth'.format(step)))
        torch.save(states, os.path.join(self.config.ckpt_dir, 'checkpoint.pth'))

    def make_sample_dir_this_step(self,):
        sample_dir_this_step = self.get_sample_dir_this_step()
        os.makedirs(sample_dir_this_step, exist_ok=True)
        print("sample dir this step is", sample_dir_this_step)

    def get_sample_dir_this_step(self,):
        subdir_name = f"{self.step:07d}"
        sample_dir_this_step = os.path.join(self.sample_dir, subdir_name) 
        return sample_dir_this_step

    def log_images(self, sample, step):
       
        self.make_sample_dir_this_step()
        sample_dir_this_step = self.get_sample_dir_this_step() 
       
        # make grid 
        samples = sample
        samples = samples.cpu()
        samples_bsz = samples.shape[0]
        grid_size = int(np.floor(np.sqrt(samples_bsz)))
        samples = samples[0:grid_size**2]
        image_grid = make_grid(samples, nrows=grid_size)
        grid_path = os.path.join(sample_dir_this_step, 'image_grid.png')

        # save individua samples
        bsz = samples.shape[0]
        for i in range(bsz):
            sample_i = samples[i].cpu().float()
            step = self.train_steps
            save_image(sample_i, os.path.join(sample_dir_this_step, f'steimg_{i}.png'))
        print("images saved")

        # log to wandb:        
        if self.config.use_wandb:
            image_grid = image_grid.permute(1,2,0)

            image_grid = torch.clamp(image_grid * 255.0, min = 0.0, max = 255.0).byte()

            image_grid = image_grid.numpy()
            for_wandb = [
                wandb.Image(np.array(image_grid))
            ]
            key = 'model_samples'
            wandb.log({key:for_wandb}, step = step)

    @torch.no_grad()
    def sample_during_training(self, step):

        if self.config.use_ema:
            test_score_model = self.ema_helper.ema_copy(self.score_model)
        else:
            test_score_model = score

        test_score_model.eval()

        init_samples = torch.rand(36, self.config.C, self.config.W , self.config.H, device=self.config.device)
        
        init_samples = self.data_transform(init_samples)
        verbose = False
        all_samples = ald(init_samples, test_score_model, self.sigmas.cpu().numpy(),
                                               self.config.sampling_n_steps_each,
                                               self.config.sampling_step_lr,
                                               final_only=True, verbose=verbose,
                                               denoise=self.config.sampling_denoise)

        bsz = all_samples[-1].shape[0]
        shape = (bsz, self.config.C, self.config.H, self.config.W)
        sample = all_samples[-1].view(shape)
        sample = self.inverse_data_transform(sample)

        self.log_images(sample, step)
        
        del test_score_model
        del all_samples

    def proper_sampling_with_fid_eval(self,):
        assert False, "todo"
        '''
        total_n_samples = self.config.sampling_num_samples_fid
        n_rounds = total_n_samples // self.config.sampling_bsz
        if self.config.sampling.data_init:
            dataloader = DataLoader(dataset, batch_size=self.config_sampling.bsz, shuffle=True,
                                    num_workers=4)
            data_iter = iter(dataloader)

        img_id = 0
        for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
            if self.config.sampling.data_init:
                try:
                    samples, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                samples = samples.to(self.config.device)
                samples = data_transform(self.config, samples)
                samples = samples + sigmas_th[0] * torch.randn_like(samples)
            else:
                samples = torch.rand(self.config.sampling_bsz, self.config.data.channels,
                                     self.config.data.image_size,
                                     self.config.data.image_size, device=self.config.device)
                samples = data_transform(self.config, samples)

            all_samples = ald(samples, score, sigmas,
                                                   self.config.sampling.n_steps_each,
                                                   self.config.sampling.step_lr, verbose=False,
                                                   denoise=self.config.sampling.denoise)

            samples = all_samples[-1]
            for img in samples:
                img = inverse_data_transform(self.config, img)

                save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                img_id += 1
        '''
        return False

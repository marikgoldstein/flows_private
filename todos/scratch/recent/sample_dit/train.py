import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torchvision.datasets.utils import download_url
import torch
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from utils import find_model, download_model, create_diffusion, center_crop_arr
from dit import DiT          

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader



class Args:

    def __init__(self,):

        self.num_sampling_steps = 250
        self.seed = 0
        self.image_size = 256
        self.num_classes = 1000
        self.cfg_scale = 4.0
        self.latent_size = self.image_size // 8
        self.model_name = "DiT-XL/2"
        self.ckpt_path = f"DiT-XL-2-{self.image_size}x{self.image_size}.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.data_path = '../data/imagenet_1k_val_subset'
        self.data_path = '../data/imagenet_tiny'
        self.num_workers = 4
        self.batch_size = 4

if __name__ == '__main__':

    args = Args()
    
    torch.manual_seed(args.seed)

    # load models
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, 
                        input_size = args.latent_size, num_classes = args.num_classes)
    model = model.to(args.device)
    state_dict = find_model(args.ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(args.device)

    noise_real_data = False
    sample_from_model = True#False

    if noise_real_data:

        # Setup data:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        (x,y) = next(iter(loader))
        del loader
        x = x.to(args.device)

        x = x[:4]
        something = x[2]
        turkey = x[3]
        turkeys = turkey[None,:,:,:].repeat(16, 1, 1, 1)

        turkeys[1] = something

        for i in range(14):
            turkeys[2 + i] = .5 * turkey + .5 * something


        x = turkeys
        z = vae.encode(x).latent_dist.sample().mul_(0.18215)
        print("x shape", x.shape)
        print("z shape", z.shape)
        def noise(x, t):
            mean_coef_squared = 1-t
            variance = t
            mean_coef = np.sqrt(mean_coef_squared)
            return mean_coef * x + np.sqrt(variance) * torch.randn_like(x)
        
        def decode(latent):
            return vae.decode(latent / 0.18215).sample

        #lst = [decode(z)] + [decode(noise(z,0.05)) for i in range(3)]
        #grid = torch.cat(lst, dim=0)
        grid = decode(noise(z, .05))

        save_image(grid, 'both.png', nrow=4, normalize=True, value_range=(-1,1))
        #save_image(not_noised, "sample_not_noised.png", nrow=4, normalize=True, value_range=(-1, 1))
        #save_image(noised, "sample_noised.png", nrow=4, normalize=True, value_range=(-1, 1))
        print("done")

    if sample_from_model:


        # sampling arg
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
        class_labels = [207]
        n = len(class_labels)
        z = torch.randn(n, 4, args.latent_size, args.latent_size, device=args.device)
        y = torch.tensor(class_labels, device=args.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=args.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=args.device, verbose=True
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


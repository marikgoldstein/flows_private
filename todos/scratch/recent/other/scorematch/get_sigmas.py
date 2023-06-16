import numpy as np
import torch

def get_sigmas(config):                                                                                                                                                                               
    if config.model_sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model_sigma_begin), np.log(config.model_sigma_end),
                               config.model_num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model_sigma_begin, config.model_sigma_end, config.model_num_classes)
        ).float().to(config.device)
    else:
        raise NotImplementedError('sigma distribution not supported')
    return sigmas


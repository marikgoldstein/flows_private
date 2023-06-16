
import numpy as np
import torch
import matplotlib.pyplot as plt
from cleanfid import fid                                                                                                                                                                      
directory = './ckpts/cow/wb_gyw1089y/images/step_0'
dataset_split="train"
dataset_split='test'
score = fid.compute_fid(directory, dataset_name='cifar10', dataset_res=32, dataset_split = dataset_split, mode='clean')
print("score!", score)
#To use the CLIP features when computing the FID [Kynkäänniemi et al, 2022], specify the flag model_name="clip_vit_b_32"


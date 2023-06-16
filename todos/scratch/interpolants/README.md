
# usage

## Shorter:
- just add your wandb info into config.py
- then run main.py

For a debug run of mnist
```
python train.py --debug 1 --use_wandb 1 --dataset mnist
```
or for a regular run of cifar
```
python train.py --debug 0 --use_wandb 1 --dataset cifar
```

## Longer:
- two sets of args: main args for train.py and the args in config.py
- main args are just 
	- use_wandb (currently an int 0 or 1)
	- results_dir (defaults to "./ckpts")
	- debug (currently an int 0 or 1). overfits to 1 datapoint if ture. Useful for tesig code. You can make "debug" mean anything you want.
	- dataset (mnist/cifar). the data_utils.py file also support imagenet if you download and preprocess it. TODO add instructions here. 
- check config.py for any config preferences re flow, optimization, logging, etc. If you end up changing an arg frequently, you can add it to main args in train.py and copy it into the config

# Debugging option:

- Currently, the debug flag makes the model overfit on one datapoint. For mnist/cifar, you should see that the first sampling step (not the step 0 sampling step, but eg the step 500 sampling step) produces the datapoint pretty well.

# Model architecture 

I took models.py from [the diffusion transformer repo](https://github.com/facebookresearch/DiT). Check the get dit function at bottom of dit.py. Currently there are two presets, small and large

# Encoder:

- for mnist and cifar, the Encoder class (see encoder.py) assumes the data is in [0,1], and stretches it to [-1,1]. It support dequantization (noising the data to make it continuous but this is off by default). When samping, the produced data, for a good model, is roughly in [-1,1], so the decoding step shrinks the data back to roughly [0,1].

- for imagenet (which requires a little extra setup) the default in the code is to use the pretrained stable diffusion vae, like in Saining's diffusion transformer repo. This has been tested in my other repo but I didn't yet do the quick debug checks for this feature in this code base. In this case, the encoder uses the vae's encode and decode and does not use its centering/uncentering mentioned in the previous bullet point.

# TODOS
- lets check out [this mila repo for reference](https://github.com/atong01/conditional-flow-matching)
- add slurm sbatch scripts
- better sampling algorithms besides vanilla euler integration of the ode
- add fid computation back in 
	- add in the the code to save model samples to a directory for the fid. Be very careful with this and use the processing in the ImageProcessing class included in this directory. FID is very sensititive to the settings for saving images.
- add in the DDP multigpu stuff
- add in some kind of likelihood or elbo evaluation
- lr scheduler, not urgent
- ask Saining about DiT hyperparams. Check the "get dit" function in dit.py
- implement the random label dropping for doing mixed conditional/unconditional modeling. See you will label_drop = False in the config file. When you implement this, make sure to add +1 to the num classes in the dit initialization in get dit in dit.py to make room for the null token. During training, randomly drop the label to the null token with small probability. During sampling, many options (only null token, only real classes, mixed w/ guidance scale, etc).


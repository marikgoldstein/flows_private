
# usage
```
sh 1gpu.sh
```

# Model architecture 

I took models.py from [the diffusion transformer repo](https://github.com/facebookresearch/DiT). Check the get dit function at bottom of dit.py. Currently there are two presets, small and large

# Encoder:

- for mnist and cifar, the Encoder class (see encoder.py) assumes the data is in [0,1], and stretches it to [-1,1]. It support dequantization (noising the data to make it continuous but this is off by default). When samping, the produced data, for a good model, is roughly in [-1,1], so the decoding step shrinks the data back to roughly [0,1].

- for imagenet (which requires a little extra setup) the default in the code is to use the pretrained stable diffusion vae, like in Saining's diffusion transformer repo. This has been tested in my other repo but I didn't yet do the quick debug checks for this feature in this code base. In this case, the encoder uses the vae's encode and decode and does not use its centering/uncentering mentioned in the previous bullet point.

# TODOS
- lets check out [this mila repo for reference](https://github.com/atong01/conditional-flow-matching)
- add in some kind of likelihood or elbo evaluation
- lr scheduler, not urgent
- ask Saining about DiT hyperparams. Check the "get dit" function in dit.py


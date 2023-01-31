# Diffusion models

## Target space

- latent vs pixel space
  - latent spaces are preferable
    - continuous space
      - VAE + GAN
        - where vector quantization is absorbed in the decoder
    - discrete space
      - VQGAN vector quantization and the diffusion model is going to be discrete as well
  - pixel space models are too expensive to train them
- notes
  - Gaussian noise makes the score function is trained on the entire space
  - But in high dimensional spaces, the most samples are clustered around certain areas



## Data issues

##### Small datasets

- overfitting
  - for small datasets, a large diffusion model may memorize the training samples
- augmentation regularization looks useful
  - augment data with the augmentation parameters are provided as conditions

##### Videos

- Joint training on video and image modeling
  - concatenate frames of a video and independent images
  - mask temporal attention for those independent pairs

##### Super resolution

- conditioning augmentation looks useful
  - for super resolution models augment the low-resolution conditional images
  - reduces the sensitivity to domain gap between the cascade stages
  - example
    - train with Gaussian noise augmentation using a randomly samples SNR
    - for inference use a fixed SNR ratio such as 3 or 5
  - possible sub-methods
    - blurring augmentation
    - truncated conditioning augmentation
    - non-truncated conditioning augmentation
  - refer to [Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/abs/2106.15282) for more details

## Conditioning

- types
  - concatenate conditions as a part of input channels
  - provide conditions as inputs for predicting layer normalization parameters
  - provide conditions as inputs for QV values of cross attention module
- classifier free guidance (CFG)
  - looks useful and widely used 
  - $p_\text{uncond}=0.1$ looks widely used. (?)



## Architectures

- U-Net + self/cross-attention
  - mostly used
- Transformers
  - recently achieved SOTA

- multi staged models are known to be useful
  - taking account of capacity-efficiency trade-offs 



## Network blocks

- self-attention with relative positional encodings



## Code bases

- k-diffusion looks clean but still incomplete
- LDM uses DDPM + pytorch lightning
  - note that current pytorch lightning does not fit to calculating FID scores for which we don't need to iterate on validation set chunks



## Inference

- samplers
  - ancestral samplers
    - DDPM
  - k-diffusion
    - looks good but a bit heuristic
  - DDIM
    - can be set to be deterministic
- rejection sampling based methods
  - we can reject generated samples if the provided conditions can be estimated from the generated samples
- notes
  - practically using a good scheduler is important since calculating FID is quite expensive



## Training

- train until FID/KID converges
- check memorization when training with small datasets
  - find the closest training samples to each generated sample
  - increase of validation loss is a necessary condition for memorization
- hyperparameters
  - $T$
    - the larger the better
    - usually set to 1000 or 2000
    - consider going to the continuous time models introduced by the "Score SDE" paper



## Evaluation

##### (Images)

- metrics
  - fid
    - using 50000 training samples and 50000 generated samples is preferable.
    - but practically 2000 samples for each of them is used.
  - kid
    - known as good for smaller samples
  - training and validation loss values
    - also useful
- pretrained networks to be used
  - Inception V3
- notes
  - for conditional models we need to think about how to provided conditions
    - picking a fixed condition
    - random sampling from a well-known distribution
    - resampling

#####  (Videos)

- metrics
  - FID
  - IS
  - FVD
- pretrained networks to be used
  - I3D
  - C3D



## Controllable generation

- conditional generation
  - methods
    - using a pretrained classifier
      - which is trained with noisy data and the class labels
    - CFG
- inverse problem
  - to recover un unknown signal from observed measurements
    - inpainting (or imputation)
    - super resolution
  - methods
    - projection-onto-convex-sets (POCS)
      - measurement consistency step
    - ILVR
      - POCS using lower resolution images
    - come-closer-diffuse-faster (CCDF)
      - POCS + starting the reverse diffusion steps from an estimated noisy sample
    - manifold constrained gradient (MCG)
      - POCS + MCG
    - reconstruction guided sampling
      - (original replacement method introduced in Score SDE) + (a correcting term)
      - [Video Diffusion Models](https://arxiv.org/abs/2204.03458)



## References

(non diffusion prerequisites)

- VQ-VAE

- VQGAN



(diffusion cores methods)

- non-equilibrium thermodynamics (NET)
- DDPM
- iDDPM
- DDIM
- Score SDE
- guided-diffusion
- k-diffusion
- Variational diffusion models (VDM)
- Latent diffusion models (LDM)
- CFG
- DiT
- Video diffusion models



(applications)

- ILVR
- CCDF
- MCG
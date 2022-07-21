# Diffusion based models

## DDPM review

- questions
  - the range of pixel values that are to be normally distributed
    - with zero mean and one standard deviation
  - meaning of
    - "multiple noise levels during training and with annealed Langevin dynamics during sampling"
  - how did they measure log likelihoods?
  - what kind of likelihoods based models are there?
  - what's annealed importance sampling
  - what are annealed Langevin dynamics

- Denoising Diffusion Probabilistic Models (DDPM)
  - https://arxiv.org/abs/2006.11239
  - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  - contributions
    - showed that high quality image generation is possible by DDPM
    - showed connections/equivalence to/with denoising score matching when
      - training with multiple noise levels
      - sampling with annealed Langevin dynamics
        - https://en.wikipedia.org/wiki/Langevin_dynamics
    - training on a weighted variational bound
    - seeing DDPM as progressive decoding in the context of lossy decompression
      - explains relatively poor log likelihoods
      - the majority of the models' lossless codelength are consumed to describe imperceptible image details
  - architecture
    - PixelCNN++
    - TODO
  - Diffusion models
    - latent variable models
    - $p_\theta(\mathbf{x}_0) = \int p_\theta (\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$
    - $\mathbf{x}_0$: observed examples
    - $\mathbf{x}_{1:T}$: latent variables
      - $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$
    - forward process
      - diffusion process
      - fixed to a Markov Chain that gradually adds Gaussian noise
      - with variance schedule
        - $\beta_1, \beta_2, ..., \beta_T$
    - reverse process
      - starting at $\beta_T$
      - a parameterized Markov chain is assumed


## Remarkable Papers

(2021)

- Score-Based Generative Modeling through Stochastic Differential Equations
  - https://arxiv.org/abs/2011.13456
  - ICLR 2021, Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

- Denoising Diffusion Implicit Models (DDIM)
  - https://arxiv.org/abs/2010.02502
  - Jiaming Song, Chenlin Meng, Stefano Ermon
  - reduce number of steps by using non-Markovian deterministic function

- Improved Denoising Diffusion Probabilistic Models
  - https://arxiv.org/abs/2102.09672
  - Alex Nichol, Prafulla Dhariwal
  - improved DDPM

- Diffusion Models Beat GANs on Image Synthesis
  - https://arxiv.org/abs/2105.05233
  - Prafulla Dhariwal, Alex Nichol
  - Tried various attention sizes.
    - e.g. 32×32, 16×16, 8×8
    - originally only 16×16 was used
  - Increased number of attention heads
  - Class Guidance

- Variational Diffusion Models
  - https://openreview.net/forum?id=2LdBqxc1Yv
  - NeurIPS 2021
  - Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho

(2020)

- DDPM

(2019)


- Generative Modeling by Estimating Gradients of the Data Distribution
  - NeurIPS 2019, Yang Song, Stefano Ermon
  - https://arxiv.org/abs/1907.05600
  - https://youtu.be/m0sehjymZNU
  - "annealed" Langevin dynamics
    - for better data generation
    - aneal down the noise level
    - do Langevin dynamics multiple times
      - using the result samples from the previous step as initial data points
  - score matching
    - definition
      - score of probability density $p(\mathbf{x})$
        - defined to be $\nabla_\mathbf{x}\log{p(\mathbf{x})}$
      - score network
        - a neural network parameterized by $\theta$
        - $s_\theta(\mathbf{x}): \mathbb{R}^D \to \mathbb{R}^D$
        - trained to approximate the score of $p_\text{data}(\mathbf{x})$
      - where
        - $\mathbf{x} \sim p_\text{data}(\mathbf{x})$
        - $D$: data dimension
    - learn the purturbed gradients of the data distribution
      - purturb the data
        - because it's hard to learn high dimensinoal gradients while the data resides on low-dimensional manifolds
    - requires no sampling during training
  - flexible model architectures
  - application
    - image inpainting
  - TODO
    - read from 2.1

(2015)

- Deep unsupervised learning using nonequilibrium thermodynamics
  - https://arxiv.org/abs/1503.03585
  - DDPM
  - acheived bot tractability and flexibility
  - thousands of time steps
  - terminologies
    - quasy-static process
  - TODO Read from 1.2


## Prerequisite

- [FID](./fid.md)

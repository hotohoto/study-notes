# I$^2$SB: Image-to-Image Schrödinger Bridge

- https://arxiv.org/abs/2302.05872
- NVIDIA
- ICML2023
- I2SB
- training requires paired images

## 1 Introduction

## 2 Preliminaries

### 2.1 Score-based generative model (SGM)

### 2.2 Schrödinger bridge (SB)

##### SGM as a special case of SB

## 3 Image-to-image Schrödinger bridge (I$^2$SB)

### 3.1 Mathematical framework

##### Solving SB using SGM framework

##### Theorem 3.1

##### A tractable class of SB

### 3.2 Algorithmic design

##### Sampling proposal for training and generation

##### Proposition 3.3

##### Parameterization & objective

### 3.3 Connection to flow-based optimal transport (OT)

##### Proposition 3.4

##### Corollary 3.5

### 3.4 Comparison to standard conditional diffusion model

## 4 Related work

##### Conditional SGMs (CSGMs)

##### Diffusion-based inverse model (DIM)

## 5 Experiments

### 5.1 Experimental setup

##### Model

##### Baselines

##### Evaluation

### 5.2 Experimental results

##### I$^2$SB surpasses standard CSGM on many tasks

##### I$^2$SB matches DIM without knowing corrupted operators and outperforms standard SB on all tasks

##### I$^2$SB yields interpretable & efficient generation

### 5.3 Discussions

##### Sampling proposals

##### Diffusion vs. OT-ODE

##### General image-to-image translation

##### Comparison to inpainting GANs

##### Limitation

## 6 Conclusion

## References

- Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory
  - https://arxiv.org/abs/2110.11291
  - citations: 36
- Stochastic control liaisons: Richard Sinkhorn meets Gaspard Monge on a Schroedinger bridge
  - https://arxiv.org/abs/2005.10963
  - citations: 64
- Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling
  - https://arxiv.org/abs/2106.01357
  - citations: 111
- Denoising Diffusion Restoration Models
  - https://arxiv.org/abs/2201.11793
  - citations: 92
- A survey of the Schrödinger problem and some of its connections with optimal transport
  - https://arxiv.org/abs/1308.0215
  - citation: 368
- Pseudoinverse-Guided Diffusion Models for Inverse Problems
  - https://openreview.net/forum?id=9_gsMA8MRKQ
  - ICLR 2023

## A Proof

##### Proof of theorem 3.1

TODO

##### Proof of corollary 3.2

TODO

##### How corollary 3.2 reduced to SGM

TODO

##### Proof of proposition 3.3

TODO

##### Proof of proposition 3.4

TODO

##### Proof of corollary 3.5

TODO

## B Introduction to Schrödinger bridge

TODO

## C Experiment details

#### C.1 Additional experimental setup

##### Deblurring and JPEG restoration

##### 4x super-resolution

##### Inpainting

##### Evaluation

##### Palette implementation

### C.1 Additional qualitative results

#### C.3 Additional discussions

##### More ablation study on OT-ODE

##### Other parameterization

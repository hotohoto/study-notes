# Diffusion Models already have a Semantic Latent Space

- https://arxiv.org/abs/2210.10960
- Mingi Kwon, Jaeseok Jeong, Youngjung Uh
- Yonsei Univ.
- Asyrp

## 1 Introduction

## 2 Background

### 2.1 Denoising diffusion probability model

### 2.2 Denoising diffusion implicit model

### 2.3 Image manipulation with CLIP

## 3 Discovering semantic latent space in diffusion models
### 3.1 Problem
##### Theorem 1

### 3.2 Asymmetric reverse process

### 3.3 h-space

### 3.4 Implicit neural directions

## 4 Generative process design

### 4.1 Editing process with Asyrp

### 4.2 Quality boosting with stochastic noise injection

### 4.3 Overall process of image editing

## 5 Experiments

##### Implementation details

### 5.1 Versatility of h-space with Asyrp

### 5.2 Quantitative comparison

### 5.3 Analysis on h-space

##### Homogeneity

##### Linearity

- TODO: Oh, how did they check this? 😮

##### Robustness

##### Consistency across timesteps

## 6 Conclusion

## References

## A Related work

## B More discussion

## C Proof of theorem 1

## D Additional supports for h-space with Asyrp

### D.1 Random perturbation on $\epsilon$-space without Asyrp

### D.2 Robustness and smantics in h-space and $\epsilon$-space with Asyrp

### D.3 Choice of h-space in U-Net

## E Implicit neural directions

## F Quality improvements by non-accelerated sampling with scaled $\Delta h_t$

## G Editing strength and editing flexibility

### G.1 Editing strength and $t_\text{edit}$

### G.2 Editing flexiblity and $t_\text{boost}$

## H Quality boosting

## I Algorithm

## J Training details

### J.1 Loss oefficients

### J.2 Training with random sampling instead of the training datasets

## K Evaluation

### K.1 User study

### K.2 Segmentation consistency and directional CLIP similarity

## L Directions

### L.1 Global direction

### L2. compare three methods

## M Random sampling

## N More samples

### N.1 ImageNet

### N.2 Multi-interpolation

### N.3 More results on all datasets

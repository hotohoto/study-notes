# Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models

- https://arxiv.org/abs/2211.17091

## 1 Introduction

- Discriminator Guidance
  - given a pretrained score model
  - adjust the score for the generated samples to be closer to the real-world data

## 2 Preliminary and related works

$$
\mathbf{x}_{t+1} = \sqrt{1 - \beta_t} \mathbf{x}_t + \beta_t \mathbf{\epsilon}_t
$$


$$
\mathrm{d}\mathbf{x}_t = \mathbf{f}(\mathbf{x}_t, t) \mathrm{d}t + g(t)\mathrm{d}\mathbf{w}_t
\tag{1}
$$

- $t \in [0, T]$
- $\mathbf{f}(\mathbf{x}_t, t)$
  - drift coefficient
- $g(t)$
  - volatility coefficient

$$
\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t)\nabla \log p_r^t (\mathbf{x}_t)\right] \mathrm{d}\bar{t} + g(t)\mathrm{d}\bar{\mathbf{w}}_t
\tag{2}
$$

- $p_r^t$
  - the marginal density of the data forward diffusion of Eq. (1).
- $\mathrm{d}\bar{t}$
  - infinitesimal reverse time $\mathrm{d}t$
- $\mathrm{d}\bar{\mathbf{w}}_t$
  - infinitesimal reverse-time Brownian motion 

$$
\mathcal{L}_\mathbf{\theta} = 1/2 \int_0^T \xi(t) \mathbb{E} \left[ \Vert \mathbf{s}_\mathbf{\theta}(\mathbf{x}_t, t) - \nabla \log p_r^t(\mathbf{x}_t) \Vert_2^2 \right] \mathrm{d}t
\tag{3}
$$

- $\xi$ 
  - the temporal weight



perturbation strategies

- variance exploding SDE
  - $\mathrm{d}\mathbf{x}_t = g(t) \mathrm{d}\mathbf{w}_t$
  - $\mathbf{x} \sim \mathcal{N}(\mathbf{x}_0, \sigma^2(t)\mathbf{I})$

- variance preserving SDE
  - Ornstein-Uhlenbeck process
    - https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

## 3 Refining generative process with discriminator guidance

### 3.1 Correction of pre-trained score estimation

$$
\mathrm{d}\mathbf{x}_t = \left[\mathbf{f}(\mathbf{x}_t, t) - g^2(t)\mathbf{s}_{\mathbf{\theta}_\infty} (\mathbf{x}_t, t)\right] \mathrm{d}\bar{t} + g(t)\mathrm{d}\bar{\mathbf{w}}_t
\tag{4}
$$

- $\mathbf{s} _{\mathbf{\theta} _\infty}$
  - the score network after convergence
  - $\mathbf{\theta}_\infty$
    - would be a local optimum
  - $\mathbf{\theta} _*$
    - global optimum

##### Theorem 1

If $\mathbf{s} _{\mathbf{\theta} _\infty} = \nabla \log \pi(\mathbf{x})$, where $\pi$ is the prior distribution, and log-likelihood $\log p _{\mathbf{\theta} _\infty}$ equals its evidence lower bound $\mathcal{L} _{\mathbf{\theta} _\infty}$, then the reverse time SDE $\mathrm{d}\mathbf{x} _t = \left[\mathbf{f}(\mathbf{x} _t, t) - g^2(t)\nabla \log p _r^t (\mathbf{x} _t)\right] \mathrm{d}\bar{t} + g(t)\mathrm{d}\bar{\mathbf{w}} _t$, coincides with a diffusion process with adjusted score, $\mathrm{d}\mathbf{x} _t = \left[\mathbf{f}(\mathbf{x} _t, t) - g^2(t)(\mathbf{s} _{\mathbf{\theta} _\infty} + \mathbf{c} _{\mathbf{\theta} _\infty}) (\mathbf{x} _t, t)\right] \mathrm{d}\bar{t} + g(t)\mathrm{d}\bar{\mathbf{w}} _t$, for $\mathbf{c} _{\mathbf{\theta} _\infty}(\mathbf{x} _t, t) := \nabla \log {p _r^t(\mathbf{x} _t) \over p _{\mathbf{\theta} _\infty}^t (\mathbf{x} _t)}$.



### 3.2 Discriminator guidance

TODO

### 3.3 Theoretical analysis

TODO

### 3.4 Connection with classifier guidance

TODO

## 4 Related works

## 5 Experiments

### 5.1 A toy 2-dimensional case

### 5.2 Image generation

##### Discriminator network

##### Quantitative analysis

##### Qualitative analysis

### 5.3 Image-2-image translation

## 6 Discussion

## 7 Conclusion



## References

## A Proofs and more analysis

### A.1 Proof of theorem 1

### A.2 Proof of theorem 2

### A.3 Validity of assumption in theorem 1

### A.4 Why $p_{\theta}^t$ is defined as a forward marginal rather than a generative marginal

## B More on Bregman divergences

## C Detail on image-to-image translation

## D Experimental details

### D.1 Training and sampling details

### D.2 More ablation studies

#### D.2.1 Discriminator training

#### D.2.2 After discriminator training

### D.3 Uncurated samples






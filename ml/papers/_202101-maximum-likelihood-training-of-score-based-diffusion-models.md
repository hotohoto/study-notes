# Maximum Likelihood Training of Score-Based Diffusion Models

- https://arxiv.org/abs/2101.09258
- Yang Song, Conor Durkan, Iain Murray, Stefano Ermon
- NeurIPS 2021 (Spotlight)
- Introduces a continuous-time generalization of the ELBO in diffusion probabilistic models
- ScoreFlow

## 1 Introduction

- score-based diffusion models
  - score-based generative models
    - SMLD
    - Score SDE
  - diffusion probabilistic models
    - NET
    - DDPM

## 2 Score-based diffusion models

### 2.1 Diffusion data to noise with an SDE

$$
\mathrm{d}\mathbf{x} = \boldsymbol{f}(\mathbf{x}, t) \mathrm{d}t + g(t) \mathrm{d}\mathbf{w}
\tag{1}
$$

### 2.2 Generating samples with the reverse SDE

$$
\mathrm{d}\mathbf{x} = \left[
\boldsymbol{f}(\mathbf{x}, t) - g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})
\right] \mathrm{d}t
+ g(t) \mathrm{d}\bar{\mathbf{w}}
\tag{2}
$$

$$
\mathcal{J}_\text{SM}(\boldsymbol{\theta}; \lambda(\cdot)) := {1 \over 2} \int_0^T \mathbb{E}_{p_t(\mathbf{x})}
\left[
  \lambda(t)
  \Vert
  	\nabla_\mathbf{x} \log p_t(\mathbf{x})
  	- \boldsymbol{s}_\boldsymbol{\theta}(\mathbf{x}, t)
  \Vert
  _2^2
\right]
\mathrm{d}t
\tag{3}
$$

$$
\mathcal{J}_\text{DSM}(\boldsymbol{\theta}; \lambda(\cdot)) := {1 \over 2} \int_0^T \mathbb{E}_{p_t(\mathbf{x})p_{0t}(\mathbf{x}^\prime|\mathbf{x})}
\left[
  \lambda(t)
  \Vert
  	\nabla_{\mathbf{x}^\prime} \log p_{0t}(\mathbf{x}^\prime|\mathbf{x})
  	- \boldsymbol{s}_\boldsymbol{\theta}(\mathbf{x}^\prime, t)
  \Vert
  _2^2
\right]
\mathrm{d}t
\tag{4}
$$

## 3 Likelihood of score-based diffusion models

$p _\boldsymbol{\theta} ^\text{SDE} (\mathbf{x})$ is given by:
$$
\mathrm{d}\hat{\mathbf{x}} = \left[
\boldsymbol{f}(\hat{\mathbf{x}}, t) - g(t)^2 \boldsymbol{s}_\boldsymbol{\theta}(\hat{\mathbf{x}}, t)
\right] \mathrm{d}t
+ g(t) \mathrm{d}\bar{\mathbf{w}}
\tag{5a}
$$

$$
\hat{\mathbf{x}}_\boldsymbol{\theta} \sim \pi
\tag{5b}
$$

.

ODE corresponding to the SDE in Eq. (1) is given by:
$$
{\mathrm{d}\mathbf{x} \over \mathrm{d}t} = \boldsymbol{f}(\mathbf{x}, t) - {1 \over 2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})
\tag{6}
$$
.

$p _\boldsymbol{\theta} ^\text{ODE} (\mathbf{x})$ corresponding to $p _\boldsymbol{\theta} ^\text{SDE} (\mathbf{x})$ is given by:
$$
{\mathrm{d}\tilde{\mathbf{x}} \over \mathrm{d}t} = \boldsymbol{f}(\tilde{\mathbf{x}}, t) - {1 \over 2} g(t)^2 \boldsymbol{s}_\boldsymbol{\theta}(\tilde{\mathbf{x}}, t)
\tag{7}
$$
.

- It's possible to construct a loss function and train $\boldsymbol{s} _\boldsymbol{\theta}$ with it.
  - Eq. (7) is deterministic and computing $\log p _\boldsymbol{\theta} (\mathbf{x})$ for an individual datapoint $\mathbf{x}$ is tractable.
  - But calling an ODE solver to do so is too expensive. 🫤

## 4 Bounding the likelihood of score-based diffusion models

### 4.1 Bounding the KL divergence with likelihood weighting

##### Theorem 1

##### Corollary 1

(table1)

##### Theorem 2

### 4.2 Bounding the log-likelihood on individual datapoints

##### Theorem 3

### 4.3 Numerical stability

### 4.4 Related work

## 5 Improving the likelihood of score-base diffusion models

### 5.1 Variance reduction via importance sampling

### 5.2 Variational dequantization

### 5.3 Experiments

## 6 Conclusion

## References

- [15] Continuous Normalizing Flow (CNF)
- [19] DDPM
- [23] Score matching
- [33] iDDPM
- [34] Stochastic differential equations: an introduction with applications. Springer
- [43] NET
- [44] SMLD

- [45] Improved Techniques for Training Score-Based Generative Models
- [46] Sliced Score Matching

- [48] Score SDE
- [56] A Connection Between Score Matching and Denoising Autoencoders

## A Proofs

##### Notations

##### Assumptions

##### Theorem 1

##### Theorem 2

##### Theorem 4

##### Remark

##### Theorem 5

##### Theorem 3

## B Numerical stability

## C Experimental details

##### Datasets

##### Model architectures

##### Training

##### Confidence intervals

##### Sample quality


# Variational Diffusion Models

- https://openreview.net/forum?id=2LdBqxc1Yv
- NeurIPS 2021
- Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho
- I'm not sure but "Variational" might mean it's similar to VAE in that they've trained encoders by optimizing the noise schedule
  - https://youtu.be/yR81b3UxgaI?t=4159


## 1 Introduction

- simplify VLB in terms of signal-to-noise ratio

  - can show equivalence between several models (TODO)
  - continuous-time VLB is invariant to the noise schedule except for the signal-to-noise ratio at its endpoint

- improve likelihood

  - optimize the noise schedule
    - minimizing the variance of VLB
    - also leading to faster optimization
  - incorporate Fourier features

- analyze models in terms of bit-back compression scheme

  - show the competitive lossy compression rates

  

## 2 Related work
## 3 Model
### 3.1 Forward time diffusion process

$$
q(\mathbf{z}_t| \mathbf{x}) = \mathcal{N}(\alpha_t \mathbf{x}, \sigma_t^2 \mathbf{I})
\tag{1}
$$

- $t \in [0, 1]$

$$
\operatorname{SNR}(t) = \alpha_t^2 / \sigma_t^2
\tag{2}
$$

- (the variance-preserving diffusion process)
  - $\alpha _t^2 + \sigma _t^2 = 1$
- (the variance-exploding diffusion process)
  - $\alpha_t^2 = 1$

**We assume the variance-preserving diffusion process case in this paper!**

#### (Extra expressions)

Many papers referring to this work express SNR as $\lambda_t = \log (\alpha_t^2 / \sigma_t^2)$ and $\sigma_{t|s}^2$ as below.
$$
\sigma_{t|s}^2 = (1 - e^{\lambda_t - \lambda_s})\sigma_t^2
$$
And here is the derivation.
$$
{\begin{aligned}
\sigma_{t|s}^2 &= \sigma_t^2 - {\alpha_{t}^2 \over \alpha_s^2} \sigma_s^2 \\
&= \left( 1 - {\alpha_t^2 \sigma_s^2\over \sigma_t^2\alpha_s^2} \right) \sigma_t^2\\
&= \left(1 -e^{\lambda_t}\cdot {1 \over e^{\lambda_s}}\right) \sigma_t^2 \\
&= \left(1 - e^{\lambda_t - \lambda_s}\right) \sigma_t^2 \\
\end{aligned}}
$$
given
$$
e^{\lambda_t} = {\alpha_t^2 \over \sigma_t^2}
$$
.

Note that this equation is related to Eq. (32).

### 3.2 Noise schedule

$$
\sigma_t^2 = \operatorname{sigmoid}(\gamma_{\boldsymbol{\eta}}(t))
$$

- $\gamma_{\boldsymbol{\eta}}(t)$
  - a monotonic neural network
  - with parameters $\boldsymbol{\eta}$
  - refer to Appendix H for more details.

$$
\alpha_t^2 = \operatorname{sigmoid}(-\gamma_{\boldsymbol{\eta}}(t))
\tag{4}
$$

$$
\sigma_t^2 = \operatorname{sigmoid}(\gamma_{\boldsymbol{\eta}}(t))
$$

$$
\operatorname{SNR}(t) = \exp(-\gamma_{\boldsymbol{\eta}}(t))
\tag{5}
$$

- Note that $\gamma_{\boldsymbol{\eta}}(t)$ corresponds to $-\lambda_t$ in many other papers referring to this work.

### 3.3 Reverse time generative model

$$
p(\mathbf{x}) = \int_\mathbf{z} p(\mathbf{z}_1) p(\mathbf{x|\mathbf{z}_0}) \prod\limits_{i=1}^T p(\mathbf{z}_{s(i)} | \mathbf{z}_{t(i)})
\tag{6}
$$

- the discrete-time case 
  - $T$
    - number of steps
  
  - $\tau  = 1 / T$
  - $s(i) := (i -1) / T$
  - $t(i) = i/T$
  

$$
p(\mathbf{z}_1) = \mathcal{N}(\mathbf{z}_1; \mathbf{0}, \mathbf{I})
\tag{7}
$$

- $\because q(\mathbf{z} _1|\mathbf{x}) \approx \mathcal{N}(\mathbf{z} _1; \mathbf{0}, \mathbf{I})$
  - for the variance-preserving diffusion process with sufficiently small $\operatorname{SNR}(1)$

$$
p(\mathbf{x}|\mathbf{z}_0) = \prod\limits_{i} p(x_i | z_{0, i})
\tag{8}
$$

- factorized into the product of component wise conditional distributions
- where
  - $p(x _i|z _{0, i}) \propto q(z _{0, i} | x _i)$

$$
p(\mathbf{z}_s | \mathbf{z}_t) := q(\mathbf{z}_s | \mathbf{z}_t, \mathbf{x} = \hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t))
\tag{9}
$$

- See Appendix A where more required derivations for implementation are found.

### 3.4 Noise prediction model and Fourier features

$$
\hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t) = (\mathbf{z}_t - \sigma_t \hat{\boldsymbol{\epsilon}}_\boldsymbol{\theta(\mathbf{z}_t; t)}) / \alpha_t
\tag{10}
$$

- $\hat{\mathbf{x}} _{\boldsymbol{\theta}}(\mathbf{z} _t; t)$
  - the denoising model
- $\hat{\boldsymbol{\epsilon}}_\boldsymbol{\theta}(\mathbf{z}_t; t)$
  - the noise prediction model

(Fourier features)

- $\mathbf{x}$ scaled to the range $[-1, 1]$
- append $nk$ additional channels
  - $\sin(2^n\pi \mathbf{z})$ and $\cos(2^n\pi \mathbf{z})$
  - where
    - $n$  runs over a range of integers $\{n _\min, ..., n _\max\}$
    - $k$ the original number of input channels
- see Appendix C for further details

### 3.5 Variational lower bound

$$
- \log p(\mathbf{x}) \le - \operatorname{VLB}(\mathbf{x})
= \underbrace{D_\text{KL}(q(\mathbf{z}_1 | \mathbf{x}) \Vert p(\mathbf{z}_1))}_\text{Prior loss}
+ \underbrace{\mathbb{E}_{q(\mathbf{z}_0 | \mathbf{x})}[- \log p(\mathbf{x|\mathbf{z}_0})]}_\text{Reconstruction loss}
+ \underbrace{\mathcal{L}_T(\mathbf{x})}_\text{Diffusion loss}
\tag{11}
$$



## 4 Discrete-time model

Let's consider the diffusion loss term $\mathcal{L}_T(\mathbf{x})$ in VLB where the time points are discrete.
$$
\mathcal{L}_T(\mathbf{x})=\sum_{i=1}^T \mathbb{E}_{q\left(\mathbf{z}_{t(i)} \mid \mathbf{x}\right)} D_{K L}\left[q\left(\mathbf{z}_{s(i)} \mid \mathbf{z}_{t(i)}, \mathbf{x}\right) \| p\left(\mathbf{z}_{s(i)} \mid \mathbf{z}_{t(i)}\right)\right]
\tag{12}
$$

- $s := s(i)$
- $t := t(i)$

And we get
$$
\mathcal{L}_T(\mathbf{x})=\frac{T}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), i \sim U\{1, T\}}\left[(\operatorname{SNR}(s)-\operatorname{SNR}(t))\left\|\mathbf{x}-\hat{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}_t ; t\right)\right\|_2^2\right]
\tag{13}
$$

.

- note that $s$ and $t$ are the endpoints of just a single step.
- Refer to appendix E.1 and E.2 for the derivation

Then we get
$$
\mathcal{L}_T(\mathbf{x})=\frac{T}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), i \sim U\{1, T\}}\left[\left(\exp \left(\gamma_{\boldsymbol{\eta}}(t)-\gamma_{\boldsymbol{\eta}}(s)\right)-1\right)\left\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}\left(\mathbf{z}_t ; t\right)\right\|_2^2\right]
\tag{14}
$$

where you may refer to the equations below for the derivation.
$$
{\begin{align}

\mathbf{z}_t &= \alpha_t \mathbf{x} + \sigma_t \boldsymbol{\epsilon}\\

\mathbf{x} &= {1 \over \alpha_t} \mathbf{z}_t - {\sigma_t\over \alpha_t} \boldsymbol{\epsilon} \\

\hat{\mathbf{x}}_\theta(\mathbf{z}_t; t) &= {1 \over \alpha_t} \mathbf{z}_t - {\sigma_t\over \alpha_t} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t; t) \\

\mathbf{x} - \hat{\mathbf{x}}_\theta(\mathbf{z}_t; t) &= - {\sigma_t\over \alpha_t} \boldsymbol{\epsilon} + {\sigma_t\over \alpha_t} \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t; t) \\

\Vert\mathbf{x} - \hat{\mathbf{x}}_\theta(\mathbf{z}_t; t)\Vert_2^2 &= {\sigma_t^2\over \alpha_t^2}\Vert \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t; t)\Vert_2^2 \\
\Vert\mathbf{x} - \hat{\mathbf{x}}_\theta(\mathbf{z}_t; t)\Vert_2^2 &= \operatorname{SNR}(t)^{-1}\Vert \boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t; t)\Vert_2^2 \\
\end{align}}
$$
.

### 4.1 More steps leads to a lower loss

$$
\mathcal{L}_{2T}(\mathbf{x}) \le \mathcal{L}_T(\mathbf{x})
$$

- Refer to appendix F.

## 5 Continuous-time model: $T \to \infty$
$$
\begin{aligned}
\mathcal{L}_{\infty}(\mathbf{x}) & =-\frac{1}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \int_0^1 \operatorname{SNR}^{\prime}(t)\left\|\mathbf{x}-\hat{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}_t ; t\right)\right\|_2^2 d t \\
& =-\frac{1}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}(0,1)}\left[\operatorname{SNR}^{\prime}(t)\left\|\mathbf{x}-\hat{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}_t ; t\right)\right\|_2^2\right]
\end{aligned}
\tag{15-16}
$$

- Refer to appendix E.3.

$$
\mathcal{L}_{\infty}(\mathbf{x})=\frac{1}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}), t \sim \mathcal{U}(0,1)}\left[\gamma_{\boldsymbol{\eta}}^{\prime}(t)\left\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}\left(\mathbf{z}_t ; t\right)\right\|_2^2\right]
\tag{17}
$$

### 5.1 Equivalence of diffusion models in continuous time

Change the variable $t$ in Eq. (17) with $v := \operatorname{SNR}(t)$

- $t = \operatorname{SNR}^{-1}(v)$
- note that $\operatorname{SNR}(t)$ is monotonic and invertible

then we get:
$$
\mathcal{L}_{\infty}(\mathbf{x})=\frac{1}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \int_{\mathrm{SNR}_{\min }}^{\mathrm{SNR}_{\max }}\left\|\mathbf{x}-\tilde{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}_v, v\right)\right\|_2^2 d v
\tag{18}
$$

where

- $\mathbf{z} _v = \alpha_v \mathbf{x} + \sigma _v \boldsymbol{\epsilon}$
- $\tilde{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}, v\right) := \tilde{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}, \operatorname{SNR}^{-1}(v)\right)$
- $\text{SNR} _\text{max} = \operatorname{SNR}(0)$
- $\text{SNR} _\text{min} = \operatorname{SNR}(1)$

.

**takeaway⭐:**

- The only effects the functions $\alpha(t)$ and $\sigma(t)$ have on the diffusion loss is through the values $\operatorname{SNR}(t) = \alpha _t^2 / \sigma _t^2$ at end points $t=0$ and $t=1$.
- The diffusion loss is invariant to the shape of $SNR(t)$ between $t=0$ and $t=1$.

### 5.2 Weighted diffusion loss

$$
\mathcal{L}_{\infty}(\mathbf{x}, w)=\frac{1}{2} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})} \int_{\mathrm{SNR}_{\min }}^{\mathrm{SNR}_{\max }} w(v)\left\|\mathbf{x}-\tilde{\mathbf{x}}_{\boldsymbol{\theta}}\left(\mathbf{z}_v, v\right)\right\|_2^2 d v
\tag{19}
$$



### 5.3 Variance minimization

- use a low discrepancy sampler for $t$ 
  - to reduce the variance of the loss estimator
  - refer to appendix I.1 for more details
- optimize the schedule shape between its endpoints
  - to minimize the variance of the loss estimator
    - empirically
      - to train faster
      - to get the most out of including Fourier features
  - by using the Monte Carlo method
  - refer to appendix I.2 for more details
- optimize the schedule endpoints
  - w.r.t. the VLB

## 6 Experiments
- augmentation regularization
  - (not always though)
  - methods
    - random flips
    - 90-degree
    - color channel swapping

### 6.1 Likelihood and samples

### 6.2 Ablations

- Fourier features play an important role

### 6.3 Lossless compression
## 7 Conclusion
## References

## A Distribution details

### A.1
$$
q(\mathbf{z}_t | \mathbf{z}_s) = \mathcal{N}(\alpha_{t|s} \mathbf{z}_s, \sigma^2_{t|s}\mathbf{I})
\tag{20}
$$
- $0 \le s \lt t \le 1$ 

$$
\alpha_{t|s} = \alpha _t / \alpha _s
\tag{21}
$$

$$
\sigma^2_{t|s} = \sigma^2 _t - \alpha _{t|s}^2 \sigma^2 _s
\tag{22}
$$
- Note that in the variance-preserving diffusion process,
  - $\alpha^2 + \sigma^2 = 1$
  - $\alpha _{t|s}^2 + \sigma _{t|s} ^2 = 1$

.

### A.2
$$
q(\mathbf{z}_s| \mathbf{z}_t, \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_Q(\mathbf{z}_t,\mathbf{x}; s, t), \sigma^2_Q(s, t)\mathbf{I})
\tag{23}
$$

$$
\sigma^2_Q(s, t) = \sigma^2_{t|s} \sigma^2_s / \sigma^2_t
\tag{24}
$$

$$
\boldsymbol{\mu}_Q(\mathbf{z}_t,\mathbf{x}; s, t)
= {1 \over \alpha_{t|s}} (\mathbf{z}_t + \sigma_{t|s}^2) \nabla_{\mathbf{z}_t} \log q(\mathbf{z}_t| \mathbf{x})
= {\alpha_{t|s} \sigma_s^2 \over \sigma_t^2} \mathbf{z}_t + {\alpha_{s} \sigma_{t|s}^2 \over \sigma_t^2} \mathbf{x}
\tag{25}
$$
### A.3
$$
p(\mathbf{z}_s | \mathbf{z}_t) := q(\mathbf{z}_s|\mathbf{z}_t, \mathbf{x} = \hat{\mathbf{x}}_\theta(\mathbf{z}_t; t))
\tag{26}
$$

$$
p(\mathbf{z}_s|\mathbf{z}_t) =
\mathcal{N}(\mathbf{z}_s; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t; s, t), \sigma^2_Q(s, t)\mathbf{I})
\tag{27}
$$

$$
\begin{align}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t; s, t)
&= {\alpha_{t|s} \sigma_s^2 \over \sigma_t^2} \mathbf{z}_t + {\alpha_{s} \sigma_{t|s}^2 \over \sigma_t^2} \hat{\mathbf{x}}(\mathbf{z}_t; t)\\
&= {1 \over \alpha_{t|s}} \mathbf{z}_t - {\sigma_{t|s}^2 \over \alpha_{t|s} \sigma_t} \hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t) \\
&= {1 \over \alpha_{t|s}} \mathbf{z}_t + {\sigma_{t|s}^2 \over \alpha_{t|s}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{z}_t; t)
\tag{28}
\end{align}
$$
where
$$
\hat{\boldsymbol{\epsilon}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t)
= (\mathbf{z}_t - \alpha_t \hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t)) / \sigma_t
\tag{29}
$$

and
$$
\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{z}_t; t)
= (\alpha_t \hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{z}_t; t) - \mathbf{z}_t) / \sigma_t^2
\tag{30}
$$
### A.4 Further simplication of $p(\mathbf{z} _s| \mathbf{z} _t)$

$$
\boldsymbol{\mu}_\boldsymbol{\theta}(\mathbf{z}; s, t)
= {\alpha_s \over \alpha_t}(\mathbf{z}_t + \sigma_t(e^{\gamma_\boldsymbol{\eta}(s) - \gamma_\boldsymbol{\eta}(t)} - 1) \hat{\boldsymbol{\epsilon}}_\boldsymbol{\theta}(\mathbf{z}_t; t))
\tag{31}
$$

$$
\sigma_Q^2(s,t)
= \sigma_s^2 (1 -e^{\gamma_\boldsymbol{\eta}(s) - \gamma_\boldsymbol{\eta}(t)})
\tag{32}
$$

Plug in these two equations into Eq. (27) then we get
$$
\mathbf{z}_s = \sqrt{\alpha_s^2 \over \alpha_t^2}(\mathbf{z}_t - (1 -e^{\gamma_\boldsymbol{\eta}(s) - \gamma_\boldsymbol{\eta}(t)})\hat{\boldsymbol{\epsilon}}_\boldsymbol{\theta(\mathbf{z}_t; t)}) + \sqrt{(1 - \alpha_s^2)(1 -e^{\gamma_\boldsymbol{\eta}(s) - \gamma_\boldsymbol{\eta}(t)})}\boldsymbol{\epsilon}
\tag{33}
$$
where

- $\alpha _t ^2 = \operatorname{sigmoid}(-\gamma_\boldsymbol{\eta}(t))$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

.

## B Hyperparameters, architecture and implementation details

### B.1 Model and implementation

- no internal downsampling/upsampling within U-Net

### B.2 Settings for each dataset

## C Fourier features for improved fine scale prediction

> Since our reconstruction model $p(\mathbf{x}|\mathbf{z} _0)$ given in Equation 8 is weak, the burden of modeling these fine scale details falls on our denoising diffusion model $\hat{\mathbf{x}}_\boldsymbol{\theta}$. In initial experiments, we found that the denoising model had a hard time accurately modeling these details. At larger noise levels, the latents $z_t$ follow a smooth distribution due to the added Gaussian noise, but at the smallest noise levels the
> discrete nature of 8-bit image data leads to sharply peaked marginal distributions $q(z_t)$.

So they added Fourier features to capture the fine scale details of the data.
$$
\begin{align}
f_{i,j,k}^n &= \sin(z_{i,j,k} 2^n\pi), \\
g_{i,j,k}^n &= \cos(z_{i,j,k} 2^n\pi)
\end{align}
\tag{34}
$$

## D As a SDE

$$
d\mathbf{z}_t = [f(t)\mathbf{z}_t - g^2(t)s_{\boldsymbol{\theta}}]dt + g(t)d\mathbf{W}_t
\tag{35}
$$

$$
f(t) = {d\log\alpha_t \over dt} \\
g^2(t) = {d \sigma^2(t) \over dt} - 2 {d \log \alpha_t \over dt} \sigma^2(t)
\tag{36}
$$

The derivation of Eq. (36) can be found by evaluating the expression below.
$$
\lim\limits_{\Delta t \to 0}{\Delta\mathbf{z}_t \over \Delta t}
= \lim\limits_{\Delta t \to 0}{\mathbf{z}_{t^\prime} - \mathbf{z}_t\over \Delta t}
= \lim\limits_{\Delta t \to 0}{\alpha_{t^\prime|t}\mathbf{z}_t + \sigma_{t^\prime | t}\boldsymbol{\epsilon} - \mathbf{z}_t\over \Delta t}
$$

- where
  - $T \to \infty$
    - the number of steps
  - $\Delta t = t^\prime - t$
    - defined as the time difference for a single step
    - $T \Delta t = 1$
  - $\sigma(t) = T \sigma _t = {\sigma _t \over \Delta t}$
  - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
  - $\lim\limits_{\Delta t \to 0}\sqrt{\Delta t}\boldsymbol{\epsilon} = {d \mathbf{W}_t \over dt}$

## E Derivation of the VLB estimators

### E.1 Discrete-time VLB

### E.2 Estimator of $\mathcal{L}_T(\mathbf{x})$

### E.3 Infinite depth ($T \to \infty$)

## F Influence of the number of steps $T$ on the VLB

 ## G Equivalence of diffusion specifications

## H Implementation of monotonic neural net noise schedule $\gamma _\eta(t)$

- $\operatorname{SNR}(t) = \exp(-\gamma _{\boldsymbol{\eta}}(t))$
- $\tilde{\gamma}_{\boldsymbol{\eta}}(t) = l_1(t) + l_3(\phi(l_2(l_1(t))))$
  - It's going to be postprocessed for variance minimization.
  - Refer to Appendix I.2.
- $\phi$
  - the sigmoid function
- $l_1$
  - a linear layer
  - has a single output
  - weights are restricted to be positive
- $l_2$
  - a linear layer
  - has 1024 outputs
  - weights are restricted to be positive
- $l_3$
  - a linear layer
  - has a single output
  - weights are restricted to be positive

## I Variance minimization

### I.1 Low-discrepancy sampler

- $t^i = \mod(u _0 + i/k, 1)$
  - $k$: batch size
  - $i \in \{1, ..., k\}$
  - $u_0 \sim U[0, 1]$

### I.2 Optimizing the noise schedule w.r.t. the variance of the diffusion loss
$$
\gamma_{\boldsymbol{\eta}}(t) = \gamma_0 + (\gamma_1 - \gamma_0){\tilde{\gamma}_{\boldsymbol{\eta}}(t) - \tilde{\gamma}_{\boldsymbol{\eta}}(0) \over \tilde{\gamma}_{\boldsymbol{\eta}}(1) - \tilde{\gamma}_{\boldsymbol{\eta}}(0)}
\tag{59}
$$

- $\gamma _0 = - \log(\operatorname{SNR} _\max)$
- $\gamma _1 = - \log(\operatorname{SNR} _\min)$


$$
\mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ \mathcal{L}_\infty^\text{MC}(\mathbf{x}, w, \gamma_{\boldsymbol{\eta}})^2\right]
= \mathcal{L}_\infty (\mathbf{x}, w)^2
+ \operatorname{Var}_{t, \boldsymbol{\epsilon}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, w, \gamma_{\boldsymbol{\eta}})\right]
$$

$$
\mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ \nabla_{\boldsymbol{\eta}}\mathcal{L}_\infty^\text{MC}(\mathbf{x}, w, \gamma_{\boldsymbol{\eta}})^2\right]
= \nabla_{\boldsymbol{\eta}}\operatorname{Var}_{t, \boldsymbol{\epsilon}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, w, \gamma_{\boldsymbol{\eta}})\right]
\tag{60}
$$

$$
{d \over d \boldsymbol{\eta}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, \gamma_{\boldsymbol{\eta}})^2\right] =
{d \over d \text{SNR}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, \text{SNR})^2\right]
{d \over d \boldsymbol{\eta}}\left[\operatorname{SNR}(\boldsymbol{\eta})\right]
\tag{61}
$$

- ${d \over d \boldsymbol{\eta}}$ seems to be equivalent to $\nabla_\boldsymbol{\eta}$. 🤔
  - https://onlinelibrary.wiley.com/doi/pdf/10.1002/0471705195.app3

$$
{d \over d \text{SNR}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, \text{SNR})^2\right] =
2 {d \over d \text{SNR}}\left[\mathcal{L}_\infty^\text{MC}(\mathbf{x}, \text{SNR})\right] \odot \mathcal{L}_\infty^\text{MC}(\mathbf{x}, \text{SNR})
\tag{62}
$$

- $\odot$
  - elemetwise multiplication
  - not sure why it's needed. 🤔

## J Numerical stability

There are numerically stable methods used in this work.

- `expm1(x)`
  - $= e^x - 1$
- `softplus(x)`
  - $= \log(e^x - 1)$

## K Comparison to DDPM and NCSN objectives

- DDPM, NCSNv1, iDDPM use $L_\text{simple}$ where $w(v) = 1 / \gamma^\prime(t)$ which is not fixed as $1$.
- In this case, also  the shape of schedule as well as $\text{SNR}_\text{min}$ and $\text{SNR}_\text{max}$ can change the diffusion loss.

## L Consistency

- Let's say we have an EBM model to train.
- Even though we may reweight the loss function of the denoising score matching (DSM) method
- we'll learn the score function.

## M Additional samples from our models

## N Lossless compression

TODO 🤔

## O Density estimation on additional data sets

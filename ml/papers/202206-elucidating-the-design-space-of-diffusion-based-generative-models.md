# Elucidating the Design Space of Diffusion-Based Generative Models



- NVIDIA
- https://arxiv.org/abs/2206.00364
- contributions
  - Analysis on the freedom of design space and heuristic improvement
  - Higher order Runge-Kutta method for the deterministic sampling
  - Found non-leaking augmentation was helpful
- taken together achieved SOTA



## 1 Introduction

## 2 Expressing diffusion models in a common framework

- (notations)
  - (common)
    - ${\boldsymbol{y}_0, \boldsymbol{y}_1, \cdots, \boldsymbol{y}_Y}$
      - training data set
      - $Y$
        - number of samples in the training data set
    - $p_\text{data} (\boldsymbol{x}) = {1 \over Y}\sum\limits_{i=1}^Y \delta(\boldsymbol{x} - \boldsymbol{y}_i)$
      - observed data distribution
    - $\sigma_\text{data}$
      - standard deviation of the data
    - $p(\boldsymbol{x}; \sigma) = {1 \over Y} \sum\limits_{i=1}^Y \mathcal{N}(\boldsymbol{x}; \boldsymbol{y}_i, \sigma^2 \mathrm{I}) = p_\text{data}(\boldsymbol{x}) *\mathcal{N}(0, \sigma^2 \mathrm{I})$
      - perturbed data distribution
      - $*$ is the convolution operator
    - $p_t(\mathbf{x}) = s(t)^{-d}\left[{p_{\text {data }} * \mathcal{N}\left(\mathbf{0}, \sigma(t)^{2} \mathbf{I}\right)}\right](\boldsymbol{x}/s(t)) = s(t)^{-d} p(\boldsymbol{x}/s(t); \sigma(t))$
      - marginal distribution
    - $\boldsymbol{n} \sim \mathcal{N}(\boldsymbol{0}, \sigma^2 \mathrm{I})$
      - noise
    - $N$
      - number of ODE solver iterations
    - ${t_i}$
      - what kind of times we want to take
    - $\sigma(t)$
      - how much noise we want to add depending on $t$
    - $s(t)$
      - how much fast we want to move the mean to the final mean
  - (related to real images) 🖼️
    - $i=N$
    - $t_N=0$
    - $\boldsymbol{x}_N$
    - $\sigma_N = 0$
  - (related to noisy images) 🌫️
    - $i=0$
    - $\boldsymbol{x}_0 \sim p(\boldsymbol{0}, \sigma_\text{max} \mathrm{I})$
    - $\sigma_0 = \sigma_\text{max}$
    - $t_0 = 1$
- ODE formulation
  - $\mathrm{d} \boldsymbol{x}= -\dot{\sigma}(t) \sigma(t) \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x} ; \sigma(t)) \mathrm{d} t$
  - a probability flow ODE
  - appendix B.2
- denoising score matching

  - $\mathbb{E}_{\boldsymbol{y} \sim p_{\text {data }}} \mathbb{E}_{\boldsymbol{n} \sim \mathcal{N}\left(\mathbf{0}, \sigma^2 \mathbf{I}\right)}\|D(\boldsymbol{y}+\boldsymbol{n} ; \sigma)-\boldsymbol{y}\|_2^2 $

  - $\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x} ; \sigma)=(D(\boldsymbol{x} ; \sigma)-\boldsymbol{x}) / \sigma^2$

    - Note that in DDPM and the other literatures
    - $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$
      - $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = - {\mathbf{\epsilon} \over \sqrt{1 - \bar{\alpha}_t}}$

  - appendix B.3
- time-dependent signal scaling

  - $s(t)$
    - scale schedule
    - $\boldsymbol{x} = s(t) \hat{\boldsymbol{x}}$
  - $\mathrm{d} \boldsymbol{x}=\left[\dot{s}(t) \boldsymbol{x} / s(t)-s(t)^2 \dot{\sigma}(t) \sigma(t) \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x} / s(t) ; \sigma(t))\right] \mathrm{d} t$
  - appendix B.2
- solution by descretization
- putting it together

## 3 Improvements to deterministic sampling

- Descretizaiton and higher-order integrators
  - Deterministic sampling using Heun's 2nd order method with arbitrary $\sigma(t)$ and $s(t)$
- Trajectory curvature and noise schedule
- Discussion

## 4 Stochastic sampling (no big improvement)

- Background
- Our stochastic sampler
- Practical consideration
- Evaluation

## 5 Preconditioning and training

- Loss weighting and sampling
- Augmentation regularization



## A. Additional results

## B. Derivation of formulas

- B.1 Original ODE/SDE formulation from previous work
- B.2 Our ODE formulation (Eq. 1 and Eq. 4)
- B.3 Denoising score matching (Eq. 2 and Eq. 3)
- B.4  Evaluating our ODE in practice (Algorithm 1)
- B.5 Our SDE formulation (Eq. 6)
  - B.5.1 Generating the marginals by heat diffusion
    - To solve heat equations, it requires various methods depending on the boundary conditions
      - for $x \in [-\infty, \infty]$ case, Fourier transform method is required
    - there's only one solution for the heat equation (?), and we want it to represent the marginalized distributions?? seems not true
  - B.5.2 Derivation of our SDE
    - 
- B.6 Our preconditioning and training (Eq. 8)

## C. Reframing previous methods in our framework

- C.1 Variance preserving formulation
  - C.1.1 VP sampling
  - C.1.2 VP preconditioning
  - C.1.3 VP training
  - C.1.4 VP practical considerations
- C.2 Variance exploding formulation
  - C.2.1 VE sampling in theory
  - C.2.2 VE sampling in practice
  - C.2.3 VE preconditioning
  - C.2.4 VE training
  - C.2.5 VE practical considerations
- C.3 Improved DDPM and DDIM
  - C.3.1 DDIM ODE formulation
  - C.3.2 iDDPM time step discretization
  - C.3.3 iDDPM preconditioning and training
  - C.3.4 iDDPM practical considerations

## D. Further analysis of deterministic sampling

- D.1 Truncation error analysis and choice of discretization parameters
- D.2 General family of 2nd order Runge-Kutta variants

## E. Further results with stochastic sampling

- E.1 Image degradation due to excessive stochastic iteration
- E.2 Optimal stochasticity parameters

## F. Implementation details

- F.1 FID calculation
- F.2 Augmentation regularization
- F.3 Training configurations
- F.4 Network architectures
- F.5 Licenses

- analysis on sampling and an alternative stochastic sampler



## Extra notes



### Source code

- train.py
  - inner_model = K.config.make_model(config)
    - ImageDenoiserModelV1
      - .mapping_cond
        - nn.Linear
      - .mapping
        - MappingNet
          - nn.Linear
          - nn.GELU
      - .proj_in
        - nn.Conv2d
      - .unet
      - .proj_out
        - nn.Conv2d
          - may include an extra channel for the variance
            - the mean of which can be considered as its variance
  - model = K.config.make_denoiser_wrapper(config)(inner_model)
    - Denoiser
      - forward(input, sigma)
        - returns inner_model(input * c_in, sigma) * c_out + input * c_skip
        - c_skip
          - to use input as it is
        - c_out
        - c_in
    - DenoiserWithVariance(Denoiser)
      - loss(input, noise, sigma)
        - input: real data
        - noise ~ N(0, 1)
  - model_ema = deepcopy(model)
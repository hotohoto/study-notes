# Diffusion based models

## Outlooks on important diffusion papers

| Publication        | Year | Title (Alias)                                                        | FID*  | FID** | Remark                                                                             | Architecture                  |
| ------------------ | ---- | -------------------------------------------------------------------- | ----- | ----- | ---------------------------------------------------------------------------------- | ----------------------------- |
| JMLR               | 2005 | Estimation of Non-Normalized Statistical Models by Score Matching    |       |       | Introduce score matching                                                           |                               |
| Neural Computation | 2011 | A Connection Between Score Matching and Denoising Autoencoders       |       |       | Connect score matching and denoising AEs                                           |                               |
| ICLR               | 2014 | Auto-Encoding Variational Bayes (VAE)                                |       |       |                                                                                    |                               |
| (PMLR)             | 2015 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics (NET) |       |       | Introduce probabilistic diffusion models                                           | CNN                           |
| NeurIPS            | 2019 | (SMLD aka NCSN)                                                      | 25.32 |       | Use score matching with Langevin dynamics                                          | RefineNet, CondInstanceNorm++ |
| NeurIPS            | 2020 | (DDPM)                                                               | 3.17  |       | Simplify the loss function and investigate the connection to NCSN                  | UNet, self-attn, GN           |
| (PMLR)             | 2021 | (Improved DDPM)                                                      | 2.94  |       | TODO                                                                               |                               |
| ICLR               | 2021 | (NCSN++ / DDPM++)                                                    | 2.2   |       | TODO Connect SMLD and DDPM with SDE                                                |                               |
| ICLR               | 2021 | (DDIM)                                                               |       |       | TODO Faster deterministic sampling within non-Markovian diffusion process          |                               |
| NeurIPS            | 2021 | Diffusion Models Beat GANs on Image Synthesis                        |       |       | Do more experiments and find some insights                                         |                               |
| NeurIPS            | 2021 | Variational Diffusion Models                                         |       |       | TODO                                                                               |                               |
| ICLR               | 2022 | (Analytic-DPM)                                                       |       |       | TODO                                                                               |                               |
|                    | 2022 | Elucidating the Design Space of Diffusion-Based Generative Models    | 1.97  | 1.79  | Simplify diffusion model frameworks for heuristic but better sampling and training |                               |
|                    | 2022 | Improving Diffusion Model Efficiency Through Patching                |       |       |                                                                                    | PatchedUNet                   |

- (*) Unconditional FID on CIFAR-10
- (**) Conditional FID on CIFAR-10

## Papers

(2022)

- Retrieval-Augmented Diffusion Models
  - RDM
  - https://arxiv.org/abs/2204.11824
  - method
    - keep explicit memory/database for known data samples
    - retrieve visually similar samples to the training instance and encode them using CLIP
    - use those embedding when generating a sample

- High-Resolution Image Synthesis with Latent Diffusion Models
  - CVPR 2022
  - LDM
  - https://arxiv.org/abs/2112.10752
  - (stable diffusion is based on this)
  - auto-encoder
    - training
      - perceptual loss
      - a patch-based adversarial object
      - latent space
        - 2D
          - the same as the original number of dimensions
          - U-Net can be used by the following latent diffusion model
        - regularization
          - KL-reg
          - VQ-reg
            - was better than KL-reg
            - embedded in the decoder
    - fixed when training the latent diffusion model
  - latent diffusion model
  - cross-attention layers
    - enables multi-modal training
  - References
    - https://github.com/CompVis/latent-diffusion
    - https://github.com/huggingface/diffusers
    - https://github.com/CompVis/stable-diffusion
  - TODO
    - how to do inpainting/super resolution?
    - what is the actual shape of latent values

- Elucidating the Design Space of Diffusion-Based Generative Models
  - NVIDIA
  - https://arxiv.org/abs/2206.00364
  - SOTA but a bit heuristic
  - contents
    - 2 Expressing diffusion models in a common framework
      - ODE formulation
      - denoising score matching
      - time-dependent signal scaling
      - solution by descretization
      - putting it together
    - 3 Improvements to deterministic sampling
      - Descretizaiton and higher-order integrators
        - Deterministic sampling using Heun's 2nd order method with arbitrary $\sigma(t)$ and $s(t)$
      - Trajectory curvature and noise schedule
      - Discussion
    - 4 Stochastic sampling (no big improvement)
      - Background
      - Our stochastic sampler
      - Practical consideration
      - Evaluation
    - 5 Preconditioning and training
      - Loss weighting and sampling
      - Augmentation regularization
  - analysis on sampling and an alternative stocahstic sampler
  - TODO readme

- Hierarchical Text-Conditional Image Generation with CLIP Latents
  - https://openai.com/dall-e-2/
  - [DALLE2](./dall-e-2.md)

- Diffusion-Based Representation Learning
  - https://arxiv.org/abs/2105.14257
  - TODO

- Gotta Go Fast When Generating Data with Score-Based Models
  - https://arxiv.org/abs/2105.14080
  - ICLR 2022
  - TODO readme

- Subspace Diffusion Generative Models
  - MIT
  - https://arxiv.org/abs/2205.01490

(2021)

- Diffusion Models Beat GANs on Image Synthesis
  - https://arxiv.org/abs/2105.05233
  - Prafulla Dhariwal, Alex Nichol
  - found a better architecture by ablations
  - architecture
    - U-Net
      - deeper rather than wider
        - deeper means more residual blocks for each resolution
      - BigGAN residual blocks
      - multi-head attention
        - various attention sizes.
          - 32×32, 16×16, 8×8
          - (originally only 16×16 was used)
        - more number of attention heads
          - 1 -> 4
      - AdaGN
        - $\operatorname{AdaGN}(h, y) = y_s \operatorname{GroupNorm}(h) + y_b$
          - where
            - $y = [y_s, y_b] = \operatorname{dense}(\text{(class embedding)}, \text{t})$
  - class guidance for conditional generation
    - can trade off diversity for fidelity
      - but only when it comes to the labled datasets
  - ADM
  - ADM-G
  - ADM-U

- Image Super-Resolution via Iterative Refinement
  - Called SR3
  - https://arxiv.org/abs/2104.07636
  - Enlarge an image by bicubic interpolation and anti-aliasing
  - and use it as a condition which is an extra input chnannel for a diffusion model
  - metrics used
    - Peak Signal-to-Noise Ratio (PSNR)
    - Structural Similarity Index Map (SSIM)
    - Mean Opinion Score (MOS) (?)
    - 2-alternative forced-choice (?)
    - fool rate (?)
  - References
    - https://m.blog.naver.com/mincheol9166/221771426327

- Variational Diffusion Models
  - https://openreview.net/forum?id=2LdBqxc1Yv
  - NeurIPS 2021
  - Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho

- ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models
  - ICCV 2021
  - https://arxiv.org/abs/2108.02938
  - Iterative Latent Variable Refinement (ILVR)

- Score-Based Generative Modeling through Stochastic Differential Equations
  - https://arxiv.org/abs/2011.13456
  - ICLR 2021, Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
  - both SMLD and DDPM can be seen in the perspective of SDE
  - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-noise-conditioned-score-networks-ncsn
  - contributions
    - unified framework generalizing NCSNs and DDPMs
    - flexible sampling
      - general-purpose SDE solvers
      - predictor-corrector samplers
      - deterministic samplers
    - controllerble generation without retraining
  - contents
    - 3 Score-based generative modeling with SDEs
      - 3.1 Perturbing data with SDEs
        - $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt+g(t) d\mathbf{w}$
          - $\mathbf{w}$
            - the standard wiener process
          - $\mathbf{f}(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$
            - the drift coefficient of $\mathbf{x}(t)$
          - $g: \mathbb{R} \to \mathbb{R}$
            - the diffusion coefficient of $\mathbf{x}(t)$
          - This SDE has a unique solution as long as the coefficients are globally Lipschitz in both state and time.
        - $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt + \mathbf{G}(\mathbf{x}, t) d\mathbf{w}$
          - (for more general coefficients)
        - $p_{st}(\mathbf{x}(t) \vert \mathbf{x}(s))$
          - transition kernel from $\mathbf{x}(s)$ to $\mathbf{x}(t)$
      - 3.2 Generating samples by reversing the SDE
        - $d\mathbf{x}=\left[\mathbf{f}(\mathbf{x}, t)-g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt+g(t) d\bar{\mathbf{w}}$
          - $\bar{\mathbf{w}}$
            - a standard Wiener process when time flows backwards fro $T$ to $0$
          - $dt$
            - an infinifestimal negative timestep
        - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt+\mathbf{G}(\mathbf{x}, t) d\overline{\mathbf{w}}$
      - 3.3 Estimating scores for the SDE
        - $\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \mathbb{E}_t\left\{\lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t) \mid \mathbf{x}(0)}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)-\nabla_{\mathbf{x}(t)} \log p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0))\right\|_2^2\right]\right\}$
          - Assumes the drift and diffusion coefficient of an SDE are affine
        - $\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \mathbb{E}_t\left\{\lambda(t) \mathbb{E}_{\mathbf{x}(0)} \mathbb{E}_{\mathbf{x}(t)} \mathbb{E}_{\mathbf{v} \sim p_{\mathbf{v}}}\left[\frac{1}{2}\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t)\right\|_2^2+\mathbf{v}^{\top} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) \mathbf{v}\right]\right\}$
      - 3.4 Examples: VE, VP SDEs and beyond
        - VE SDE
          - $d\mathbf{x}=\sqrt{\frac{d\left[\sigma^2(t)\right]}{dt}} ~d\mathbf{w}$
          - Variance exploding SDE
          - A continuous generalization of SMLD
          - e.g.
            - SMLD
              - $d\mathbf{x}=\sigma_{\min }\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^t \sqrt{2 \log \frac{\sigma_{\max }}{\sigma_{\min }}} d\mathbf{w}, \quad t \in(0,1]$
              - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0))=\mathcal{N}\left(\mathbf{x}(t) ; \mathbf{x}(0), \sigma_{\min }^2\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^{2 t} \mathbf{I}\right), \quad t \in(0,1]$
            - NCSN++
        - VP SDE
          - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)} d\mathbf{w}$
          - Variance preserving SDE
          - A continuous generalization of DDPM
          - e.g.
            - DDPM
              - $d\mathbf{x}= -\frac{1}{2} (\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })) \mathbf{x} dt + \sqrt{\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })}d\mathbf{w}, \quad t \in(0,1]$
              - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \mathbf{I}-\mathbf{I} e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right), \quad t \in[0,1]$
            - DDPM++
        - Sub-VP SDE
          - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)\left(1-e^{-2 \int_0^t \beta(s) ds}\right)} d\mathbf{w}$
          - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \left[1 - e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right]^2\mathbf{I}\right), \quad t \in[0,1]$
          - The variance is always bounded by the VP SDE at every intermediate time step
          - seems better in terms of likelihoods
          - especially for low resolution images
    - 4 Solving the reverse SDE
      - 4.1 General-purpose numerical SDE solvers
        - Euler-Maruyama method
        - stochastic Runge-Kutta method
        - ancestral sampling
          - the same as the DDPM sampler
          - just a special discretization of the reverse-time VP SDE
        - reverse diffusion samplers
          - discretize the reverse-time SDE in the same way as the forward one
          - TODO read Appendix E/F
      - 4.2 Predictor-corrector samplers
        - TODO
        - read Appendix G
      - 4.3 Probability flow and connection to neural ODEs
        - Probability flow
          - $d\mathbf{x}=\left[\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$
          - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\boldsymbol{\top}}\right]-\frac{1}{2} \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt$
          - Its trajactories shares the same marginal probability densities $\{p_t(\mathbf{x})\}_{t=0}^T$ as the SDE
          - Derived via Fokker-Planck equations
          - The Fokker-Planck equation can be derived from a general SDE
            - using Itô's lemma and integration by parts
            - references
              - [Fokker Planck Equation Derivation](https://youtu.be/MmcgT6-lBoY)
              - [Ito's calculus](https://en.wikipedia.org/wiki/It%C3%B4_calculus)
        - Exact likelihood computation
          - Now we can calculate likelihood in a deterministic way
          - how?
            - 1 sample $\mathbf{x}(T)$
            - 2 obtain $\mathbf{x}(t)$ by solving the probability flow ODE
            - 3 calculate the log likelihood
              - $\log p_0(\mathbf{x}(0))=\log p_T(\mathbf{x}(T))+\int_0^T \nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}(\mathbf{x}(t), t) dt$
          - $Computing $\nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}$ is expensive
            - estimate it with Skilling-Hutchinson trace estimator
              - $\nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}(\mathbf{x}, t)=\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\boldsymbol{\epsilon}^{\top} \nabla \tilde{\mathbf{f}}_\theta(\mathbf{x}, t) \boldsymbol{\epsilon}\right]$
        - Manipulating latent representations
          - interpolation
          - temperature rescaling (by modifying norm of embedding)
        - Uniquely identifiable encoding
          - How?
            - Given a outcome from the dataset
            - Obtain $\mathbf{x}(T)$ using the probability flow ODE
          - For the same inputs, Model A and Model B provide encodings that are close in every dimension
            - despite having different model architectures and training runs
        - Efficient sampling
          - How?
            - Given
              - $d\mathbf{x}=\mathbf{f}(\mathbf{x}, t) dt+\mathbf{G}(t) d\mathbf{w}$
            - Discretize it and we get
              - $\mathbf{x}_{i+1}=\mathbf{x}_i+\mathbf{f}_i\left(\mathbf{x}_i\right)+\mathbf{G}_i \mathbf{z}_i, \quad i=0,1, \cdots, N-1$
            - Plug the coefficients to the probability flow ODE
              - $\mathbf{x}_i=\mathbf{x}_{i+1}-\mathbf{f}_{i+1}\left(\mathbf{x}_{i+1}\right)+\frac{1}{2} \mathbf{G}_{i+1} \mathbf{G}_{i+1}^{\top} \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, i+1\right), \quad i=0,1, \cdots, N-1$
          - examples
            - SMLD
              - $\mathbf{x}_i=\mathbf{x}_{i+1}+\frac{1}{2}\left(\sigma_{i+1}^2-\sigma_i^2\right) \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, \sigma_{i+1}\right), \quad i=0,1, \cdots, N-1$
            - DDPM
              - $\mathbf{x}_i=\left(2-\sqrt{1-\beta_{i+1}}\right) \mathbf{x}_{i+1}+\frac{1}{2} \beta_{i+1} \mathbf{s}_{\theta^*}\left(\mathbf{x}_{i+1}, i+1\right), \quad i=0,1, \cdots, N-1$
      - 4.4 Architecture improvements
        - TODO
    - 5 Controllable generation
      - TODO
    - Appendix
      - D.2
        - TODO check out Chen et al 2020 and understand eq 39
      - E Reverse diffusion sampling
        - TODO
      - F Ancestral sampling for SMLD models
        - TODO
      - G Predictor-Corrector samplers
        - TODO
      - H Architecture improvements
        - TODO
        - H.1 Settings for architecture exploration
        - H.2 Results on CIFAR-10
        - H.3 High resolution images
      - I Controllable generation
        - TODO
        - I.1 Class-conditional sampling
        - I.2 Imputation
        - I.3 Colorization
        - I.4 Solving general inverse problems
  - TODO
    - check out NCSN++ architecuture
      - It's said that worth looking at
  - resources
    - https://www.math.snu.ac.kr/~syha/Lecture-4.pdf
    - https://youtu.be/yqF1IkdCQ4Y?t=3459


- Denoising Diffusion Implicit Models (DDIM)
  - ICLR 2021
  - https://arxiv.org/abs/2010.02502
  - Jiaming Song, Chenlin Meng, Stefano Ermon
  - speeded up diffusion model sampling
    - generates high quality samples with much fewer steps
  - introduced a deterministic generative process
    - enables meaningful interpolatation in the latent variable

- Improved Denoising Diffusion Probabilistic Models
  - PMLR 2021
  - https://arxiv.org/abs/2102.09672
  - Alex Nichol, Prafulla Dhariwal
  - improved DDPM
  - a cosine-based variance schedule
    - a heuristic suggestion
  - learn $\sigma_t$ by interpolating between the upper bound and the lower bound
    - mix $L_\text{VLB}$ and $L_\text{simple}$ when calculating
      - $L_\text{VLB}$
        - original loss function from which $L_\text{simple}$ has been originated
        - use the learnt $\sigma_t$
      - $L_\text{simple}$
        - only the exponent term is used ignoring the $\Sigma_\theta$ part
  - give more weights on the time component of the loss
    - use importance sampling
      - sample some t values more where the loss component of which is bigger
      - $L_\text{VLB}$ decreases in a smoothed curve


(2020)

- Denoising Diffusion Probabilistic Models (DDPM)
  - https://arxiv.org/abs/2006.11239
  - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  - what they did
    - simplified the loss function of the diffusion models
      - (mainly from the perspective of the comparison to SMLD)
      - fix $\beta$ as constants
      - fix $\Sigma_\theta(\mathbf{x}_t, t)$ as untrained time-dependent constants
      - replace weights terms with 1 so that focus more on large $t$ when the task is more difficult
  - contributions
    - showed that high quality image generation is possible by DDPM
    - showed connections/equivalence to/with denoising score matching when
      - training with multiple noise levels
      - sampling with annealed Langevin dynamics
        - https://en.wikipedia.org/wiki/Langevin_dynamics
    - training on a "weighted" variational bound
    - seeing DDPM as progressive decoding in the context of lossy decompression
  - notations
    - $\mathbf{x}_0$: observed examples
    - $\mathbf{x}_{1:T}$: latent variables
      - $\mathbf{x}_T \sim \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$
    - $p_\theta$
      - stochastic process described by our model
    - $q$
      - stochastic process we want to approximate
    - $\mathbf{z}, \mathbf{z_t}, ...$
      - each of them $\sim \mathcal{N}(\mathbf{0}, I)$
    - $\beta_t = 1 - \alpha_t$
    - $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
  - Diffusion models
    - latent variable models
    - $p_\theta(\mathbf{x}_0) = \int p_\theta (\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$
    - forward process
      - diffusion process
      - fixed to a Markov Chain that gradually adds Gaussian noise
      - with variance schedule
        - $\beta_1, \beta_2, ..., \beta_T$
      - $\mathbf{x_t} = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\mathbf{z}_t$
    - reverse process
      - starting at $\beta_T$
      - a parameterized Markov chain is assumed
      - $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mathbf{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t I)$
        - we need to be able to calculate these two terms analytically to calculate the loss
          - $\tilde{\mathbf{\mu}}(\mathbf{x}_t, \mathbf{x}_0)$
            - $= \frac{\sqrt{\alpha_{t}}\left(1-\bar{\alpha}_{t-1}\right)}{1-\bar{\alpha}_{t}} \mathbf{x}_{t}+\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1-\bar{\alpha}_{t}} \mathbf{x}_{0}$
            - $= \frac{1}{\sqrt{\alpha_{t}}}\left(\mathbf{x}_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}} \mathbf{z}_{t}\right)$
          - $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t}-1}{1-\bar{\alpha}_{t}} \beta_{t}$
      - loss function
        - $\mathbb{E}_{q}[\underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{T} \mid \mathbf{x}_{0}\right) \| p_{\theta}\left(\mathbf{x}_{T}\right)\right)}_{L_{T}}+\sum_{t=2}^{T} \underbrace{D_{\mathrm{KL}}\left(q\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}, \mathbf{x}_{0}\right) \| p_{\theta}\left(\mathbf{x}_{t-1} \mid \mathbf{x}_{t}\right)\right)}_{L_{t-1}}-\underbrace{\log p_{\theta}\left(\mathbf{x}_{0} \mid \mathbf{x}_{1}\right)}_{L_{0}}]$
  - architecture
    - U-Net based on a Wide ResNet (as in PixelCNN++)
    - group normalization
    - self-attention block at the 16 x 16 resolution
  - etc
    - they've got relatively poor log likelihoods in spite of high quality of generated samples
      - the majority of the models' lossless codelength are consumed to describe imperceptible image details
        - for the best sample from CIFAR-10, $L_1 + L_2 + \cdots + L_T < L_0$ was found.
        - they considered $L_1 + L_2 + \cdots + L_T$ as rate and $L_0$ as distortion.
    - lossy compression
      - [rate-distortion theory](information-theory.md#rate-distortion-theory)
    - rao blackwell theorem (? TODO)
      - https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem
      - https://www.youtube.com/results?search_query=rao+blackwell+theorem+
      - https://arxiv.org/abs/2101.01011

(2019)

- Generative Modeling by Estimating Gradients of the Data Distribution
  - NeurIPS 2019, Yang Song, Stefano Ermon
  - https://arxiv.org/abs/1907.05600
  - contribution
    - tried to use score matching along with Langevin dynamics for generative modeling
  - Noise Conditional Score Network (NCSN)
    - which is Score Matching with Langevin Dynamics (SMLD)
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
  - pros
    - flexible model architectures
  - application
    - image inpainting
  - architecture
    - RefineNet (a variant of U-Net)

- Sliced Score Matching: A Scalable Approach to Density and Score Estimation
  - Song et al.
  - https://arxiv.org/abs/1905.07088
  - cons
    - requires four times more computations than denoising score matching
  - code
    - https://github.com/ermongroup/ncsn/blob/7f27f4a16471d20a0af3be8b8b4c2ec57c8a0bc1/losses/sliced_sm.py#L121
    - $\mathbf{v}^T \nabla_\mathbf{x}(s_\theta) \mathbf{v} = \mathbf{v} \cdot \nabla_\mathbf{x}(s_\theta \cdot \mathbf{v})$
    - sums values from the current batch

- Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters
  - ICLR 2019
  - https://arxiv.org/abs/1810.00440
  - TODO readme

(2015)

- Deep unsupervised learning using nonequilibrium thermodynamics
  - PMLR 2015
  - https://arxiv.org/abs/1503.03585
  - contributions
    - Introduced probabilistic diffusion models
      - pros
        - flexible model structure
        - easy sampling
        - easy multiplication with other distribution in order to compute a posterior
        - easy to evaluate the model log likelihood
  - Diffusion models
    - (the same as the model description in DDPM paper's background section)
    - (but with slightly different notations)
  - Model probability
    - we can calculate a probability value for a given data sample according to the trained model
    - $p(\mathbf{x}^{(0)}) = \int d\mathbf{x}^{1 \cdots T} q(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}) p(\mathbf{x}^{(T)})\prod\limits_{t=1}^T {p(\mathbf{x}^{(t-1)}\mid\mathbf{x}^{(t)}) \over q(\mathbf{x}^{(t)}\mid\mathbf{x}^{(t-1)})}$
      - For infinifestimal $\beta$ the forward and reverse distribution over trajectories can be made identical.
        - Then only a single sample from $q(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)})$ is required to evalute the integral above.
        - This corresponds to the case of a quasy-static process.
  - Model multiplied by another distribution
    - useful when we compute posterior probability or represent inpainting with mathematical notations
    - if we can represent the modified reverse process with respect to the original reverse process, it becomes tractable
    - $\tilde{p}(\mathbf{x}^{(t)}) = {1 \over \tilde{Z}_t} p(\mathbf{x}^{(t)})r(\mathbf{x}^{(t)})$
    - $\tilde{p}(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t+1)}) = {1 \over \tilde{Z}_t(\mathbf{x}^{(t+1)})}p(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t+1)})r(\mathbf{x}^{(t)})$
  - Entropy of reverse process
    - defines upper/lower bound for each reverse process step in terms of entropy
    - $H_q(\mathbf{X}^{(t)} \mid \mathbf{X}^{(t-1)}) + H_q(\mathbf{X}^{(t-1)} \mid \mathbf{X}^{(0)}) - H_q(\mathbf{X}^{(t)} \mid \mathbf{X}^{(0)}) \le H_q(\mathbf{X}^{(t-1)} \mid \mathbf{X}^{(t)}) \le H_q(\mathbf{X}^{(t)} \mid \mathbf{X}^{(t-1)})$
    - note that
      - they are known distribution assumed
      - so they can be represented with some parameters in a closed form

(2011)

- A Connection Between Score Matching and Denoising Autoencoders
  - Neural Computation 2011
  - https://ieeexplore.ieee.org/abstract/document/6795935
  - Connected score matching and denoising AEs
  - Introduced a scalable loss function called denoising score matching
    - $L_\text{DSM} = \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}\mid \mathbf{x})p_\text{data}(\mathbf{x})}[\Vert \mathbf{s}_\mathbf{\theta}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid \mathbf{x}) \Vert_2^2]$

(2007)

- The Communication Complexity of Correlation
  - IEEE 2007
  - TODO

(2005)

- Estimation of Non-Normalized Statistical Models by Score Matching
  - JMLR 2005
  - Aapo Hyvärinen
  - https://www.jmlr.org/papers/v6/hyvarinen05a.html
  - contributions
    - Introduced score matching and how to train the score
  - Note that "score" here is the gradient of the log density with respect to the data vector
    - not with resepect to the parameters like in statistics usually

## References

- [Sample quality metrics](./sample-quality-metrics.md)
- [What are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - DDPM, Improved DDPM, DDIM, ...

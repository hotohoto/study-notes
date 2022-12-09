# Diffusion based models

## Outlooks on important diffusion papers

| Publication        | Year | Title (Alias)                                                | FID*  | FID** | Remark                                                       | Architecture                  |
| ------------------ | ---- | ------------------------------------------------------------ | ----- | ----- | ------------------------------------------------------------ | ----------------------------- |
| JMLR               | 2005 | Estimation of Non-Normalized Statistical Models by Score Matching |       |       | Introduce score matching                                     |                               |
| Neural Computation | 2011 | A Connection Between Score Matching and Denoising Autoencoders |       |       | Connect score matching and denoising AEs                     |                               |
| ICLR               | 2014 | Auto-Encoding Variational Bayes (VAE)                        |       |       |                                                              |                               |
| (PMLR)             | 2015 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics (NET) |       |       | Introduce probabilistic diffusion models                     | CNN                           |
| NeurIPS            | 2019 | (SMLD aka NCSN)                                              | 25.32 |       | Use score matching with Langevin dynamics                    | RefineNet, CondInstanceNorm++ |
| NeurIPS            | 2020 | (DDPM)                                                       | 3.17  |       | Simplify the loss function and investigate the connection to NCSN | UNet, self-attn, GN           |
| (PMLR)             | 2021 | (Improved DDPM)                                              | 2.94  |       | TODO                                                         |                               |
| ICLR               | 2021 | (NCSN++ / DDPM++)                                            | 2.2   |       | TODO Connect SMLD and DDPM with SDE                          |                               |
| ICLR               | 2021 | (DDIM)                                                       |       |       | Faster deterministic sampling by introducing non-Markovian diffusion process which has the same loss function |                               |
| NeurIPS            | 2021 | Diffusion Models Beat GANs on Image Synthesis                |       |       | Do more experiments and find some insights                   |                               |
| NeurIPS            | 2021 | Variational Diffusion Models                                 |       |       | TODO                                                         |                               |
| ICLR               | 2022 | (Analytic-DPM)                                               |       |       | TODO                                                         |                               |
|                    | 2022 | Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction |       |       | Define an alternative reverse trajectory that uses an unconditional diffusion model and also converges to a conditional data sample. At the same time skip reverse steps by using the condition information. |                               |
|                    | 2022 | Elucidating the Design Space of Diffusion-Based Generative Models | 1.97  | 1.79  | Analyze diffusion model theory in a common framework and suggest  heuristics for better sampling and training |                               |
|                    | 2022 | Improving Diffusion Model Efficiency Through Patching        |       |       |                                                              | PatchedUNet                   |

- (*) Unconditional FID on CIFAR-10
- (**) Conditional FID on CIFAR-10

## TODO

- lossy compression
- why Sub-VP SDE

## Papers

Retrieval-Augmented Diffusion Models
- RDM
- https://arxiv.org/abs/2204.11824
- method
  - keep explicit memory/database for known data samples
  - retrieve visually similar samples to the training instance and encode them using CLIP
  - use those embedding when generating a sample

### High-Resolution Image Synthesis with Latent Diffusion Models

- CVPR 2022
- LDM
- https://arxiv.org/abs/2112.10752
- (stable diffusion is based on this)
- latent space
  - 2D
    - the same as the original number of dimensions
    - So U-Net can be used by LDM
- encoder/decoder
  - auto-encoder based perceptual compression
  - fixed when training the latent diffusion model
  - training
    - perceptual loss
    - architecture
      - VAE
          - KL-reg
            - give a little ($10^{-6}$) KL-penalty toward standard normal on learned latent
          - VQ-reg
            - vector quantization
            - was better than KL-reg
            - embedded in the decoder
      - VQ-GAN
        - stage1
            - VQ-VAE based encoder/decoder/codebook training
        - stage2
            - training an autoregressive transformer model
                - to translate the latent vector using the code book as vocabulary
            - trained separately with the encoder/decoder/code-book freezed
            - a patch-based adversarial object
- latent diffusion model
- cross-attention layers
  - enables multi-modal training
  - Not only decoder but also encoder of U-Net has attention
- References
  - https://github.com/CompVis/latent-diffusion
  - https://github.com/huggingface/diffusers
  - https://github.com/CompVis/stable-diffusion
- TODO
  - how to do inpainting?
  - how to do super resolution?

### Elucidating the Design Space of Diffusion-Based Generative Models

- NVIDIA
- https://arxiv.org/abs/2206.00364
- contributions
  - Analysis on the freedom of design space and heuristic improvement
  - Higher order Runge-Kutta method for the deterministic sampling
  - Found non-leaking augmentation was helpful
- taken together achieved SOTA
- contents
  - 1 Introduction
  - 2 Expressing diffusion models in a common framework
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
  - Appendix
    - A. Additional results
    - B. Derivation of formulas
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
    - C. Reframing previous methods in our framework
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
    - D. Further analysis of deterministic sampling
      - D.1 Truncation error analysis and choice of discretization parameters
      - D.2 General family of 2nd order Runge-Kutta variants
    - E. Further results with stochastic sampling
      - E.1 Image degradation due to excessive stochastic iteration
      - E.2 Optimal stochasticity parameters
    - F Implementation details
      - F.1 FID calculation
      - F.2 Augmentation regularization
      - F.3 Training configurations
      - F.4 Network architectures
      - F.5 Licenses
- analysis on sampling and an alternative stochastic sampler
- TODO readme

### Improving Diffusion Models for Inverse Problems using Manifold Constraints

- https://arxiv.org/abs/2206.00941

- NeurIPS 2022

- Projection Onto Convex Sets (POCS)

  - projects point $\boldsymbol{x}$ to the plane $\boldsymbol{y} = \boldsymbol{H}\boldsymbol{x}$
  - $\boldsymbol{y}$
    - measurement
  - the conventional extra update we did in ILVR or CCDF which we called measurement consistency step
  - it dates back to further in the old days.

- Manifold Constrained Gradient (MCG) 

  - $- \alpha {\frac {\partial} {\partial \boldsymbol{x}_i}} \Vert \boldsymbol{W}(\boldsymbol{y - \boldsymbol{H}\hat{\boldsymbol{x}}_0})\Vert_2^2$
    - $i$
      - current reverse diffusion step index
    - $\boldsymbol{x}_i$
      - the estimate of the current reverse diffusion step $i$
      - it might by noisy when $i > 0$
    - $\hat{\boldsymbol{x}}_0$
      - direct estimate of $\boldsymbol{x}_0$

- Theorem 1

  - $\frac{\partial}{\partial \boldsymbol{x}_i}\left\|\boldsymbol{W}\left(\boldsymbol{y}-\boldsymbol{H} \hat{\boldsymbol{x}}_0\right)\right\|_2^2=-2 \boldsymbol{J}_{Q_i}^T \boldsymbol{H}^T \boldsymbol{W}^T \boldsymbol{W}\left(\boldsymbol{y}-\boldsymbol{H} \hat{\boldsymbol{x}}_0\right) \in T_{\hat{\boldsymbol{x}}_0} \mathcal{M}$
    - $Q_i$
      - $\boldsymbol{x}_i \mapsto \hat{\boldsymbol{x}}_0$
      - projection to the estimated data manifold
    - $\boldsymbol{J}$
      - Jacobian of $Q_i$
  - MCG is in the tangent plane of $\hat{\boldsymbol{x}}_0$'s manifold space

- basic idea

  - points in the noisy data manifolds are mostly in a dense area due to high dimensionality
  - our score network will be trained mostly on these dense area
  - conventional POCS is not enough in that it will move intermediate estimates into a subspace where the score network is not trained on
  - So let's do a gradient step which would move the current estimate around the current noisy manifold toward $\boldsymbol{y} = \boldsymbol{H}\boldsymbol{x}$

- resources

  - https://youtu.be/mjfinYDJTMc
  - https://sang-yun-lee.notion.site/Improving-Diffusion-Models-for-Inverse-Problems-using-Manifold-Constraints-01e82afda989428e8b5faad1c3bbebf2 

  

### Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction

- https://arxiv.org/abs/2112.05146
- [2022 CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.html)

#### Notations

- $0 \le t_0 \le 1$
- $N' = N t_0 \lt N$
  - the time step where we want to start the reverse phase from
- $\tilde{\boldsymbol{x}}_0$
  - a ground truth sample
- $\boldsymbol{x}_0$
  - initial estimate
- $\mathcal{E}_0 = \Vert \boldsymbol{x}_0 - \tilde{\boldsymbol{x}}_0\Vert^2$
  - estimation error between the initial estimate and the real sample
- $\bar{\mathcal{E}}_{N'} = \mathbb{E}\Vert \boldsymbol{x}_{N'} - \tilde{\boldsymbol{x}}_{N'}\Vert^2$
  - expected estimate error between the generated samples and the ground truth at the step $N'$ 
- $\bar{\mathcal{E}}_{j, r}$
  - expected estimate error between the generated samples following the reverse trajectory and the ground truth at the step j 

####  Definition

- contraction mapping $\boldsymbol{f}$
  - $: \mathbb{R}^n \to \mathbb{R}^n$ is defined as $\Vert\boldsymbol{f}(\boldsymbol{x}) - \boldsymbol{f}(\boldsymbol{x})\Vert \le \lambda \Vert \boldsymbol{x} - \boldsymbol{y} \Vert$
    - where $0 \le \lambda \lt 1$


####  Reverse diffusion

- (13): standard reverse diffusion step
- (14): alternating step

$$
\boldsymbol{x}'_{i-1} = \boldsymbol{f}(\boldsymbol{x}_i, i) + g(\boldsymbol{x}_i, i)\boldsymbol{z}_i\tag{13}
$$

$$
\boldsymbol{x}_{i-1} = \boldsymbol{A}\boldsymbol{x}'_{i-1} + \boldsymbol{b}\tag{14}
$$

####  Key ideas

- $\boldsymbol{x}_i$ converges to $\tilde{\boldsymbol{x}}_i$ exponentially where (13), (14) altogether is a contraction mapping.
  - The estimation error at the final step has exponentially contracting upper bound. (Theorem 1.)
    - $\bar{\mathcal{E}}_{0, r} \le {2C\tau \over 1 - \lambda^2} + \lambda^{2N'}\bar{\mathcal{E}}_{N'}$
    - where
      - $\tau = {\operatorname{Tr}({\boldsymbol{A}^T\boldsymbol{A}}) \over n}$
      - $\lambda$
        - constant depending on the standard diffusion
      - $C$
        - constant depending on the standard diffusion
- There exist a minimum $N'$ which is a shortcut path. (Theorem 2.)
  - such that
    - $\bar{\mathcal{E}}_{0, r} \le \mu\mathcal{E}_0$
    - For any $0 \lt \mu \le 1$

####  Aapplications

$\boldsymbol{A} = (I - \boldsymbol{P})$

$b = \hat{\boldsymbol{x}}$

- super resolution
  - $\boldsymbol{P}$: Gaussian blur
- image inpainting
  - $\boldsymbol{P}$: measurement mask
- ...

### Hierarchical Text-Conditional Image Generation with CLIP Latents

- https://openai.com/dall-e-2/
- [DALLE2](./dall-e-2.md)

### Diffusion-Based Representation Learning

- https://arxiv.org/abs/2105.14257
- TODO

### Gotta Go Fast When Generating Data with Score-Based Models

- https://arxiv.org/abs/2105.14080
- ICLR 2022
- high-order methods were significantly slower (6-8 times)
- TODO readme

### Subspace Diffusion Generative Models

- MIT
- https://arxiv.org/abs/2205.01490

### Classifier-Free Diffusion Guidance

- Jonathan Ho, Tim Salimans
- https://arxiv.org/abs/2207.12598
- training
  - loss
    - $\nabla_\theta\Vert\mathbf{\epsilon}_\theta(\mathbf{z_\lambda}, \mathbf{c}) - \mathbf{\epsilon}\Vert^2$
      - randomly replace $c$ with null token with the probability of $p_\text{uncond}$ 
        - $p_\text{uncond} \in [0.1, 0.5]$ in the paper
- sampling
  - $\tilde{\mathbf{\epsilon}}_t = (1 + w)\mathbf{\epsilon}_\theta(\mathbf{z}, \mathbf{c}) - w \mathbf{\epsilon}_\theta(\mathbf{z})$
    - $w \in [0, 4]$ in the paper
  - $\tilde{\mathbf{x}}_t = {(\mathbf{z}_{t} - \sigma_{\lambda_t}\tilde{\mathbf{\epsilon}}_t) / \alpha_{\lambda_t}}$
  - $z_{t+1}$
    - $ \sim \mathcal{N}(\tilde{\mathbf{\mu}}_{\lambda_{t+1}}(\mathbf{z}_t, \tilde{\mathbf{x}}_t), (\tilde{\sigma}_{\lambda_{t+1} \vert \lambda_t}^2)^{1-v}(\sigma_{\lambda_{t+1} \vert \lambda_t}^2)^v)$
      - if $t \lt T$
    - $= \tilde{\mathbf{x}}_t$
      - if $t = T$
- TODO
  - Read math more carefully. (it requires to recap many preceding papers.)

### Diffusion Models Beat GANs on Image Synthesis

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

### SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models

https://arxiv.org/abs/2104.14951

- Learn the difference between SR and LR rather than SR directly
- Use (pretrained) LR encoder which is of Residual in Residual Dense Block (RRDB)
- extensions
  - interpolation between two possible SR images from a LR image
  - Content fusion
    - e.g. Use image 1 for the face as it is and image 2 for the eyes as it can be slightly modified
- limitations
  - no codes
  - no ablation study

### Image Super-Resolution via Iterative Refinement

- Called SR3
- https://arxiv.org/abs/2104.07636
- DDPM conditioned on an LR image
- from the network's perspective, the condition is extra input channels
- clarified PSNR and SSIM do not reflect human preference
  - (not reliable when it comes to low resolution inputs and the magnification factor is large)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Map (SSIM)
- the other super resolution metrics
  - consistency
    - by comparing real and fake images in the low resolution
  - classifier accuracy
    - on the first 1000 images
    - Top-1 Error
    - Top-5 Error
  - cascaded generations
    - FID
- proposed a human evaluation
  - a 2-alternative forced-choice (2AFC) paradigm
  - "Which of the two images is a better high quality version of the low resolution image in the middle?"
- they found cascade models can perform the same with less iterative steps (?)
  - 64 x 64 -> 256 x 256 -> 1024 x 1024
- $\boldsymbol{x}$
  - LR image
- $\boldsymbol{y}_t$
  - latent random variables and the observed random variable in DDPM process
- References
  - https://m.blog.naver.com/mincheol9166/221771426327

### Variational Diffusion Models

- https://openreview.net/forum?id=2LdBqxc1Yv
- NeurIPS 2021
- Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho

### ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models

- ICCV 2021
- https://arxiv.org/abs/2108.02938
- Iterative Latent Variable Refinement (ILVR)
- user controllability
  - N
    - downsampling factor
  - conditioning steps
- applications
  - paintings to real images
  - real images to paintings
  - cats to dogs
  - scribbles to modify an image

### Score-Based Generative Modeling through Stochastic Differential Equations

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
      - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt+\mathbf{G}(\mathbf{x}, t) d\bar{\mathbf{w}}$
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
            - their optimal architecture for the VE SDE
          - NCSN++ cont.
            - trained the score network using the continuous loss function in Eq. (7)
      - VP SDE
        - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)} d\mathbf{w}$
        - Variance preserving SDE
        - A continuous generalization of DDPM
        - e.g.
          - DDPM
            - $d\mathbf{x}= -\frac{1}{2} (\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })) \mathbf{x} dt + \sqrt{\bar{\beta}_{\min } + t (\bar{\beta}_{\max } - \bar{\beta}_{\min })}d\mathbf{w}, \quad t \in(0,1]$
            - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \mathbf{I}-\mathbf{I} e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right), \quad t \in[0,1]$
          - DDPM++
            - their optimal architecture for the VP SDE
      - Sub-VP SDE
        - $d\mathbf{x}=-\frac{1}{2} \beta(t) \mathbf{x} dt+\sqrt{\beta(t)\left(1-e^{-2 \int_0^t \beta(s) ds}\right)} d\mathbf{w}$
        - $p_{0 t}(\mathbf{x}(t) \mid \mathbf{x}(0)) =\mathcal{N}\left(\mathbf{x}(t) ; e^{-\frac{1}{4} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-\frac{1}{2} t \bar{\beta}_{\min }} \mathbf{x}(0), \left[1 - e^{-\frac{1}{2} t^2\left(\bar{\beta}_{\max }-\bar{\beta}_{\min }\right)-t \bar{\beta}_{\min }}\right]^2\mathbf{I}\right), \quad t \in[0,1]$
        - The variance is always bounded by the VP SDE at every intermediate time step
          - (less variance than of VP SDE)
        - seems better in terms of likelihoods
        - especially for low resolution images
  - 4 Solving the reverse SDE
    - 4.1 General-purpose numerical SDE solvers
      - Euler-Maruyama method
      - stochastic Runge-Kutta method
      - **ancestral sampling**
        - the same as the DDPM sampler
        - just a special discretization of the reverse-time VP/VE SDE
          - can be applied to SMLD
      - **reverse diffusion samplers**
        - discretize the reverse-time SDE in the same way as the forward one
    - 4.2 Predictor-corrector samplers
      - prediction
        - the solution of a numerical SDE solver
      - correction
        - score based MCMC
    - 4.3 **Probability flow** and connection to neural ODEs
      - Probability flow
        - $d\mathbf{x}=\left[\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$
        - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\frac{1}{2} \nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\boldsymbol{\top}}\right]-\frac{1}{2} \mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right\} dt$
        - Its trajactories shares the same marginal probability densities $\{p_t(\mathbf{x})\}_{t=0}^T$ with the original SDE
        - Derived via Fokker-Planck equations
          - We find the Fokker-Planck equation from diffusion SDE
          - We find ODE that generates the same Fokker-Plank equation
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
        - Computing $\nabla \cdot \tilde{\mathbf{f}}_{\boldsymbol{\theta}}$ is expensive
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
      - NCSN++
        - Finite Impulse Response (FIR) upsampling/downsampling
        - rescaled skip connection
        - BigGAN-type residual blocks
        - 4 residual blocks per resolution (instead of 2)
        - use residual for input
        - no progressive growing architecture for output
      - DDPM++
        - rescaled skip connection
        - BigGAN-type residual blocks
        - 4 residual blocks per resolution (instead of 2)
        - use residual for input
      - NCSN++ cont. deep
        - best FID on CIFAR-10
      - DDPM++ cont. deep
        - best NLL on CIFAR-10
      - (naming)
        - ++
          - for the best architecture they found
        - deep
          - for  doubling the network depth
        - cont.
          - for training the score network via the continuous version of the loss
      - more details
        - 1.3M iterations
        - save one checkpoint per 50k iterations
        - FID
          - on 50k samples
        - batch_size=128
        - anti-aliasing based on Finite Impulse Response (FIR)
        - StyleGAN-2 hyper parameters
        - progressive architecture implemented according to StyleGAN-2
        - The Exponential Moving Average (EMA) rate
          - 0.999 for VE
          - 0.9999 for VP
      - High resolution images
        - 1024 x 1024 CelebA-HQ
        - NCSN++
        - batch_size=8
        - 2.4 M iterations
        - EMA rate
          - 0.9999
  - 5 Controllable generation
    - $d\mathbf{x}=\left\{\mathbf{f}(\mathbf{x}, t)-\nabla \cdot\left[\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top}\right]-\mathbf{G}(\mathbf{x}, t) \mathbf{G}(\mathbf{x}, t)^{\top} \nabla_{\mathbf{x}} \log p_t(\mathbf{x}\vert \mathbf{y})\right\} dt+\mathbf{G}(\mathbf{x}, t) d\bar{\mathbf{w}}$
      - $\nabla_\mathbf{x}\log p_t(\mathbf{x}\vert \mathbf{y}) = \nabla_\mathbf{x}\log p_t(\mathbf{x}) + \nabla_\mathbf{x}\log p_t(\mathbf{y}\vert \mathbf{x})$
    - with an auxiliary classifier
      - Class-conditional sampling
    - without an auxiliary classifier
      - Imputation
        - separate $\mathbf{x}$, $\mathbf{f}$, $\mathbf{G}$ into known and unknown dimensions
      - Colorization
        - transform the image to map the gray-scale image to a separate channel (known dimension)
          - a 3x3 orthogonal matrix is used to decouple known data dimensions
        - learn the other two channels (unknown dimension)
        - reverse-transform the learned image
    - Solving general inverse problem
      - similar to the class-conditional sampling problem
        - requires assumptions
          - $p(\mathbf{y}(t) \vert \mathbf{y})$ is tractable
          - $p_t(\mathbf{x}(t) \vert \mathbf{y}(t), \mathbf{y}) \approx p_t(\mathbf{x}(t) \vert \mathbf{y}(t))$
            - notes
              - for small $t$, these two are almost the same
              - for large $t$, the discrepancy matters less for the final sample
- resources
  - https://www.math.snu.ac.kr/~syha/Lecture-4.pdf
  - https://youtu.be/yqF1IkdCQ4Y?t=3459

### Denoising Diffusion Implicit Models (DDIM)

- ICLR 2021
- https://arxiv.org/abs/2010.02502
- Jiaming Song, Chenlin Meng, Stefano Ermon
- speeded up diffusion model sampling
  - generates high quality samples with much fewer steps
- introduced a deterministic generative process
  - enables meaningful interpolatation in the latent variable
- TODO

### Improved Denoising Diffusion Probabilistic Models

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

### Denoising Diffusion Probabilistic Models (DDPM)

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
  - Rao Blackwell theorem (? TODO)
    - https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem
    - https://www.youtube.com/results?search_query=rao+blackwell+theorem+
    - https://arxiv.org/abs/2101.01011

### Generative Modeling by Estimating Gradients of the Data Distribution

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

### Sliced Score Matching: A Scalable Approach to Density and Score Estimation

- Song et al.
- https://arxiv.org/abs/1905.07088
- cons
  - requires four times more computations than denoising score matching
- code
  - https://github.com/ermongroup/ncsn/blob/7f27f4a16471d20a0af3be8b8b4c2ec57c8a0bc1/losses/sliced_sm.py#L121
  - $\mathbf{v}^T \nabla_\mathbf{x}(s_\theta) \mathbf{v} = \mathbf{v} \cdot \nabla_\mathbf{x}(s_\theta \cdot \mathbf{v})$
  - sums values from the current batch

Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters
- ICLR 2019
- https://arxiv.org/abs/1810.00440
- TODO readme

### Deep unsupervised learning using nonequilibrium thermodynamics

- PMLR 2015
- aka NET
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
            - we put the sampled $\mathbf{x}^{(t)}$ into the model to get $p(\mathbf{x}^{(t-1)}\mid\mathbf{x}^{(t)})$.
          - This corresponds to the case of a quasy-static process.
        - But this is still a heavy operation in that we need to calculate the output of the neural net T times.
          - There seems no code within the official GitHub repository but they calculate the lower bound of the log likelihood at uniformly sampled t values.
    - code analysis
      - LogLikelihood.do()
        - do()
          - print_stats([model.cost(batch) for batch in dataloader])
        - print_stats()
          - mean
          - std
          - stderr
      - model.cost()
        - model.cost_single_t(X_noiseless)
          - X_noisy, t, mu_posterior, sigma_posterior = generate_forward_diffusion_sample(X_noiseless)
          - mu, sigma = get_mu_sigma(X_noisy, t)
            - (go through one reverse step from $x^(t)$ to get the mu and the sigma of $x^{(t-1)}$)
            - Z = mlp.apply(X_noisy)
            - temporal_readout(Z, t)
          - negL_bound = get_negL_bound(mu, sigma, mu_posterior, sigma_posterior)
            - 
          - return negL_bound
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

### A Connection Between Score Matching and Denoising Autoencoders

- Neural Computation 2011
- https://ieeexplore.ieee.org/abstract/document/6795935
- Connected score matching and denoising AEs
- Introduced a scalable loss function called denoising score matching
  - $L_\text{DSM} = \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}\mid \mathbf{x})p_\text{data}(\mathbf{x})}[\Vert \mathbf{s}_\mathbf{\theta}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid \mathbf{x}) \Vert_2^2]$

### The Communication Complexity of Correlation

- IEEE 2007
- TODO

### Estimation of Non-Normalized Statistical Models by Score Matching

- JMLR 2005
- Aapo Hyvärinen
- https://www.jmlr.org/papers/v6/hyvarinen05a.html
- contributions
  - Introduced score matching and how to train the score
- Note that "score" here is the gradient of the log density with respect to the data vector
  - not with resepect to the parameters like in statistics usually

### References

- [Sample quality metrics](./sample-quality-metrics.md)
- [What are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - DDPM, Improved DDPM, DDIM, ...

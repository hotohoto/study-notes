# Diffusion based models

## Papers

(2022)

- Hierarchical Text-Conditional Image Generation with CLIP Latents
  - https://openai.com/dall-e-2/
  - [](./dall-e-2.md)

- Diffusion-Based Representation Learning
  - https://arxiv.org/abs/2105.14257
  - TODO

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

- Variational Diffusion Models
  - https://openreview.net/forum?id=2LdBqxc1Yv
  - NeurIPS 2021
  - Diederik P Kingma, Tim Salimans, Ben Poole, Jonathan Ho

- Score-Based Generative Modeling through Stochastic Differential Equations
  - https://arxiv.org/abs/2011.13456
  - ICLR 2021, Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole
  - TODO

- Denoising Diffusion Implicit Models (DDIM)
  - https://arxiv.org/abs/2010.02502
  - Jiaming Song, Chenlin Meng, Stefano Ermon
  - speeded up diffusion model sampling
    - generates high quality samples with much fewer steps
  - introduced a deterministic generative process
    - enables meaningful interpolatation in the latent variable

- Improved Denoising Diffusion Probabilistic Models
  - https://arxiv.org/abs/2102.09672
  - Alex Nichol, Prafulla Dhariwal
  - improved DDPM
  - a cosine-based variance schedule
  - tried to mix $L_\text{VLB}$ and $L_\text{simple}$ when calculating
    - $L_\text{VLB}$
      - original loss function that learns only the diagonal terms of $\Sigma_\theta$
    - $L_\text{simple}$
      - only the exponent term is used ignoring the $\Sigma_\theta$ part
  - suggested a time-averaging smoothed version of $L_\text{VLB}$ with importance sampling


(2020)

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
    - TODO
    - PixelCNN++ (?)
    - U-Net
    - global attention
    - timestep embedding
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

(2015)

- Deep unsupervised learning using nonequilibrium thermodynamics
  - https://arxiv.org/abs/1503.03585
  - DDPM
  - acheived bot tractability and flexibility
  - thousands of time steps
  - terminologies
    - quasy-static process


## References

- [Sample quality metrics](./sample-quality-metrics.md)
- [What are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - DDPM, Improved DDPM, DDIM, ...

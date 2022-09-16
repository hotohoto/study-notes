# Diffusion based models

## Outlooks on important diffusion papers

| Publication        | Year | Title (Alias)                                                        | FID*  | Remark                                                            | Architecture                  |
| ------------------ | ---- | -------------------------------------------------------------------- | ----- | ----------------------------------------------------------------- | ----------------------------- |
| JMLR               | 2005 | Estimation of Non-Normalized Statistical Models by Score Matching    |       | Introduce score matching                                          |                               |
| Neural Computation | 2011 | A Connection Between Score Matching and Denoising Autoencoders       |       | Connect score matching and denoising AEs                          |                               |
| ICLR               | 2014 | Auto-Encoding Variational Bayes (VAE)                                |       |                                                                   |                               |
| (PMLR)             | 2015 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics (NET) |       | Introduce probabilistic diffusion models                          | CNN                           |
| NeurIPS            | 2019 | (SMLD aka NCSN)                                                      | 25.32 | Use score matching with Langevin dynamics                         | RefineNet, CondInstanceNorm++ |
| NeurIPS            | 2020 | (DDPM)                                                               | 3.17  | Simplify the loss function and investigate the connection to NCSN | UNet, self-attn, GN           |
| (PMLR)             | 2021 | (Improved DDPM)                                                      | 2.94  | TODO                                                              |                               |
| ICLR               | 2021 | (NCSN++ / DDPM++)                                                    | 2.92  | TODO Connect SMLD and DDPM with SDE                               |                               |
| ICLR               | 2021 | (DDIM)                                                               |       | TODO Do faster sampling and interpolation without retraining      |                               |
| NeurIPS            | 2021 | Diffusion Models Beat GANs on Image Synthesis                        |       | Do more experiments and find some insights                        |                               |
| NeurIPS            | 2021 | Variational Diffusion Models                                         |       | TODO                                                              |                               |
| ICLR               | 2022 | (Analytic-DPM)                                                       |       | TODO                                                              |                               |

- (*) FID on CIFAR-10

## Papers

(2022)

- Hierarchical Text-Conditional Image Generation with CLIP Latents
  - https://openai.com/dall-e-2/
  - [DALLE2](./dall-e-2.md)

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
      - explains relatively poor log likelihoods
      - the majority of the models' lossless codelength are consumed to describe imperceptible image details
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
  - rao blackwell theorem (? TODO)
    - https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem
    - https://www.youtube.com/results?search_query=rao+blackwell+theorem+
    - https://arxiv.org/abs/2101.01011

(2019)

- Generative Modeling by Estimating Gradients of the Data Distribution
  - NeurIPS 2019, Yang Song, Stefano Ermon
  - https://arxiv.org/abs/1907.05600
  - https://youtu.be/m0sehjymZNU
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

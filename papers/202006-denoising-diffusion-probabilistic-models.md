# Denoising Diffusion Probabilistic Models (DDPM)

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
  - Rao Blackwell theorem (? TODO)
    - https://en.wikipedia.org/wiki/Rao%E2%80%93Blackwell_theorem
    - https://www.youtube.com/results?search_query=rao+blackwell+theorem+
    - https://arxiv.org/abs/2101.01011





### 4.3 Progressive coding

- they've got relatively poor log likelihoods in spite of high quality of generated samples

- the majority of the models' lossless codelength are consumed to describe imperceptible image details
  - for the best sample from CIFAR-10, $L_1 + L_2 + \cdots + L_T = 1.78 < L_0 = 1.97$ was found.
    - where
      - $L_1 + L_2 + \cdots + L_T$ is considered as rate (bits/dim) since they have positive signs
      - $L_0$ is considered as distortion since it has a negative sign
    - note that $L_0 = 1.97 \text{(bits/dim)}$ amounts to RMSE 0.95

- [rate-distortion theory](information-theory.md#rate-distortion-theory)

#### Progressive lossy compression

![image-20221210235714877](image-20221210235714877.png)

- The sender knows $\mathbf{x}_0$ and the real distribution $q$
- The sender and receiver share $p$ so they can encode / decode data



![image-20221211003153985](image-20221211003153985.png)

- Again, it says, the majority of the bits are indeed allocated to imperceptible distortions.

- rate

  - cumulative number of bits received

- distortion

  - $\sqrt{\Vert \mathbf{x}_0 - \hat{\mathbf{x}}_0\Vert^2/D}$
  
- $\hat{\mathbf{x}}_{0}$ is the predicted $\mathbf{x}_{0}$ at time $t$ as below.

$$
\mathbf{x}_0 \approx \hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t)) / \sqrt{\bar{\alpha}_t}\tag{15}
$$

- The latent diffusion models paper which is well known for Stable Diffusion tries to separate the generative model training process into two phases - semantic compression and perceptual compression. So they leave a diffusion model to learn semantics and an autoencoder model to learn the perceptual details ignoring imperceptible details. Note that diffusion models can learn perceptual details very well ignoring imperceptible details but its not efficient enough in terms of computational resources required.

#### Progressive generation

![image-20221211004637122](image-20221211004637122.png)

- large scale image features appear first
- details appear last

![image-20221211004653452](image-20221211004653452.png)

#### Connection to autoregressive decoding

$$
L = D_\text{KL}(q(\mathbf{x}_T) \Vert p(\mathbf{x}_T)) + \mathbb{E}_q \left[ \sum\limits_{t \ge 1} D_\text{KL}(q(\mathbf{x}_{t-1}\vert\mathbf{x}_t)\Vert p_\theta(\mathbf{x}_{t-1}\vert\mathbf{x}_t)) \right] + H(\mathbf{x}_0)\tag{16}
$$

- diffusion based generative models can be seen as a kind of autoregressive models.
  - T = number of sequence
  - making $\mathbf{x}_0,...,\mathbf{x}_t$
  - predicting $\mathbf{x}_t$ given $\mathbf{x}_T, ..., \mathbf{x}_{t-1}$
- From the perspective of the previous work [Subscale Pixel Networks(SPN)](https://arxiv.org/abs/1812.01608), you may notice that,
  - diffusion based generative models
    - adds noise into the images rather than the masks
    - can have $T=1000$ which is smaller than the data dimension $32\times32\times3$ or $256 \times 256 \times 3$

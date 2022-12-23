# High-Resolution Image Synthesis with Latent Diffusion Models


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
          - give a little KL-penalty toward standard normal on learned latent
        - VQ-reg
          - vector quantization
          - was better than KL-reg
          - embedded in the decoder
      - VQ-GAN
        - stage1
          - VQ-VAE based encoder/decoder/codebook training
          - a patch-based adversarial object
        - stage2
          - training an autoregressive transformer model
            - to generate the latent vectors using the codebook as vocabulary
          - trained separately with the encoder/decoder/code-book frozen
- cross-attention layers
  - enables multi-modal training
  - Not only decoder but also encoder of U-Net has attention
- References
  - https://github.com/CompVis/latent-diffusion
  - https://github.com/huggingface/diffusers
  - https://github.com/CompVis/stable-diffusion
  - https://github.com/AUTOMATIC1111/stable-diffusion-webui
- TODO
  - how to do inpainting?
  - how to do super resolution?



## 1 Introduction

##### Democratizing high-resolution image synthesis

- requires massive computation resources
- training a powerful diffusion model may take150-1000 V100 days

##### Departure to latent space

- two phases in training generative models
  - semantic compression
    - learn the semantic and conceptual composition
    - let a latent diffusion model to learn
  - perceptual compression
    - remove high-frequency details
    - learn little semantic variation
    - let the encoder and the decoder to learn

## 2 Related work

##### Generative models for image synthesis

- GAN
- VAE
- autoregressive models
- Diffusion Probabilistic Models (DPM)

##### Two-stage image synthesis

- VQ-VAEs
- VQ-GANs

## 3 Method

<img src="./assets/image-20221223114146546.png" alt="image-20221223114146546" style="zoom: 50%;" />

### 3.1 Perceptual image compression

Notations:

- $x \in \mathbb{R}^{H \times W \times 3}$
  - image
- $z \in \mathbb{R}^{h \times w \times c}$
  - non-quantized latent vector

- $c$
  - the dimensionality of codes

- $\mathcal{E}$
  - encoder
- $\mathcal{D}$
  - decoder

- $z = \mathcal{E}(x)$
  - an encoding from the encoder
- $\tilde{x} = \mathcal{D}(\mathcal{E}(x))$
  - an estimated image

Regularizations:

- $\text{KL-reg}$
  - a slight KL-penalty toward a standard normal on learnt latent
  - similar to VAE
- $\text{VQ-reg}$
  - the quantization layer absorbed by the decoder

### 3.2 Latent diffusion models

##### Diffusion models

$$
L_\text{DM} = \mathbb{E}_{x, \epsilon \sim \mathcal{N(0, 1)}, t} \left[ \Vert \epsilon - \epsilon_\theta (x_t, t) \Vert_2^2\right] \tag{1}
$$

- where $t$ is uniformly sample from $\{1, ..., T\}$

##### Generative modeling of latent representations

$$
L_\text{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N(0, 1)}, t} \left[ \Vert \epsilon - \epsilon_\theta (z_t, t) \Vert_2^2\right] \tag{2}
$$

- note that
  - here $\mathcal{D}$ or the codebook is not involved
  - $z \sim p(z)$ can be decoded to image space with $\mathcal{D}$, but $z_t$ cannot.

### 3.3 Conditioning mechanism

Loss function:

$$
L_\text{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N(0, 1)}, t} \left[ \Vert \epsilon - \epsilon_\theta (z_t, t, \tau_\theta) \Vert_2^2\right] \tag{3}
$$

- where
  - $\epsilon_\theta$ and $\tau_\theta$ are jointly optimized

  - $\epsilon_\theta(z_t, t, y)$
    - implementation for $p(z|y)$

  - $\tau_\theta: y \mapsto \tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$
    - domain specific encoder




Cross attention layer:

- $\varphi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$
  - intermediate representation of the UNet

- $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left( {QK^T \over \sqrt{d}} \right) \cdot V$
- $Q = W_Q^{(i)} \cdot \varphi_i(z_t)$
- $K = W_K^{(i)} \cdot \tau_\theta(y)$
- $V = W_V^{(i)} \cdot \tau_\theta(y)$



## 4 Experiments

### 4.1 On perceptual compression tradeoffs

### 4.2 Image generation with latent diffusion

### 4.3 Conditional latent diffusion

#### 4.3.1 Transformer encoders for LDMs

#### 4.3.2 Convolutional sampling beyond $256^2$

### 4.4 Super-resolution with latent diffusion

### 4.5 Inpainting with latent diffusion

## 5 Limitations & societal impact

##### Limitations

##### Societal impact

## 6 Conclusion

## References

- [15] guided diffusion
- [23] VQGAN
- [30] DDPM
- [32] Classifier-free diffusion guidance
- [66] Zero-shot text-to-image generation
- [67] VQ-VAE-2
- [72] SR3
- [82] NET
- [93] Score-based generative modeling in latent space
- [96] VQ-VAE






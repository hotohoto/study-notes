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
- learn $\sigma_t$ by interpolating in the log space between the upper bound and the lower bound
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

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

## Questions ❓

- EDM에서 정리한거랑 VDM에서 막 함수 형태를 바꾸는데 결국 같은가 다른가..
  - 다르다면 왜 하필 EDM은 그런식으로 잡은거지..? 그냥 Score SDE 페이퍼를 계승했다고 봐야하지 않나 싶음.
  - 노이즈 제곱이랑 시그널 제곱 더하면 왜 1 이 되게 하는거지? v-parameterization 에서도 이게 중요해 보이던데..
    - 이걸 다른 모델 학습할때에도 쓸수 있는건지??
- DDIM이 VE-SDE의 PF ODE라는 이야기가 있던데.. DDIM은 DDPM 에서 온거였는데 이야기가 이렇게 되나.. 그런데 EDM 페이퍼에 보면 DDIM이랑 VE-SDE의 $\sigma(t)$가 다른데.. 어떻게 된거지..
  - TODO:
    - DDIM 다시 보기. 특히 증명 부분도.
    - EDM 페이퍼의 DDIM식 유도부분을 다시 보기.




## Answered ✅

- score sde에서 f랑 g를 마음데로 세팅하고 diffusion 모델을 만든다고 생각하면 뭐가 필요한가??
  - 일단 perturbation kernel 이 있어야 함.
    - 그래야 원본데이터에 노이즈를 추가할수 있고.
    - marginal distribution 이 정의될 수 있음.
    - marginal distribution 이 정의되어야지 score function 이 정의될 수 있음.
- perturbation kernel에 scale이 포함되면 marginal distribution 이 바뀌고 그러면 score function 이 또 바뀌는 것인가??
  - perturbation kernel 이 바뀌면 score function 도 바뀌는게 맞음.
- 학습할때 쓰는 noise scheduler 와 scaling scheduler 함수가 sampling할때는 다른걸 써도 되나?
  - 네트워크에 결국 t 와 perturbation 된 이미지가 들어갈텐데 그럼 학습할때 사용한 schedule 함수를 그대로 써야지 네트워크가 알아먹을 듯.
  - 그럼 어떻게 다른 스케줄러를 막 쓰는거지?
    -  diffusers구현 보면 막쓰는거 같지는 않아. compatible한 scheduler들 끼리는 서로를 알고 있다구.
    - 뭐가 되었든 score function 만 찾을 수 있으면 된다구?? epsilon/image/velocity 뭔든 예측하면 된다구?
    - 뭐 비슷하긴해.. 
      - 일단 학습때 정의한 score 함수를 어떻게 만드는지는 알기는 알야하지.
        - diffusers에서는 기본적으로 DDPM 기준의 스케줄러를 쓰는거 같아..🤔
        - 그리고 pretrained된 네트워크의 재 사용을 위해 scale schedule 정도는 바꿀수 있게 해주는거 같아.
          - 왜냐하면 diffusers에서 EDM 의 scheduler들을 쓰고 있고, EDM에서 score function 에 input을 넣을때 scale했던걸 되돌려주는식으로 수식을 formulation 했거든..
      - 그리고 네트워크의 입력이 어떻게 들어갔는지는 기억하고 있어야 한다구.
        - diffusers에서는 기본적으로 DDPM 기준의 스케줄러를 쓰는거 같아.. 스케줄러들끼리는 compatible한 스케줄러를 알고 있따구.
        - num_trainsteps 도 scheduler들이 따로 관리하는거 보면,, 역시 중요한거 같아.
- Fokker Plank equation 을 그대로 쓸수는 없는지. 왜 구지 ODE나 SDE형태로 바꾸는지

  - distribution 을 안다고 해서 샘플을 안는것은 아님.
  - 대부분의 경우 p가 normal을 따르지 않음
  - 샘플링 가능한 곳에서 샘플링한 후에 그걸 sde/ode로 보내서 데이터를 생성해냄
- 임의의 노이즈/시그널 함수를 쓰면 왜 안되는거지?
  - 안되는거 같지 않음. 
- SDE에 임의의 f/g 를 쓰면 왜 안되는거지?
  - 안되는 것 같지는 않음. 
  - 단 original data에 noise를 씌운 후에 denoising할 것이므로 forward SDE가 노이즈를 더해가는 방향이어야 함.
  - 그리고 perturbation kernel 을 만들어야하므로 식을 너무 복잡하지 않는 범위에서 바꾸는게 맞아 보임.
  - 그리고 노이즈 시그널 함수는 정의할 수 있있어야 함.
- DDPM 으로 만든 SDE 은 왜 VP-SDE 랑 다른거지
  - 노이즈 스케줄이 결과적으로 비슷해지면서 식이 깔끔한 방향으로 잡은 것으로 보임.
- SMLD 으로 만든 SDE는 왜 VE-SDE랑 다른거지
  - 노이즈 스케줄이 결과적으로 비슷해지면서 식이 깔끔한 방향으로 잡은 것으로 보임.



## Comparison

- metric
  - A discrete Markov chain
  - perturbation kernel
    - noise schedule function
    - signal/scale schedule function
  - marginal distribution
  - SDE
- mathematical models (network agnostic)
  - DDPM
  - SMLD
  - VP-SDE
  - VE-SDE
  - DDIM
  - EDM



|        | Discrete Markov chain (forward)                              | Perturbation kernel | marginal distribution | SDE  | PF-ODE |
| ------ | ------------------------------------------------------------ | ------------------- | --------------------- | ---- | ------ |
| SMLD   |                                                              |                     |                       |      |        |
| DDPM   | $\mathbf{x}_t = \sqrt{1-\beta_i}\mathbf{x}_{t-1} + \sqrt{\beta_i} \mathbf{z}_{i-1}$ |                     |                       |      |        |
| VE-SDE |                                                              |                     |                       |      |        |
| VP-SDE |                                                              |                     |                       |      |        |
| DDIM   |                                                              |                     |                       |      |        |
| EDM    |                                                              |                     |                       |      |        |



## Papers

### Retrieval-Augmented Diffusion Models

- RDM
- https://arxiv.org/abs/2204.11824
- method
  - keep explicit memory/database for known data samples
  - retrieve visually similar samples to the training instance and encode them using CLIP
  - use those embedding when generating a sample



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
- aka  guided-diffusion
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



## References

- [Sample quality metrics](./sample-quality-metrics.md)
- [What are diffusion models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - DDPM, Improved DDPM, DDIM, ...

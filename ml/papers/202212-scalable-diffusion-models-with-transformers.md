# Scalable Diffusion Models with Transformers

- (paper) https://arxiv.org/abs/2212.09748
- (project) https://www.wpeebles.com/DiT
- (code) https://github.com/facebookresearch/DiT
  - JAX
  - CC-BY-NC license (so we cannot use this in a startup.)
- (review) https://youtu.be/eTBG17LANcI



![image-20230104022935598](./assets/image-20230104022935598.png)

- DiT models need to be compared to `LDM-8-G` in that they all use encoding factor=8
- `-G` seems to mean guidance
- SOTA fid on ImageNet256 and ImageNet512
- GFLOPs
  - floating point operations (in giga)
- trained on TPU v3-256 pod



![image-20230104022908132](./assets/image-20230104022908132.png)



- a latent diffusion model but with transformers as the backbone instead of U-Net 
  - f=8
  - 256x256x3 👉 32x32x4
  - reuse stable diffusion's pretrained encoder/decoder
- with DDPM like settings
- predict $\Sigma_\theta$ as well as iDDPM does
- Use Classifier Free Guidance (CFG) if needed
  - scale
    - 4.0 (for generating 256x256 images)
    - 6.0 (for generating 512x512 images)
- patchify on the latent image
  - p = 2, 4, 8
- block design
  - (option 1) in-context conditioning
  - (option 2) cross-attention
  - (option 3) adaLN
  - (option 4) adaLN-Zero
    - this was the best
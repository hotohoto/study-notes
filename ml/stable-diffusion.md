# Stable diffusion

## Versions

### 3.5

- https://huggingface.co/stabilityai/stable-diffusion-3.5-large
- https://stability.ai/news/introducing-stable-diffusion-3-5
- Multimodal Diffusion Transformer (MMDiT)
- typography improved

### 3.0

[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](_202403-scaling-rectified-flow-transformers-for-high-resolution-image-synthesis.md)

- based on
    - [Rectified Flow](202209-flow-straight-and-fast-learning-to-generate-and-transfer-data-with-rectified-flow.md)
    - [EDM2](_202312-analyzing-and-improving-the-training-dynamics-of-diffusion-models.md)
    - [DiT](202212-scalable-diffusion-models-with-transformers.md)
- typography enabled

### 2.1

- more data and more training
    - and less restrictive filtering of dataset
- new NSFW filter
- support for non-standard aspect ratio

### 2.0

- OpenCLIP
    - text encoder
- default resolutions (?)
    - 512x512
    - 768x768
- include an upscaler diffusion model
- introduce Depth2img
    - structure-preserving image-to-image synthesis
    - shape-conditional image synthesis

### 1.x

- OpenAI CLIP
- support for inpainting
- environment
    - under 10GB VRAM
    - default resolution
        - 512x512
- 1.4
    - https://huggingface.co/CompVis/stable-diffusion-v1-4
- 1.5
    - https://huggingface.co/runwayml/stable-diffusion-v1-5

## Source codes

### Stable diffusion v2

https://github.com/Stability-AI/stablediffusion

(configurations)

- `v2-1_768-ema-pruned.yaml`
    - model
        - first_stage_config
            - z_channels: 4
            - ch: 128
            - attn_resolutions: []
            - num_res_blocks: 2
            - ch_mult: [1, 2, 4, 4]
        - unet_config
            - params
                - model_channels: 320
                - attention_resolutions: [4,2,1]
                    - which is equivalent to [True, True, True, True] with respect to the current `channel_mult` setting
                    - actual attention resolutions
                        - for a 512x512 image: [64,32,16]
                        - for a 768x768 images: [96, 48, 24]
                - num_res_blocks: 2
                    - depth for each channel_mult
                - channel_mult: [1, 2, 4, 4]
                - context_dim: 1024
                    - the prompt condition is going to be 77x1024

(classes)

- LatentDiffusion
    - (first_stage_model): AutoencoderKL(embed_dim=4, use_ema=False)
        - (encoder)
        - (decoder)
        - (quant_conv): Conv2d(8, 8, ...)
        - (post_quant_conv): Conv2d(4, 4, ...)
    - (cond_stage_model): FrozenOpenCLIPEmbedderWithCustomWords
        - (id_start): 49406
        - (id_end): 49407
        - (wrapped): FrozenOpenCLIPEmbedder
            - (max_length): 77
            - (model): CLIP
                - (vocab_size): 49408
                - (token_embedding): Embedding(49408, 1024)
                - (transformer): Transformer
    - (model): DiffusionWrapper(sequential_cross_attn=False, conditioning_key='crossattn')
        - (diffusion_model): UNetModel(n_embed=None)
            - (time_embed)
            - (input_blocks)
            - (middle_block)
            - (output_blocks)

(training)
- (first_stage_model)
    - loss
        - reconstruction loss
            - L1
            - LPIPS loss (perceptual loss)
                - squared difference within VGG16's feature space
        - KL divergence loss
            - of posterior values
                - with a very small weight
                - e.g. $10^{-6}$
        - generator loss
        - discriminator loss

(scripts)

```sh
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1

# not working??
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python scripts/txt2img.py --prompt="a professional photograph of an astronaut riding a horse" --ckpt=/workspaces/stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.ckpt --config=/workspaces/stable-diffusion-webui/models/Stable-diffusion/v2-1_768-ema-pruned.yaml --H=768 --W=768 --n_samples=1 --n_iter=1
```

### Stable diffusion v1

https://github.com/CompVis/stable-diffusion

### Latent diffusion

https://github.com/CompVis/latent-diffusion

### VQ-GAN

https://github.com/CompVis/taming-transformers

## Model signatures

### Stable diffusion model v2.1

- [AutoencoderKL](./assets/repr-dump-stable-diffusion-model-v2.1-autoencoder-kl.txt)
- [LatentDiffusion](./assets/repr-dump-stable-diffusion-model-v2.1-latent-diffusion.txt)

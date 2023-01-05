# Stable diffusion



## Versions

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

### 1.0

- support for inpainting
- environment
  - under 10GB VRAM
  - default resolution
    - 512x512

## Source codes



### Stable diffusion web UI

https://github.com/AUTOMATIC1111/stable-diffusion-webui

- features
  - https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features
  - styles can be applied as weighted conditions
  - high-resolution fix
    - txt2img generates an LR image
    - img2img  generates an HR image from the LR image 
  
- ui
  - modules.ui
  - based on gradio
- api
  - modules.api
  - based on FastAPI
- (Generate clicked)
  - modules.txt2img
    - p = modules.processing.StableDiffusionProcessingTxt2Img
    - modules.processing.process_images(p)
      - modules.processing.process_images_inner(p)
        - samples_ddim = p.sample()
          - p.sampler = sd_samplers.create_sampler(...)
            - module.sd_samplers.KDiffusionSampler(...)
              - (model_wrap_cfg): CFGDenoiser
                - (inner_model): CompVisVDenoiser
                  - (inner_model): LatentDiffusion
          - p.sampler.sample(...)
        - ...
        - p.sd_model.decode_first_stage(x)
          - ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage()
            - self.first_stage_model.decode(z)

```sh
CUDA_VISIBLE_DEVICES=9 python launch.py --no-half --no-half-vae
```



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

#### AutoencoderKL

```py
AutoencoderKL(
  (encoder): Encoder(
    (conv_in): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (down): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (downsample): Downsample(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
      )
    )
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (norm_out): GroupNorm(32, 512, eps=1e-06, affine=True)
    (conv_out): Conv2d(512, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (decoder): Decoder(
    (conv_in): Conv2d(4, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (mid): Module(
      (block_1): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (attn_1): AttnBlock(
        (norm): GroupNorm(32, 512, eps=1e-06, affine=True)
        (q): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (k): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (v): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (proj_out): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      )
      (block_2): ResnetBlock(
        (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
    (up): ModuleList(
      (0): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 128, eps=1e-06, affine=True)
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
      )
      (1): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (2): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (3): Module(
        (block): ModuleList(
          (0): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (1): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (2): ResnetBlock(
            (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (attn): ModuleList()
        (upsample): Upsample(
          (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
    )
    (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
    (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (loss): Identity()
  (quant_conv): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
  (post_quant_conv): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
)
```

#### LatentDiffusion

```py
LatentDiffusion(
  (model): DiffusionWrapper(
    (diffusion_model): UNetModel(
      (time_embed): Sequential(
        (0): Linear(in_features=320, out_features=1280, bias=True)
        (1): SiLU()
        (2): Linear(in_features=1280, out_features=1280, bias=True)
      )
      (input_blocks): ModuleList(
        (0): TimestepEmbedSequential(
          (0): Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        (1): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=320, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=320, out_features=320, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=320, out_features=320, bias=False)
                  (to_v): Linear(in_features=320, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=320, out_features=2560, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=1280, out_features=320, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=1024, out_features=320, bias=False)
                  (to_v): Linear(in_features=1024, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=320, out_features=320, bias=True)
          )
        )
        (2): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=320, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 320, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=320, out_features=320, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=320, out_features=320, bias=False)
                  (to_v): Linear(in_features=320, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=320, out_features=2560, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=1280, out_features=320, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=320, out_features=320, bias=False)
                  (to_k): Linear(in_features=1024, out_features=320, bias=False)
                  (to_v): Linear(in_features=1024, out_features=320, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=320, out_features=320, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((320,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=320, out_features=320, bias=True)
          )
        )
        (3): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (4): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 320, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=640, out_features=640, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=1024, out_features=640, bias=False)
                  (to_v): Linear(in_features=1024, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=640, out_features=640, bias=True)
          )
        )
        (5): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=640, out_features=640, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=1024, out_features=640, bias=False)
                  (to_v): Linear(in_features=1024, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=640, out_features=640, bias=True)
          )
        )
        (6): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (7): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
          )
        )
        (8): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
          )
        )
        (9): TimestepEmbedSequential(
          (0): Downsample(
            (op): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          )
        )
        (10): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
        )
        (11): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Identity()
          )
        )
      )
      (middle_block): TimestepEmbedSequential(
        (0): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
        (1): SpatialTransformer(
          (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
          (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
          (transformer_blocks): ModuleList(
            (0): BasicTransformerBlock(
              (attn1): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (ff): FeedForward(
                (net): Sequential(
                  (0): GEGLU(
                    (proj): Linear(in_features=1280, out_features=10240, bias=True)
                  )
                  (1): Dropout(p=0.0, inplace=False)
                  (2): Linear(in_features=5120, out_features=1280, bias=True)
                )
              )
              (attn2): CrossAttention(
                (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                (to_out): Sequential(
                  (0): Linear(in_features=1280, out_features=1280, bias=True)
                  (1): Dropout(p=0.0, inplace=False)
                )
              )
              (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            )
          )
          (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
        )
        (2): ResBlock(
          (in_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (h_upd): Identity()
          (x_upd): Identity()
          (emb_layers): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (out_layers): Sequential(
            (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
            (1): SiLU()
            (2): Dropout(p=0, inplace=False)
            (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
          (skip_connection): Identity()
        )
      )
      (output_blocks): ModuleList(
        (0): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (1): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
        )
        (2): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): Upsample(
            (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (3): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
          )
        )
        (4): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 2560, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
          )
        )
        (5): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=1280, out_features=10240, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=5120, out_features=1280, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=1280, out_features=1280, bias=False)
                  (to_k): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_v): Linear(in_features=1024, out_features=1280, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=1280, out_features=1280, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (2): Upsample(
            (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (6): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1920, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=640, out_features=640, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=1024, out_features=640, bias=False)
                  (to_v): Linear(in_features=1024, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=640, out_features=640, bias=True)
          )
        )
        (7): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 1280, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=640, out_features=640, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=1024, out_features=640, bias=False)
                  (to_v): Linear(in_features=1024, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=640, out_features=640, bias=True)
          )
        )
        (8): TimestepEmbedSequential(
          (0): ResBlock(
            (in_layers): Sequential(
              (0): GroupNorm32(32, 960, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (h_upd): Identity()
            (x_upd): Identity()
            (emb_layers): Sequential(
              (0): SiLU()
              (1): Linear(in_features=1280, out_features=640, bias=True)
            )
            (out_layers): Sequential(
              (0): GroupNorm32(32, 640, eps=1e-05, affine=True)
              (1): SiLU()
              (2): Dropout(p=0, inplace=False)
              (3): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (skip_connection): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
          )
          (1): SpatialTransformer(
            (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
            (proj_in): Linear(in_features=640, out_features=640, bias=True)
            (transformer_blocks): ModuleList(
              (0): BasicTransformerBlock(
                (attn1): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=640, out_features=640, bias=False)
                  (to_v): Linear(in_features=640, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (ff): FeedForward(
                  (net): Sequential(
                    (0): GEGLU(
                      (proj): Linear(in_features=640, out_features=5120, bias=True)
                    )
                    (1): Dropout(p=0.0, inplace=False)
                    (2): Linear(in_features=2560, out_features=640, bias=True)
                  )
                )
                (attn2): CrossAttention(
                  (to_q): Linear(in_features=640, out_features=640, bias=False)
                  (to_k): Linear(in_features=1024, out_features=640, bias=False)
                  (to_v): Linear(in_features=1024, out_features=640, bias=False)
                  (to_out): Sequential(
                    (0): Linear(in_features=640, out_features=640, bias=True)
                    (1): Dropout(p=0.0, inplace=False)
                  )
                )
                (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
              )
            )
            (proj_out): Linear(in_features=640, out_features=640, bias=True)
          )
          (2): Upsample(
            (conv): Co...kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (nin_shortcut): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 256, eps=1e-06, affine=True)
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 256, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (2): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
        (3): Module(
          (block): ModuleList(
            (0): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (1): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
            (2): ResnetBlock(
              (norm1): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (norm2): GroupNorm(32, 512, eps=1e-06, affine=True)
              (dropout): Dropout(p=0.0, inplace=False)
              (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (attn): ModuleList()
          (upsample): Upsample(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          )
        )
      )
      (norm_out): GroupNorm(32, 128, eps=1e-06, affine=True)
      (conv_out): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (loss): Identity()
    (quant_conv): Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1))
    (post_quant_conv): Conv2d(4, 4, kernel_size=(1, 1), stride=(1, 1))
  )
  (cond_stage_model): FrozenOpenCLIPEmbedderWithCustomWords(
    (wrapped): FrozenOpenCLIPEmbedder(
      (model): CLIP(
        (transformer): Transformer(
          (resblocks): ModuleList(
            (0): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (1): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (2): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (3): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (4): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (5): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (6): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (7): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (8): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (9): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (10): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (11): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (12): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (13): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (14): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (15): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (16): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (17): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (18): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (19): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (20): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (21): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (22): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
            (23): ResidualAttentionBlock(
              (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
              )
              (ls_1): Identity()
              (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (mlp): Sequential(
                (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
                (gelu): GELU(approximate='none')
                (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
              )
              (ls_2): Identity()
            )
          )
        )
        (token_embedding): EmbeddingsWithFixes(
          (wrapped): Embedding(49408, 1024)
        )
        (ln_final): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
```
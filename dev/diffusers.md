# diffusers

## Useful snippets

Show configs such as the default resolution

```py
print(model.config)
```



Show available schedulers

```py
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
print(pipeline.scheduler.compatibles)
```



Swap schedulers

```py
from diffusers import EulerDiscreteScheduler

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```



Display images

```py
import PIL.Image
import numpy as np


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)
```



Use scheduler step by step

```py
import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)
```



Use generator and image grid

```py
from diffusers.utils import make_image_grid

def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 20

    return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}

images = pipeline(**get_inputs(batch_size=4)).images
make_image_grid(images, 2, 2)
```



Convert a local `.safetensors` file trained by webui

```py
from safetensors import safe_open
from safetensors.torch import save_file

def fix_diffusers_model_conversion(load_path: str, save_path: str):
    # load original
    tensors = {}
    with safe_open(load_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # migrate
    new_tensors = {}
    for k, v in tensors.items():
        new_key = k
        # only fix the vae
        if 'first_stage_model.' in k:
            # migrate q, k, v keys
            new_key = new_key.replace('.to_q.weight', '.q.weight')
            new_key = new_key.replace('.to_q.bias', '.q.bias')
            new_key = new_key.replace('.to_k.weight', '.k.weight')
            new_key = new_key.replace('.to_k.bias', '.k.bias')
            new_key = new_key.replace('.to_v.weight', '.v.weight')
            new_key = new_key.replace('.to_v.bias', '.v.bias')
        new_tensors[new_key] = v

    # save
    save_file(new_tensors, save_path)

fix_diffusers_model_conversion(
    load_path="./../../models/webui_model.safetensors",
    save_path="./../../models/webui_model_fixed.safetensors"
)
```



Load a local model and generate an image 

```py
pipeline = StableDiffusionPipeline.from_single_file("/workspaces/driving/models/webui_model_fixed.safetensors")
pipeline.enable_xformers_memory_efficient_attention()
pipeline = pipeline.to("cuda")

pipeline(
    prompt="a sks dog"
).images[0]
```



Use multiple ControlNet models

```py
from diffusers import StableDiffusionControlNetPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils import load_image, make_image_grid
from PIL import Image

controlnet = MultiControlNetModel([ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True
), ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16, use_safetensors=True
)])

pipeline = StableDiffusionControlNetPipeline.from_single_file(
    "/workspaces/driving/models/ir_15_all_lowlr_42010_fixed.safetensors",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
pipeline.enable_xformers_memory_efficient_attention()

batch_size = 12
canny_image = load_image("canny.png")
seg_image = load_image("seg.png")

images = pipeline(
    prompt=["happy people in seoul"]*batch_size,
    negative_prompt=["cartoon, illustration, painting"]*batch_size,
    image=[canny_image, seg_image],
    generator=[torch.Generator(device=pipeline.device).manual_seed(i) for i in range(batch_size)],
).images

make_image_grid(images, rows=2, cols=6)
```



Use StableDiffusionPipeline's internal components directly

```py
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler


device = "cuda"
pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    use_safetensors=True,
).to(device)

generator = torch.Generator(device).manual_seed(1024)
positive_prompt = ["a painting of a super cute cat"]
negative_prompt = [""] * len(positive_prompt)
num_inference_steps = 25
guidance_scale = 6.5

model = pipeline.unet
tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
image_processor = pipeline.image_processor
vae = pipeline.vae

scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
scheduler.set_timesteps(num_inference_steps)

positive_text_input = tokenizer(
    positive_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
positive_text_embeddings = text_encoder(positive_text_input.input_ids.to(device))[0]

negative_text_input = tokenizer(
    negative_prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
negative_text_embeddings = text_encoder(negative_text_input.input_ids.to(device))[0]

text_embeddings = torch.cat([positive_text_embeddings, negative_text_embeddings])

latent = torch.randn((1, 4, 64, 64), generator=generator, device=device)

for i, t in enumerate(scheduler.timesteps):
    with torch.no_grad():
        noisy_residual_cond, noisy_residual_uncond = model(
            torch.cat([latent, latent]),
            t,
            encoder_hidden_states=text_embeddings,
        ).sample.chunk(2)
    noisy_residual = noisy_residual_cond + guidance_scale * (
        noisy_residual_cond - noisy_residual_uncond
    )
    sched_output = scheduler.step(noisy_residual, t, latent)
    latent = sched_output.prev_sample

with torch.no_grad():
    image = vae.decode(latent / vae.config.scaling_factor).sample

image = image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]
image
```





## Components

- DiffusionPipeline(ConfigMixin)
  - only defines `__call__` method to sample an image
  - contains
    - models
    - schedulers
    - processors
- ModelMixin(torch.nn.Module)
  - `.enable_xformers_memory_efficient_attention()`
- SchedulerMixin
  - input
    - a noisy sample
    - a time step
  - output
    - a denoised sample



## LoRA implementation

- with PEFT multiple LoRA adapters can be set at the same time
- for a single adapter, PEFT is not required

### Components

- Accelerator
  - register_save_state_pre_hook()
  - register_load_state_pre_hook()



(diffusers/loaders/lora.py)

- LoraLoaderMixin
  - lora_state_dict()
  - load_lora_into_unet()
  - load_lora_into_text_encoder()
  - save_lora_weights()
  - unload_lora_weights()
  - fuse_lora()
  - unfuse_lora()
  - set_adapters_for_text_encoder()
  - disable_lora_for_text_encoder()
  - enable_lora_for_text_encoder()
  - set_adapters()
  - disable_lora()
  - enable_lora()
  - delete_adapters()
  - get_active_adapters()
  - get_list_adapters()
- StableDiffusionXLPipeline(LoraLoaderMixin)
  - save_lora_weights()



(diffusers/loaders/unet.py)

- UNet2DConditionLoadersMixin
  - load_attn_procs()
  - save_attn_procs()
  - fuse_lora()
  - unfuse_lora()



(diffusers/training_utils)

- unet_lora_state_dict()



(diffusers/models/lora.py)

- PatchedLoraProjection
- LoRALinearLayer
- LoRAConv2dLayer
- LoRACompatibleConv
  - (used when PEFT is not available)
  - (single LoRA layer seems to be allowed)
  - set_lora_layer()
- LoRACompatibleLinear
  - (used when PEFT is not available)
  - (single LoRA layer seems to be allowed)
  - set_lora_layer()



(diffusers/models/unet_2d_condition.py)

- UNet2DConditionModel(UNet2DConditionLoadersMixin)
  - attn_processors
  - set_attn_processor

(diffusers/models/attention_processor.py)

- ...
- LoRAAttnProcessor2_0
- LoRAXFormersAttnProcessor
- LoRAAttnAddedKVProcessor



## Optimization/special hardware

- AttnProcessor
  - https://github.com/huggingface/diffusers/blob/1a5797c6d4491a879ea5285c4efc377664e0332d/src/diffusers/models/attention_processor.py#L402



### Torch2.0 support

- `torch.compile()`
  - components
    - `TorchDynamo`
      - python bytecode to FX graphs
    - `TorchInductor`
      - FX graphs to optimized kernel
  - references
    - https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    - https://pytorch.org/blog/accelerated-pytorch-2/

- `torch.nn.functional.scaled_dot_product_attention`
  - run in a fused way that automatically selects one of the three available kernels (methods)
    - (common)
      - requirements
        - head dimension must be multiple of
          - 8 for 16-bit floating point numbers
          - 4 for 32-bit floating point numbers
      - limitations
        - only the casual mask is supported by the currently supported kernels
        - for custom kernels to use custom mask set `attn_mask` `None` and `is_causal` `False`
        - returning averaged attention is not supported so `need_weights` needs to be set `False`.
      - args
        - set `is_causal` `True` to use causal attention masking 
        - `attn_mask` is not supported for custom kernels
        - `need_weights` needs to be set `False`.
    - spda_flash
      - the Flash Attention kernel 
      - requirements
        - 16-bit floating point (float16, bfloat16)
        - Nvidia GPUs with SM80+ architecture level
      - limitations
        - no Nested Tensor support
    - spda_mem_eff
      - the xFormers memory-efficient attention kernel
      - requirements
        - Nvidia GPUs with sm5x+ architecture level
      - limitation
        - no dropout support
        - no Nested Tensor support
    - sdpa_math
      - default implementation
      - 
  - references
    - https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention





### xFormers

- memory_efficient_attention
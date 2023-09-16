# diffusers

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
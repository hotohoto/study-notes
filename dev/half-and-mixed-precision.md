# Training with half/mixed precision



## Half precision



- can reduce the VRAM being used dramatically
- but not that stable
  - you may need to reduce `eps` for your optimizers
    - e.g. eps=1e-3 instead of 1e-8
  - you may need to keep the type of `BatchNorm2d` as `float32`



## Automatic mixed precision



- It can
  - make training faster
  - make half-precision training safe
  - casts types depending on the operation
- Mixed precision alone doesn't reduce the VRAM being used that much.



### PyTorch libraries



- torch.amp
- torch.autocast()
  - can be used as contexts or decorators
  - only for
    - out-of-place operations
      - with `float16`/`float32` tensors
      - without `dtype` specified
  - For each operation, there is `dtype` defined to which the data is casted
  - use this for the forward passes
    - not recommended to use this in the backward passes
- torch.cpu.amp.autocast()
- torch.cuda.amp.autocast()
- torch.cuda.amp.GradScaler()
  - to prevent underflow multiply the network's loss by a scale factor

### Recommended GPUs



- "Mixed precision primarily benefits Tensor Core-enabled architectures (Volta, Turing, Ampere)"
  - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- list
  - Nvidia RTX 3090 Ti
  - Nvidia RTX 3090
  - Nvidia RTX 3080 Ti
  - Nvidia RTX 3080
  - Nvidia RTX 3070 Ti
  - Nvidia RTX 3070
  - Nvidia RTX 3060 Ti
  - Nvidia RTX 3060
  - Nvidia RTX 3050



## References



- half precision issues
  - https://discuss.pytorch.org/t/training-with-half-precision/11815/2
  - https://github.com/pytorch/pytorch/issues/40497 
  - https://github.com/pytorch/pytorch/issues/26218 
- mixed precision in PyTorch
  - https://pytorch.org/docs/stable/amp.html
  - https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
  - https://pytorch.org/docs/stable/notes/amp_examples.html
- mixed precision in accelerate
  - https://huggingface.co/docs/accelerate/v0.18.0/en/quicktour#mixed-precision-training
- mixed precision in nvidia
  - https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
# diffusers

- pipeline
  - only defines `__call__` method to sample an image
  - contains the model objects and the schedule
  - check if super resolution is supported
    - 👉 No
- scheduler
  - one of the k-diffusion schedulers supported: LMSDiscreteScheduler
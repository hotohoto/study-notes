# An overview of Generative Models

## A comparison of genertive models

|            | density  | fidelity | diversity | inference speed | training stability |
| ---------- | -------- | -------- | --------- | --------------- | ------------------ |
| VAE        | explicit | low      | high      | high            | high               |
| GAN        | implicit | high     | low       | high            | low                |
| Flow-based | explicit |          |           |                 |                    |
| Diffusion  | explicit | high     | high      | low             | high               |

## for Videos

- Video Diffusion Models
  - 2022, Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet
  - https://arxiv.org/abs/2204.03458
  - https://video-diffusion.github.io/
  - https://youtu.be/CqCNLH4ZRDo
  - 3d-UNET
  - TODO

- MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation
  - 2022, Mila
  - Vikram Voleti, Alexia Jolicoeur-Martineau, Christopher Pal
  - https://arxiv.org/abs/2205.09853
  - trained using less than 4 GPUs for 1-12 days
  - can be used without conditions
  - can be used conditioned on future frames
  - TODO, 3D-Conv

- Flexible Diffusion Modeling of Long Videos
  - 2022, The University of British Columbia
  - William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, Frank Wood
  - https://arxiv.org/abs/2205.11495
  - https://plai.cs.ubc.ca/2022/05/20/flexible-diffusion-modeling-of-long-videos/
  - can generate videos that are more than 1 hour long
  - TODO

- NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion
  - 2021, Microsoft Research Asia
  - https://arxiv.org/abs/2111.12417
  - https://youtu.be/InhMx1h0N40
  - pretraining
  - A modified VQ-VAE with nice formulations for video frames
  - 3d near by attention
  - NUWA, Video, Generative models

- FVD: A new Metric for Video Generation
  - ICLR 2019
  - https://openreview.net/forum?id=rylgEULtdN
  - TODO

- Video-to-Video Synthesis
  - 2018 NVIDIA
  - https://arxiv.org/abs/1808.06601
  - https://github.com/NVIDIA/vid2vid
  - https://tcwang0509.github.io/vid2vid/
  - Spatially progressive training
  - enhanced temporal consistency
  - sequential generator
  - multi-scale discriminators
    - image discriminator
    - video discriminator
  - spatially progressive training from low resolutions to high resolutions
  - temporally progressive training
  - alternating trainingg
  - TODO vid2vid

## Images

- SEIGAN: Towards Compositional Image Generation by Simultaneously Learning to Segment, Enhance, and Inpaint
  - 2018-2019
  - https://arxiv.org/abs/1811.07630

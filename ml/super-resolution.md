# Super-resolution

- https://youtu.be/OMIqkn2DCUk

## Metrics

- PSNR
- MOS
- SSIM

## Tasks

- Single Image Super Resolution (SISR)
- Video SR

## Polynomial based methods

- nearest neighbor

- linear

- cubic (1981)

- bilinear

- bicubic

## Dictionary based methods

- self similarity based (ICCV 2009)
- scene matching based (ICCP 2012)

## Deep learning based

- SRCNN

  - seems to be the first CNN based super resolution method

  - 1 channel

  - L2 loss

  - inputs and outputs have the same resolution

  - train based on small image patches

  - application
    - Waifu2x

- VDSR

  - use skip connection
    - the deeper the better

  - multi scale SISR

  - the gradient of deeper network seems to be likely to explode
    - we may try smaller learning rate along with ADAM or use gradient clipping
      - https://youtu.be/nvsYKSHw0jo?t=1815

- ESPCN (CVPR 2016)
  - upscaler layer
    - if upscale f times
    - use $f^2$ channels
    - concatenate the outputs channels

- SRResNet

  - upscaler from ESPCN

  - skip connection from ResNet

- SRGAN
  - SRResNet + GAN

- [EDSR / MDSR](./papers/201707-enhanced-deep-residual-networks-for-single-image-super-resolution.md)

- [Residual Dense Network (RDN)](./papers/201802-residual-dense-network-for-image-super-resolution.md)


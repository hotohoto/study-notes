# Taming Transformers for High-Resolution Image Synthesis

- aka VQGAN
- CVPR 2021
- Replaces VQ-VAE2's "PixelCNN + multi-head attention" with "autoregressive transformers"

## 1 Introduction

- CNN

  - has a strong locality bias

  - cons
    - bad at learning semantics
    - use shared weights across all positions

- transformers

  - pros
    - good at learning long-range relations
      - no inductive bias on locality of interactions
  - cons
    - high cost

- a codebook

## 2 Related work

- The transformer family

- Convolutional approaches

- Two-stage approaches
  - VQ-VAE
  - VQ-VAE-2

## 3 Approach

![image-20221216213800604](./assets/image-20221216213800604.png)

### 3.1 Learning an effective codebook of image constituents for use in transformers

### 3.2 Learning the composition of images with transformers

## 4 Experiments

### 4.1 attention is all you need in the latent space

### 4.2 A unified model for image synthesis tasks

### 4.3 Building context-rich vocabularies

### 4.4 Benchmarking image synthesis results

## 5 Conclusion


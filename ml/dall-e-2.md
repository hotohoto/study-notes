## TODO

- skim/read from Figure 7

## mini DALL-E 2

## Questions

- How to evaluate CLIP
- how text is preprocessed
  - huggingface is used... where is the code it is used? 🤔
  - where is the language model such as GPT3... is it in the CLIP ?
- Do we need zenml?
  - https://colab.research.google.com/github/zenml-io/zenml/blob/main/examples/quickstart/notebooks/quickstart.ipynb

## summary

- Hierarchical Text-Conditional Image Generation with CLIP Latents
  - DALL·E 2, unCLIP
  - OpenAI
  - architecture
    - prior
      - AR
        - PCA
        - transformer
      - diffusion
    - decoder
      - diffusion
  - terminologies to take a look
    - Denoising Diffusion Implicit Model (DDIM)
    - Sharpness-Aware Minimization (SAM)
    - Break Stretch Ratio (BSR)
    - Frechet Inception Distance (FID)
    - Guided Language to Image Diffusion for Generation and Editing (GLIDE)

## structure

TODO internals

- dalle2
  - clip
  - diffusion_prior
    - prior_network
    - (clip)
  - decoder
    - unet1
    - unet2
    - (clip)

## Prerequisite

(2021)

- GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
  - OpenAI

- Zero-Shot Text-to-Image Generation
  - DALL·E
  - OpenAI
  - https://openai.com/blog/dall-e/
  - https://arxiv.org/abs/2102.12092
  - https://youtu.be/CQoM0r2kMvI
  - 12 billion parameters
  - based on GPT3
    - uses transfomer's decoder
  - uses
    - CLIP
      - to choose the best image matching the text
    - VQ-VAE
    - gumble softmax relaxation
      - replacing argmax to be able to calculate gradients
      - only for training
- Learning Transferable Visual Models From Natural Language Supervision
  - Contrastive Language–Image Pre-training (CLIP)
  - OpenAI
  - Used natural language supervision so that it may be also scalable to all the images and text on the internet
  - learns a joint representation space for text and images
  - properties
    - robust to image distribution shift
    - good at zero-shot settings
    -
  - https://openai.com/blog/clip/
  - https://arxiv.org/abs/2103.00020

(2020)

- Sharpness-Aware Minimization for Efficiently Improving Generalization
  - SAM

(2019)

- Generating Diverse High-Fidelity Images with VQ-VAE2 (NIPS 2019)
  - README

(2017)

- Vector Quantized Variational AutoEncoder (VQ-VAE, 2017, NIPS)
  - README


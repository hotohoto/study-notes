# DALL·E 2

## TODO

- clear server tracking database and artifacts and restart the servers
- download more images

- Retrieval-Augmented Diffusion Models
  - https://arxiv.org/abs/2204.11824
  - stable diffusion (2/2)

- High-Resolution Image Synthesis with Latent Diffusion Models
  - CVPR 2022
  - stable diffusion (1/2)
  - https://arxiv.org/abs/2112.10752
  - https://github.com/CompVis/stable-diffusion
  - https://github.com/huggingface/diffusers

- Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
  - Imagen
  - https://arxiv.org/abs/2205.11487

## mini DALL-E 2

## Questions


- TODO https://www.assemblyai.com/blog/how-dall-e-2-actually-works/
  - AR or diffusion for prior?
    - what kind of AR model
  - only caption or both caption and text embedding?
  - super resolution??
    - what is the final image size?
- How to evaluate CLIP
- how text is preprocessed
  - huggingface is used... where is the code? 🤔
  - where is the language model such as GPT3... is it in the CLIP ?
- Do we need zenml?
  - https://colab.research.google.com/github/zenml-io/zenml/blob/main/examples/quickstart/notebooks/quickstart.ipynb

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

## Papers

(2022)

- Hierarchical Text-Conditional Image Generation with CLIP Latents
  - https://openai.com/dall-e-2/
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
    - Guided Language to Image Diffusion for Generation and Editing (GLIDE)

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
  - https://openai.com/blog/clip/
  - https://arxiv.org/abs/2103.00020
  - Contrastive Language–Image Pre-training (CLIP)
  - OpenAI
  - Used natural language supervision so that it may be also scalable to all the images and text on the internet
  - learns a joint representation space for text and images
  - properties
    - robust to image distribution shift
    - good at zero-shot settings

(2020)

- Sharpness-Aware Minimization for Efficiently Improving Generalization
  - SAM

(2019)

- Generating Diverse High-Fidelity Images with VQ-VAE2 (NIPS 2019)
  - README

(2017)

- Vector Quantized Variational AutoEncoder (VQ-VAE, 2017, NIPS)
  - README


## References



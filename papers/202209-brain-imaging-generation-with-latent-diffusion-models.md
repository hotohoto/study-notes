# Brain Imaging Generation with Latent Diffusion Models

- https://arxiv.org/abs/2209.07162
- MICCAI 2022
  - impact factor 13.828 (2022-2023)
- train an LDM model for generating brain MRI images
- an early release of UK Biobank (UKB) dataset 
  - N=31,740
- Created a synthetic dataset with 100,000 brain images
  - 591.19 GB
  - https://academictorrents.com/details/63aeb864bbe2115ded0aa0d7d36334c026f0660b
- limitations
  - no code is available
  - seems like 3d volume generation, but there is no architecture info


## 1 Introduction

## 2 Methods

### 2.1 Datasets and image preprocessing

- variables
  - min-max normalization

- image
  - a rigid body registration to a common MNI space
    - use UniRes
      - https://github.com/brudfors/UniRes
    - image registration
      - https://en.wikipedia.org/wiki/Image_registration
    - rigid body transformations
      - https://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/pdfs/Ch2.pdf
      - translations
      - rotations
      - zooms
      - shears
    - a common MNI space
      - 'the MNI "space" merely defines the boundaries around the brain, expressed in millimeters, from a set origin.'
      - https://www.lead-dbs.org/about-the-mni-spaces/
    - MNI
      - Montreal Neurological Institute
      - https://owenlab.uwo.ca/pdf/2002-Brett-NatRevNeurosci-The%20problem%20of%20functional%20localization%20in%20the%20human%20brain.pdf

### 2.2 Generative models

- autoencoder
  - from $160 \times 224 \times 160$ to $20 \times 28 \times 20$
  - objective function
    - L1
    - perceptual loss
    - KL regularization of the resulting latent space
    - patch based adversarial objective

- diffusion model
  - T=1000

- conditioned on
  - age
  - gender
  - ventricular volume
  - brain volume relative to the intracranial volume

## 3 Experiments

### 3.1 Sampling quality

<img src="./assets/image-20230316095417477.png" alt="image-20230316095417477" style="zoom:67%;" />

- quantitative evaluation
  - FID
    - using Med3D
  - MS-SSIM
    - Multi-Scale Structural Similarity Metric
  - 4-G-R-SSIM

### 3.2 Conditioning evaluation

- ventricular volume
  - label the generated images with `SynthSeg` and see the Pearson correlation
    - between the condition and the generated label
    - https://github.com/BBillot/SynthSeg
    - 0.972
- ventricular volume and head volume
  - try extrapolation and take a look at the generated images
- age
  - train a age prediction model
  - r=0.692

### 3.3 Synthetic dataset



https://figshare.com/

- requires exact link

https://www.healthdatagateway.org/

- requires exact link

via torrent

```shell
sudo apt install transmission-cli
transmission-cli https://academictorrents.com/download/63aeb864bbe2115ded0aa0d7d36334c026f0660b.torrent
```



## 4 Conclusions

## References










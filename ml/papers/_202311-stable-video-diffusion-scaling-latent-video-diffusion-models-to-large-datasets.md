# Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets

- Stability AI
- Based on SD2.1
- Large video dataset (LVD)
  - 580M annotated video clip pairs
  - (212 years of content)

## 1 Introduction

## 2 Background

## 3 Curating data for HQ video synthesis

### 3.1 Data processing and annotation

- cut-detection
  - at three different FPS levels
- caption generation
  - CoCa
    - [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)
    - image captioner
    - use mid-frame
    - (question)
      - there might be better captioners available??
        - https://paperswithcode.com/task/image-captioning
  - V-BLIP
    - video-based captioner
  - LLM
    - summarization of two captions already made
- filtering
  - dense optical flow 🤔
    - at two FPS levels
    - to filter out static scenes
  - OCR
    - to weed out clips containing large amounts of written text
  - (CLIP embedding from the first/middle/last frame)
    - aesthetic score
    - text-image similarity

### 3.2 Stage I: image pretraining

- use Stable Diffusion 2.1



TODO

---

### 3.3 Stage II: curating a video pretraining dataset

- 

### 3.4 Stage III: High-quality finetuning

- 

## 4 Training video models at scale

### 4.1 Pretrained base model

### 4.2 High-resolution text-to-video model

### 4.3 High resolution image-to-video model

### 4.4 Frame interpolation

### 4.5 Multi-view generation

## 5 Conclusion

## References

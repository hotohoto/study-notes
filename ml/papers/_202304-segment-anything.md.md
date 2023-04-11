# Segment Anything

- https://arxiv.org/abs/2304.02643

- Meta AI Research, FAIR

- Tried to train a foundation model for any segmentation tasks

  

## 1 Introduction

<img src="./assets/image-20230408223004085.png" alt="image-20230408223004085" style="zoom:67%;" />

## 2 Segment anything task

- inspired by GPT3
  - pretraining by next token generation
  - to solve diverse downstream tasks via prompt engineering
    - https://en.wikipedia.org/wiki/Prompt_engineering
- similar to interactive segmentation
- pretrain the model
  - given prompts
    - points/boxes/text/segmentation
  - predicts the ground truth
- zero-shot transfer
  - e.g.
    - any cat bounding box detector would be used for segmentation

## 3 Segment anything model

<img src="./assets/image-20230408223036480.png" alt="image-20230408223036480" style="zoom:67%;" />



- Image encoder
  - with a pretrained ViT adapted to high resolution inputs
- Prompt encoder
  - (sparse)
    - points
      - positional encoding
      - summed with own embedding for points
    - boxes
      - positional encoding
      - summed with own embedding for boxes
    - text
      - CLIP
  - (dense)
    - masks embedded using convolutions
      - summed element-wise with image embedding
- Mask decoder

<img src="./assets/image-20230408223106804.png" alt="image-20230408223106804" style="zoom:67%;" />

## 4 Segment anything data engine

- Assisted-manual stage
  - SAM
    - a browser based interactive segmentation
    - trained on public segmentation datasets
    - encoder network
      - ViT-B (at first)
      - ViT-H (later)
  - annotators masked prominent objects first
  - go to the next image once a mask took over 30 seconds to annotate
  - retrain SAM 6 times
    - using only the newly annotated masks
  - annotation time per mask decreased
- Semi-automatic stage
  - prominent objects were masked automatically
  - annotators masked any unannotated objects first
- Fully automatic stage
  - already collected enough masks to improve the model
  - developed the ambiguity aware model
  - IoU prediction module
    - select confidence segments by a threshold
  - non-maximum suppression (NMS)
    - to filter duplicates

## 5 Segment anything dataset

- SA-1B dataset
  - TODO

## 6 Segment anything RAI analysis

## 7 Zero-shot transfer experiments

## 8 Discussion

- TODO

## References

- [10] GPT3
- [82] CLIP
- [83] DALL·E

## A Proof of formula 1

## B Proof of formula 2

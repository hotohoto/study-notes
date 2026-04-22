# VGGT: Visual Geometry Grounded Transformer

- CVPR 2025 Best Paper Award
- https://arxiv.org/abs/2503.11651
- https://github.com/facebookresearch/vggt
- no 3d inductive bias

![[Pasted image 20260330202627.png]]

(outputs)

- camera intrinsics/extrinsics
    - first camera extrinsics = identity
- point maps
- depth maps
- 3D point tracks
- inference time: under 1 sec

## 0 Questions / backgrounds

### Classical methods - recap

- feature extraction
- feature matching
    - https://github.com/topics/feature-matching
- camera pose estimation
- triangulation
- bundle adjustment
- dense reconstruction
    - Multi-View Stereo

### Comparison to SLAM

|                        | SLAM localization       | VGGT localization             |
| ---------------------- | ----------------------- | ----------------------------- |
| Use of existing map    | Yes                     | No                            |
| Processing a new frame | Milliseconds            | Full transformer forward pass |
| Online suitability     | Highly suitable         | Inefficient                   |
| State maintenance      | Maintains map and graph | No persistent state           |
| Loop closure           | Explicit                | None (re-inference)           |

## 1 Introduction

## 2 Related work

## 3 Method

![[Pasted image 20260330203311.png]]

![[Pasted image 20260330211450.png]]

### 3.1 Problem definition and notation

### 3.2 Feature backbone

- alternating-attention

### 3.3 Prediction heads

- coordinate frames 🤔
- camera predictions
- dense predictions
- tracking

### 3.4 Training

- training losses
- ground truth coordinate normalization 🥲

## 4 Experiments

## 5 Discussion

## References

## A Proof of formula 1

## B Proof of formula 2

## Notes

### DPT

202103 Vision Transformers for Dense Prediction

![[Pasted image 20260330205429.png]]

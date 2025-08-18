# CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image

- https://sites.google.com/view/cast4
- https://arxiv.org/abs/2502.12894
- code not available yet

![[Pasted image 20250815094739.png]]

- Pipeline
    - pre-process module
        - Florence-2
        - Grounded-SAM2 v2
        - MoGe
    - perceptive 3d instance generator
        - (occlusion-aware) object generation module
            - latent diffusion model
            - conditions
                - partial image segments
                - (optional) point clouds
        - pose alignment generation module (AlignGen)
    - pose aware generation
        - 3DShape2VecSet

        ![[Pasted image 20250815095105.png]]

![[Pasted image 20250815095021.png]]

## 1 Introduction

## 2 Related work

## 3 Method

## 4 Experiments

## 5 Discussion

## References

## A Proof of formula 1

## B Proof of formula 2

## Notes

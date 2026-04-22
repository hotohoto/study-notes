# VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction

- https://arxiv.org/abs/2601.19887
- https://github.com/MIT-SPARK/VGGT-SLAM
- realtime
- rgb SLAM
- feed forward
- no 15 DoF
- planar degeneracy??
    - wall
    - flat floor
- loop closure
- test on NVIDIA Jetson Thor

## 1 Introduction

![[Pasted image 20260406204710.png]]

## 2 Related work

### Classical mapping techniques

### Feed-forward scene reconstruction

- geometric foundation model (GFM)
    - DUSt3R
    - MASt3R
    - VGGT
- GFM based SLAM
    - MASt3R-SLAM
    - PREF3R
    - VGGT-Long
    - SING3R-SLAM
        - fuses to global Gaussian splatting map
    - ViSTA-SLAM
        - light weight two-view association
    - MegaSaM
        - dynamic scenes
    - TTT3R
        - Test time training
        - extend CUT3R
    - TALO

### Attention layer analysis

## 3 Method

### Notation

- n
    - number of key frames for each submap

## 4 Experiments

## 5 Discussion

## References

## A Proof of formula 1

## B Proof of formula 2

## Notes

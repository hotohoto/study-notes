# 202503 Any6D - Model-free 6D Pose Estimation of Novel Objects

- https://arxiv.org/abs/2503.18673
- CVPR 2025
- KAIST, NVIDIA

![[Pasted image 20260316205103.png|480]]

## 1 Introduction

6D pose-estimation methods:
- `Instance-level`
    - requires
        - exact RGB-textured 3D model
- `Category-level`
    - requires
        - predefined object categories
    - hard to acquire comprehensive training datasets
- `Category-agnostic`
    - `model-based`
        - requires test-time RGB-textured 3D model
    - `model-free`
        - requires test-time either
            - multi-view reference image
            - video

|           | model-free        | Oryon (baseline)                          | Any6D       |
| --------- | ----------------- | ----------------------------------------- | ----------- |
| reference | multi-view images | single RGBD                               | single RGBD |
|           |                   | language queue                            |             |
|           |                   | aligning point clouds<br>(matching based) |             |
| cons      |                   | bad at occlusion                          |             |

### (datasets)

https://bop.felk.cvut.cz/datasets/

#### HO3D

- https://github.com/shreyashampali/ho3d
- "A dataset for pose estimation of hand when interacting with object and severe occlusions."

![[Pasted image 20260316211013.png|480]]

#### YCBINEOAT

- https://github.com/wenbowen123/iros20-6d-pose-tracking

![[Pasted image 20260316211455.png|480]]

- Real manipulation tasks
- 3 kinds of end-effectors
- 5 YCB objects
- 9 videos for evaluation, 7449 RGBD in total
- Ground-truth poses annotated for each frame
- Forward-kinematics recorded
- Camera extrinsic parameters calibrated

#### REAL275

???

#### Toyota Light

- 21 objects
- a table-top
- four different table cloths
- five different lighting

#### LINEMOD

- 15 texture-less household objects with discriminative color, shape and size.
- Each object is associated with a test image set showing one annotated object instance
    - significant clutter
    - mild occlusion.

#### LM-O

- ECCV 2014
- subset of LINEMOD for occlusion

## 2 Related work

## 3 Method

![[Pasted image 20260316205401.png|800]]
- $I_A$
    - RGBD
- $I_Q$
    - RBD
- Image-to-3D
    - [202404 InstantMesh - Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models](https://arxiv.org/abs/2404.07191)
- $O_N$
    - normalized

### 3.1 Coarse object alignment

![[Pasted image 20260316215427.png|640]]

### 3.2 Fine object alignment

TODO

## 4 Experiments

## 5 Discussion

## References

## Personal questions

(Answered)

- 6-DoF object pose estimation vs 6-DoF Grasp Detection??
    - 6-DoF object pose estimation
        - 👉 물체 pose 찾기
    - 6-DoF Grasp Detection
        - 👉 gripper pose 찾기
- RGBD 이미지를 써 놓고 왜 다시 size estimation 을 해야 하지??
    - 🤖 self occlusion을 고려해야 함.

(To be answered)

- Oriented-bounding box 를 어떻게 찾았지?
    - 🤖 convex hull 의 PCA로 계산했을것?

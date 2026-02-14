# NVIDIA Omniverse

## TODO

- https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/asset-structure-principles.html

## Summary

https://www.nvidia.com/en-us/omniverse/
- NVIDIA Omniverse™ is a collection of **libraries and microservices** for developing physical AI applications such as industrial digital twins and robotics simulation.
- Since 2021

## (assets)

### OpenUSD

- scene graph description
- a hierarchical format

(also supports for)
- custom properties
- realtime collaboration / seamless integration
- physics properties
    - https://openusd.org/release/api/usd_physics_page_front.html
- camera models

### SimReady assets

- A set of 3D asset specifications for industrial digitalization and other simulation use cases.
- defined by NVIDIA
- features
    - semantic labeling
    - non-visual sensor/material attributes
        - radar, lidar, thermal imaging
    - collision shapes
    - mass, center of mass
    - friction, elasticity, visual properties
- References
    - https://docs.omniverse.nvidia.com/simready/latest/
    - https://www.nvidia.com/en-us/glossary/simready/
    - https://docs.omniverse.nvidia.com/usd/latest/technical_reference/conceptual_data_mapping/index.html
    - https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/asset-structure-principles.html
    - https://huggingface.co/collections/nvidia/physical-ai

#### Asset acquisition

- sample datasets
    - https://huggingface.co/collections/nvidia/physical-ai
- asset market place
    - https://simready.com/
    - Their prices looks too expensive though.

## (rendering)

### RTX

- real-time ray-traced rendering engine
- Blender's Cycles render uses it under the name of NVIDIA OptiX
    - (Note that OptiX is a specialized API to use NVIDIA RTX graphic cards)

## (physics)

### PhysX

- physics engine

### Warp

- Newton Engine
    - an open source physics engine based on Warp

## (tools)

### NVIDIA Isaac Sim

- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html
- a reference framework application that can be extendible
- focusing on reinforcement learning, imitation learning, and motion planning
    - can generate synthetic data

### usdview

- a minimal usd file viewer

### Omniverse USD Composer

- https://docs.omniverse.nvidia.com/composer/latest/index.html
- an Omniverse app for world-building that allows users to assemble, light, simulate and render large scale scenes.
- can connect to the other 3d content editing tools (e.g. Blender)

### Omniverse Nucleus server

- https://docs.omniverse.nvidia.com/nucleus/latest/index.html
- the database and collaboration engine

### NuRec

- https://docs.nvidia.com/nurec/
- sample scene dataset
    - https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
        - 924 scenes
- rendering demo
    - https://carla.org/2025/09/16/release-0.9.16/
    - https://carla.readthedocs.io/en/latest/nvidia_nurec/
    - https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html

(pros)

- simulation ready for reinforcement learning
- can be used as a source of multiview video replay

(cons)

- Still needs to collect data from the real world
    - gather LiDAR + RGB multiview camera
    - meaning it needs to label things

#### Features

- reconstructing scenarios as 3D scenes
    - inputs
        - real world camera and lidar data
    - outputs
        - simulated 3d environment
            - in .usdz file
        - sequence track / rig trajectories
- rendering

#### Papers related

- [202410 SANA - Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629)
    - use linear transformer block
    - 32x AE
- [202412 3DGUT - Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting](https://arxiv.org/abs/2412.12507)
    - improved ray tracing based rendering of particles such as 3DGS
    - supports distorted rays
- [202501 Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575)
    - world foundation model
        - input
            - multi modal
        - output
            - video (as world state)
- [202503 Difix3D+ - Improving 3D Reconstructions with Single-Step Diffusion Models](https://arxiv.org/abs/2503.01774)
    - Remove 3DGS artifacts with fine tuned SD-Turbo

## How to download/install

- https://github.com/NVIDIA-Omniverse/kit-app-template
    - https://youtu.be/a9MuYruNZl0?si=swOd9NoBEC-Sc2el
- https://catalog.ngc.nvidia.com/

## License

- https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/
- https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-omniverse/

## Pricing

- Enterprise subscription plan 에 묶여 있는 경우가 많음.
    - (Isaac Sim 등 일부 툴들은 예외)
- 가격을 외부에 공개 하지 않으려고 하는 거 같음.

## Remark

- Carla + OSM seems in progress

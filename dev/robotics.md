# Robotics

- https://www.youtube.com/watch?v=Y4gtuQljLrY&list=PLP4rlEcTzeFIvgNQD8M1T7_PzxO3JNK5Z
- [[mujoco]]
- Visual servo control
    - https://notebooklm.google.com/notebook/930195a2-338e-4ec1-891e-c4158b6af478

## Problems

- sensing
    - camera
    - LiDAR
    - Inertial Measurement Unit (IMU)
    - tactile sensor
        - pressure, force, texture, and temperature
- perception
    - detection
    - segmentation
    - pose estimation (6DoF)
    - scene understanding
    - affordance detection
- state estimation
    - (SfM)
        - offline
        - input: image set
        - output:
            - camera pose estimation
            - 3d points
    - visual odometry
        - online
        - frame-to-frame motion
    - SLAM
        - online
        - localization
        - mapping
- world model / representation
    - spatio temporal RAG
- task planning
    - VLA
- motion planning
    - path-planning
    - collision-free trajectory
    - constraint handling
    - diffusion policy
- policy / control
    - visual servoing
        - IBVS
        - PBVS
- actuation

## Glossary

- Closed-loop system
    - Output is fed back into the input (uses feedback)
- Open-loop system
    - Output is not fed back (no feedback)
- Mapping
    - To build a map of the environment
- Localization
    - To estimate the object's pose within a map
- pose
    - x, y, z
    - orientation

## Conferences

- ICRA
- IROS
- RSS
- CoRL
- WAFR
- ISER

# MuJoCo

- https://mujoco.readthedocs.io/en/stable/python.html
- Body
    - empty axis
- Geom
    - MuJoCo’s collision detector assumes that all geoms are convex.
        - it internally replaces meshes with their convex hulls if the meshes are not convex.
        - decompose non-convex shapes to model them
- Site
    - a light geom that cannot participate in collisions

## Setting up

- 
- https://github.com/vuer-ai/vuer
    - https://docs.vuer.ai/en/latest/index.html
    - https://docs.vuer.ai/en/latest/tutorials/mujoco_interactive_simulator.html

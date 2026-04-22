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

## Glossary

- MjSpec
    - the editable blueprint used to build a model
- MjModel
    - the compiled immutable physical structure
- MjData
    - the mutable runtime state

## Calculation

`qpos` → forward kinematics → collision → dynamics

## Setting up

- https://github.com/vuer-ai/vuer
    - https://docs.vuer.ai/en/latest/index.html
    - https://docs.vuer.ai/en/latest/tutorials/mujoco_interactive_simulator.html

## Snippets

### Collision check without changing scene (draft/proposal/not-validated)

```python
import mujoco

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free_joint")
adr = model.jnt_qposadr[jid]

# 1) backup the pose (numpy array's copy())
original_pose = data.qpos[adr:adr+7].copy()

# 2) overwrite the candidate pose
data.qpos[adr:adr+7] = [x, y, z, qw, qx, qy, qz]

# 3) update forward kinematics
mujoco.mj_kinematics(model, data)

# 4) check collision
mujoco.mj_collision(model, data)
collision = data.ncon > 0

# 5) restore the original one
data.qpos[adr:adr+7] = original_pose

# it's safer the restore kinematics as well
# mujoco.mj_kinematics(model, data)
```

# Infinite Photorealistic Worlds using Procedural Generation

- https://arxiv.org/abs/2306.09310
- CVPR2023
- https://github.com/princeton-vl/infinigen

## Setup

- How to use it out of its repository folder with library-like usage? 🤔
    - Config file paths depend on `infinigen.repo_root()`?
        - See https://github.com/princeton-vl/infinigen/blob/59a2574f3d6a2ab321f3e50573dddecd31b15095/infinigen/core/init.py#L155
        - but it's fine actually if the `folder_rel` is an absolute path.
    - gin depends on the repo path?
        - some existing gin files include `infinigen_examples/`.
        - so you need to
            - (option1)
                - copy all the gin files and replace the path string with the path you use.
                - copy and modify `generate_nature.py` as well to use the copied gin files.
                    - the gin configuration folder needs to be relative to the Infinigen repo path
                - to run `manage_jobs.py` you will need to set `get_cmd.driver_script` as well to the new path of `generate_nature.py`
            - (option2 - not tested)
                - Don't use a config file but use overrides
                - Don't copy `generate_nature.py` but use it directly (maybe in the Infinigen directory).
                - Anyway, at some point we'll need to modify, at least, `infinigen_examples`. So I didn't investigate this option further.
        - note that
            - it's not possible to modify `apply_gin_configs.config_folders`' via gin
                - It's because gin is not loaded yet at that moment.
            - it's not possible to use a custom config and also the configs in the `infinigen_examples/configs_natures` directly
                - some configs have `include` statements and they are not relative to the project root while Infinigen is a submodule.

## Tasks

### generate_nature.py

(per scene)
- coarse
    - generate coarse terrain shape
        - resolution
            - 1m
            - 150m x 150m
    - put placeholders for creatures/trees/obstacles
- populate
    - replace the placeholders with unique detailed assets

(per camera)

- fine_terrain
    - resolution
        - 0.2m~20m
        - 1000m x 1000m
- render
- ground_truth
- (etc)
    - mesh_save
    - export

## Configuration

https://github.com/princeton-vl/infinigen/blob/main/docs/ConfiguringInfinigen.md

- `infinigen_examples/generate_nature.py`
    - Configurable via `--pipeline_configs`
    - Overwrites `infinigen_examples/configs_nature/base.gin`
    - use `simple.gin` reduces details for the low spec machiens
    - can use a specific scene type e.g. `desert.gin`
        - in `infinigen_examples/configs_nature/scene_types/`
- `manage_jobs.py`
    - Configurable via `--configs`
    - uses `infinigen_examples/generate_nature.py` as a driver script

## Rendering

Call graph

- generate_nature.main()
- execute_tasks.main()
- execute_tasks.render()
- render.render_image()

etc.

- the ground truth types are set by `passes_to_save`
- it's configured by `base.gin`

## OcMesher

- https://github.com/princeton-vl/OcMesher
- 👉 [2023 View-Dependent Octree-based Mesh Extraction in Unbounded Scenes for Procedural Synthetic Data](https://arxiv.org/abs/2312.08364)

![img](OcMesher.png)

## Marching cubes

- used to make clouds in Infinigen
- voxelization
    - point cloud ➡️ binary voxels
- `scipy.ndimage.distance_transform_edt()`
    - binary voxels ➡️ an array of distances
    - from non-zero to the closest zeros
- marching cubes
    - an array of distances ➡️ mesh

## Folders and files

### infinigen/

- assets/
    - (assets inherit AssetFactory and decorated by `@gin.configurable`)
- core/constraints/
- core/nodes/
    - Manage nodes for Blender's shader/geometry
- core/placement/
- core/placement/camera.py
    - configure_cameras()
        - compute_base_views()
            - camera_pose_proposal()
- core/placement/density.py
- core/placement/factory.py
    - `AssetFactory` ⭐
        - spawn_placeholder()
        - spawn_asset()
    - make_asset_collection()
- core/placement/placement.py
    - populate_all()
- core/rendering/render.py
    - render_image()
- core/util/blender.py
    - `GarbageCollect`
        - (clean up `bpy.data`)
- core/util/pipeline.py
    - `RandomStageExecutor`
        - (run with a seed and a Blender garbage collector)
- core/execute_tasks.py
    - main()
    - execute_tasks()
- core/init.py
- core/generator.py
- core/surface.py
- core/tagging.py
- core/tags.py
- datagen/customgt/
    - looks like it handles shader in opengl and blender 🤔
        - e.g. used for wings and hairs of animals
- datagen/customgt/dependencies/
    - https://github.com/p-ranav/argparse
    - https://github.com/nlohmann/json
    - (npy in c)
        - https://github.com/rogersce/cnpy
    - (linear algebra)
        - https://gitlab.com/libeigen/eigen
    - (math for opengl)
        - https://github.com/g-truc/glm
    - (abstraction layer on opengel, vulkan, etc.)
        - https://github.com/glfw/glfw
    - (image utils)
        - https://github.com/nothings/stb
- datagen/monitor_tasks.py
    - iterate_scene_tasks() ⭐
        - defines which tasks to run
- infinigen_gpl/
- OcMesher/
- terrain/core.py
    - Terrain ⭐
        - coarse_terrain()
- tools/
- launch_blender.py ⭐
    - The entry point of the Blender python script mode

### infinigen_example/

- generate_nature.py
    - main()
    - compose_nature()
    - populate_scene()

### scripts/

- install/
- launch/
- (just for the license issue)
    - https://github.com/princeton-vl/infinigen_gpl.git
- (configuration)
    - https://github.com/google/gin-config
- (c++)

## Blender file

## Analysis

- spawn_placeholder
    - hide_render=True
    - contains the placeholder cubes
        - contains location
        - contains a shaped seed object
- unique_assets
    - hide_viewport=True
        - (disabled in viewports)
    - the parent of each is a placeholder cube object

### Structure of data-blocks

- 📦 terrain
    - atmosphere
    - atmosphere_fine
    - OpagueTerrain
    - OpagueTerrain.inview_inview
        - not generated if the camera is too far away
    - OpagueTerrain_fine
    - OpagueTerrain_unapplied
- 📦 camera_rigs
- 📦 cameras
- 📦 colliders
    - OpaqueTerrain.inview_near.001.collider
- 📦 particles
    - emitter(name='camrig.0', mesh_type='cube')
    - emitter(name='camrig.0', mesh_type='plane')
- 📦 assets
    - 📦 assets:TreeFlowerFactory(543568399)
        - TreeFlowerFactory(543568399).spawn_asset(0)
        - TreeFlowerFactory(543568399).spawn_asset(1)
        - TreeFlowerFactory(543568399).spawn_asset(2)
    - 📦 assets:BushFactory(543568399).twigs
        - GenericTreeFactory(543568399).spawn_asset(0)
    - ...
- 📦 scatter
- 📦 placeholders
    - 📦 placeholders:BushFactory(36234574)
        - BushFactory(36234574).spawn_placeholder(73)
            - BushFactory(36234574).spawn_asset(1) 🔗
- 📦 unique_assets
    - 📦 unique_assets:BushFactory(36234574)
        - BushFactory(36234574).spawn_asset(1) ⭐
            - ...
- 📦 particleassets
- 📦 animhelper

### Tips

- how to see all the populated objects in Blender
    - why they are not visible?
        - placeholder can occlude unique assets
        - unique_assets are disabled in viewport
    - 👉 enable unique_assets in the view port
        - Outliner > Filter > Disable in Viewports
        - Enable all the data-blocks in `unique_assets` by clicking the monitor icon of the collection with shift
    - 👉 hide placeholders
        - Hide all the data-blocks in `placeholders` by clicking the eye icon of the collection with shift

## Performance table

- `coarse` with `simple.gin`
    - 960x540
    - gen06, cpu
        - 6m05s
- `coarse`
    - creates caves, mountains,
    - 1280x720
    - gen06 , cpu
        - 28m52s

## Slurm

- https://slurm.schedmd.com/
    - cluster management
    - job scheduling
- similar to Ray in a way
- python interface
    - https://github.com/facebookincubator/submitit

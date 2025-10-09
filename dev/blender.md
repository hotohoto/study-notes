# Blender

## Glossary

- view layer
    - can toggle each data block per view layer
- data-blocks
    - scene, collection, object, object data
- object
    - mesh object, camera object, light object, ...
- object data
    - mesh, camera, light, ...
- driver
    - sets a property from something else
    - many things including constraints are implemented via drivers

### Data-blocks

https://docs.blender.org/manual/en/2.93/files/data_blocks.html#data-block-types

- scene
    - Primary store of all data displayed and animated.
    - Used as top-level storage for objects & animation.
- collection
    - Group and organize objects in scenes.
    - Used to instance objects, and in library linking.
- object
    - An entity in the scene with location, scale, rotation
    - Used by scenes & collections.
    - types
        - mesh
        - camera
        - light
- mesh (data)
- camera (data)
- light (data)
- material
- texture
- node tree
    - group of re-usable nodes
- text
- brush
- workspace
    - predefined window layout (placed at the top of the Blender UI window)
- ...

## Initial setup

- make numpad keys available for tenkeyless keyboards
    - `Edit` > `Preferences` > `Input` > `Keyboard` > `Emulate Numpad`

## Basic UI

- Click and drag the mouse middle button
- shift + click mouse middle button + drag
- tab
    - object mode ⟷ editing mode
- 5
    - perspective view
- 0
    - camera view
- Ctrl + space
    - make the current area full view
- F3
    - open the command palette
- F12
    - render the scene
- Ctrl + F12
    - render the video
- Ctrl + \`
    - toggle the gizmo
- N
    - toggle the right side panels
- T
    - toggle the left toolbox
- Alt + D
    - make a duplicate link which shares the object data
- Shift + right click
    - move cursor
- Shift + C
    - move cursor back to the origin

## Edit mode

- select none
    - Alt + A
- select faces isolated around??
    - Ctrl + H
- hide selected faces
    - H
- show all faces again
    - Alt + H

Remarks:

- attributes seems to be duplicated when subdividing the edges

## Pose mode

- can set
    - bone level transformation
    - bone level constraints

## Coordinates and transforms

- 3D cursor 🛟
    - location
        - used as the location for the object to be added
    - rotation
- transform
    - location 🟡
        - The object's origin location in local coordinates
- rotation
    - relative to the closest global axis and the origin 🟡
- delta transform
    - the secondary transform applied on top of the primary `transform`

### Rotation

https://docs.blender.org/manual/en/latest/advanced/appendices/rotations.html

- Euler modes
    - XYZ, XZY, YXZ, YZX, ZXY, ZYX
    - (bottom)(intermediate)(top)
- Axis angle mode
    - defines an axis in (X, Y, Z)
    - rotate around that axis by a rotation angle (W)
- Quaternion mode
    - WXYZ
    - good for interpolation, multiplication, division
    - w defines the amount of rotation
    - xyz defines the direction to rotate around

## Geometry Nodes

- Geometry
    - Meshes
    - Curves
    - Point Clouds
    - Volumes
    - Instances
- fields
    - A center dot means all values are the same.
- `Ctrl + Shift + LeftClick`
    - Attach the viewer to the clicked node
- `Ctrl + Shift + D`
    - duplicate a node
- `Alt + Drag`
    - take the node keeping the connection of the other nodes

## Color management

- https://docs.blender.org/manual/en/latest/render/color_management.html
- Blender internally uses linear colors that follow the physics better
    - this is different from RGB
- (remarks)
    - `.ply` may contains a RGB color for each vertex
    - When you load such a file,
        - Blender converts the colors to its own color format
        - and keeps them in a color attribute.

## Shader

- A material is specified by one or more shaders
- surface
    - defines what happens at the surface of an object
- volume
    - defines what happens inside the object
- displacement
    - actually changes the shape of the object
- shaders
    - Principled BSDF
        - alpha
            - makes it transparent without the light interaction...? 🤔
        - transmission
            - makes it transparent with the light interaction
            - EEVEE requires some extra settings
                - Render > Screen Space Reflections ✅ > Refraction ✅
                - Material > Settings > Raytrace Refraction ✅
            - CYCLES seems just fine
        - coat
            - makes it glossy
        - sheen
            - makes it fabric like
        - emission

## Rigging

### A rigging example

Let's do rigging for the default cube 🙃

- subdivide the default cube
    - go to the edit mode
    - right click
    - subdivide
- add an armature
    - go to the object mode
    - Add > Armature
    - Show the bone
        - Armature > Object panel > Viewport Display > In Front ✅
            - (equivalently) Armature > (Armature) data > Viewport Display > In Front ✅
- Add connected bones
    - go to the edit mode
    - e to add bones
- Assign the cube to the armature
    - click the cube
    - Ctrl + click the armature
    - Ctrl + P
    - Armature Deform > With Empty Groups
- Paint weights
    - Click the mesh data of the cube
    - Go to the weight paint mode
    - Tool > Options > Auto Normalization ✅
    - Enable Mesh Symmetry for X ✅
    - Smooth weight borders
- Move bones
    - Go to the object mode
    - Click the armature
    - Go to the pose mode
    - Click a bone
    - Modify the bone

### Constraints

#### Copy location/rotation

e.g. how to make an animating soldier hold a gun

- copy location
    - target: armature
        - bone: the corresponding hand
    - target: world
    - owner: world
- copy rotation
    - target: armature
        - bone: the corresponding hand
    - replace
    - target: world
    - owner: world
- place gun in the edit mode

## Animation

(data structure)
- object / armature
    - animation_data
        - nla_tracks
            - track (can have multiple strips)
                - strips
                    - strip
        - action (current action to be edited and played on top of NLA tracks)
            - fcurves
                - fcurve
                    - key_frame_points
(basic concepts)
- action
    - the actual key frames are saved here
- `animation_data.action`
    - the active action
    - optional
    - for editing and also the main track which is played on top of the other NLA tracks
- strip
    - an instance of an action
(basic usage)
- add animation properties
- add key frame
    - It is added into object.animation_data.action
- choose another time point
- change the animation properties
- add key frame

## Physics

### Force Field

### Collision

- related to
    - particles
    - soft bodies
    - cloth

### Cloth

- Make a pillow
    - https://youtube.com/shorts/mLnhyokj_kA?si=CJRj5OwQ6mHO8Xxo

### Dynamic paint

### Soft Body

### Fluid

### Rigid Body

- active
    - can be moved
- passive
    - can not be moved
    - can interact with other rigid bodies
- collisions
    - shape
        - Box
        - ...
        - Convex Hull
        - Mesh
        - Compound Parent ⭐

### Rigid Body Constraint

## Rendering

### Depth Map

- https://youtu.be/saptddljRks?si=1X8cDq3RjTbutJFf
- use cycle
- save it as OpenEXR
    - in RGBA
    - use OpenCV for python to read it
        - it's better than OpenEXR
            - simpler
            - more stable

```python
def load_exr(path):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    return cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
```

## Scripting

```sh
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:savoury1/blender

sudo apt update
sudo apt install blender

blender scene.blend --background --python script.py
blender scene.blend --background --python script.py --python-use-system-env
blender scene.blend --background --python script.py --python-use-system-env -- 1 2 3
```

- `--python-use-system-env` makes packages installed by `pip` available
    - e.g. you may want to install `numpy` in your system python and use it in the blender
- you may pass python arguments after `--`
    - https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script

## Hide objects

- `.hide_set(True)`
    - hides in viewport
    - relate to 👁️
- `.hide_viewport = True`
    - "disables" in viewport
    - related to 🖥️
- `.hide_render = True`
    - make it not rendered
    - related to 📷

### Transforms

(The related properties of Object)

https://docs.blender.org/api/current/bpy.types.Object.html

- location
- rotation_euler
  - in radian
- matrix_world
  - local to world
- matrix_local
  - TODO
- matrix_basis
  - TODO
- dimensions
  - contains axis aligned height / width / length

### Data model

A scene snapshot:

- `Scene`
    - `Object[]`
            - `Object`
                - `Camera`
                - `animation_data`
                    - `Action[]`
                        - `Action`
                            - `Keyframe[]`
                                - `Keyframe`
                                    - `data_path`
                                        - (which property to modify)
                                    - `array_index`
                                        - (index to an element of the property)
                            - `FCurve[]`
            - `Object`
                - `Mesh`
                - `animation_data`

(Remarks)

- When calling `current_scene.frame_current = 50` the actual animated value is calculated from `FCurves`

### Manage modifiers

apply modifiers temporarily and get vertices with respect to the world coordinates:

```py
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)
vertices = [obj_eval.matrix_world @ v.co for v in obj_eval.data.vertices]
obj_eval.to_mesh_clear()
```

### Ray casting

```python
result, location, normal, face_index = ground_mesh.ray_cast(
    camera_xyz_local, cam2point_local
)
```

### Avoid collision

Use Bounding Volume Hierarchy Tree (BVHTree).

```python
from mathutils.bvhtree import BVHTree
import bpy

...

depsgraph = bpy.context.evaluated_depsgraph_get()
eval_obj = obj.evaluated_get(depsgraph)
mesh = eval_obj.to_mesh()
bvh = BVHTree.FromMesh(mesh)

...

bvh1 = BVHTree.FromObject(obj1, depsgraph)
bvh2 = BVHTree.FromObject(obj2, depsgraph)
intersections = bvh1.overlap(bvh2)
```

```python
def compute_push_vector(obj_a, obj_b, depsgraph):
    bvh_b = BVHTree.FromObject(obj_b, depsgraph)
    
    loc_a = obj_a.location
    hit = bvh_b.find_nearest(loc_a)
    
    if hit is None:
        return None
    
    (location, normal, index, distance) = hit
    
    # 밀어낼 벡터 = 겹친 지점 → 가장 가까운 지점으로 향하는 벡터
    push_vector = (loc_a - location).normalized() * (0.1 + distance)
    
    return push_vector
```

### Addon development

- (references)
  - https://docs.blender.org/api/current/
  - https://docs.blender.org/api/current/info_quickstart.html
  - https://docs.blender.org/api/current/bpy.props.html
  - https://docs.blender.org/api/current/bpy.types.UILayout.html

## Development

https://developer.blender.org/
https://developer.blender.org/docs/
https://developer.blender.org/docs/features/code_layout/

Debugging

- https://extensions.blender.org/add-ons/core-debug-tools/

### Camera internals

If the aspect ratios of image resolution and the camera sensor shape do not match,

Blender seems to pad a bit and keep all the sensor values.. ?? 🤔

```c++
/**
 * Camera Parameters:
 *
 * Intermediate struct for storing camera parameters from various sources,
 * to unify computation of view-plane, window matrix, ... etc.
 */
typedef struct CameraParams {
  /* lens */
  bool is_ortho;
  float lens;
  float ortho_scale;
  float zoom;

  float shiftx;
  float shifty;
  float offsetx;
  float offsety;

  /* sensor */
  float sensor_x;
  float sensor_y;
  int sensor_fit;

  /* clipping */
  float clip_start;
  float clip_end;

  /* computed viewplane */
  float ycor;
  float viewdx;
  float viewdy;
  rctf viewplane;

  /* computed matrix */
  float winmat[4][4];
} CameraParams;
```

```cpp
static float rna_Camera_angle_x_get(PointerRNA *ptr)
{
  Camera *cam = (Camera *)ptr->owner_id;
  return focallength_to_fov(cam->lens, cam->sensor_x);
}

static void rna_Camera_angle_x_set(PointerRNA *ptr, float value)
{
  Camera *cam = (Camera *)ptr->owner_id;
  cam->lens = fov_to_focallength(value, cam->sensor_x);
}

static float rna_Camera_angle_y_get(PointerRNA *ptr)
{
  Camera *cam = (Camera *)ptr->owner_id;
  return focallength_to_fov(cam->lens, cam->sensor_y);
}

static void rna_Camera_angle_y_set(PointerRNA *ptr, float value)
{
  Camera *cam = (Camera *)ptr->owner_id;
  cam->lens = fov_to_focallength(value, cam->sensor_y);
}
```

## Unsorted

- make the background transparent
    - `Render` > `Film` > `Transparent`
- make an object a transparent mask
    - requires the background to be transparent first
    - (2.83)
        - Add Material Slot
        - Set `Surface` as `Holdout`
    - (3.6)
        - `Object` > `Visibility` > `Mask` > `Holdout`
- attach camera to the current view
    - `N` > `View` > `Lock` > `Camera to View `
- save the rendered image when rendering the scene
    - Compositing
    - Check `Use Nodes`
    - `Add` > `Output` > `File Output`
        - set the folder name
    - Connect from the image of `Rendered Layers` to the image of `File Output`
- make an object partially transparent (without interaction)
    - Set `Material` > `Settings` > `Blend Mode` as `Alpha Blend`
    - Modify `Material` > `Surface` > `Alpha`to be less than one
- group objects
    - change it to `Object Mode`
    - shift+a > `Empty` > `Plain Axes`
    - select children objects at first and the parent object lastly
    - ctrl+p > `Object`
- add a shadow catcher
    - supported only in CYCLES
    - set the material visibility to shadow catcher in the Object properties panel
    - enable composition nodes
    - Add a mix color node
        - set the mode Mix
        - Put the background first as an input
        - Put the rendered result as the other input
    - https://youtu.be/iLj-MsnX69Y?si=ew64L0SjDeca4JuP
- use addons
    - Edit > Preferences... > Add-ons
    - e.g.
        - 3D View: Alt tab water
- use a HDRI background
    - World
    - Shader
        - Select `World` in the `Shader Type` select box
        - Add `Environment Texture` node
            - Choose a HDRI image file
            - Link the `Color` output to the `Color` input in the `Background` node
        - You may delete the default light
- use pass index for segmentation
    - https://youtu.be/xeprI8hJAH8?si=SG51QieaMcnyvyWK
    - use CYCLES
    - (for object indices)
        - `View Layer` > `Data` > `Indexes` > `Object Index` ✅
        - `Object` > `Relations` > `Pass Index`
    - (for material indices)
        - `View Layer` > `Data` > `Indexes` > `Material Index` ✅
        - `Material` > `Settings` > `Pass Index`
    - setup compositing nodes
        - `Use Nodes` ✅
        - Add `File Output`
            - `Node` in the sidebar > `Properties` > `Color` > `BW`
            - Set `Base Path`
                - (remark)
                    - The `File Output` node doesn't allow to set a specific output file name.
                    - Instead it creates a file in the specified folder with a given prefix and a number postfix.
    - Add `Math` Node
        - `Divide`
        - Set the value: 255
    - Connect `Render Layers` (`IndexOB` or `IndexMA`) - `Divide` - `File Output`
- get cartoon like non-photorealistic edges
    - https://docs.blender.org/manual/en/latest/render/freestyle/introduction.html
    - check `Render` > `Freestyle`
    - check `View Layer` > `Freestyle` > `as render pass`
        - otherwise it will render only the combined result
    - render the scene
    - you can select `freestyle` as the pass
- remeshing
    - https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/remesh.html
        - may not be useful?
    - (addons)
        - https://blender-addons.org/articles/open-source-auto-remesher/
        - https://exoside.com/
    - (etc.)
        - https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
- bake rendered materials as vertex color attribute
    - `Data` > `Color Attributes`
        - Add color attribute and select it
    - `Render` > `Render Engine` > `Cycles`
    - `Render` > `Bake` > `Output` > `Target` > `Active Color Attribute`
    - `Render` > `Bake` > `Bake`

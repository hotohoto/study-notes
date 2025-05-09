# Blender



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



## Notes

- make numpad keys available for tenkeyless keyboards
  - `Edit` > `Preferences` > `Input` > `Keyboard` > `Emulate Numpad`
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
- render the scene
  - `F12`
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
- get depth map
  - https://youtu.be/saptddljRks?si=1X8cDq3RjTbutJFf
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
  - `Render` > `Bake`  > `Bake`


## Color management

- https://docs.blender.org/manual/en/latest/render/color_management.html
- Blender internally uses linear colors that follow the physics better
  - this is different from RGB
- (remarks)
  - `.ply` may contains a RGB color for each vertex
  - When you load such a file,
    - Blender converts the colors to its own color format
    - and keeps them in a color attribute.


## A rigging example

Let's do rigging for the default cube 🙃

- subdivide the default cube
  - go to the edit mode
  - right click
  - subdivide
- add an armature
  - go to the object mode
  - Add > Armature > Single Bone
  - Show the bone
    - Data > Viewport Display > In Front ✅
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

## Constraint



### Copy location/rotation

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



## Addon development

- (references)
  - https://docs.blender.org/api/current/
  - https://docs.blender.org/api/current/info_quickstart.html
  - https://docs.blender.org/api/current/bpy.props.html
  - https://docs.blender.org/api/current/bpy.types.UILayout.html




## Edit mode

- select none
  - Alt + A
- select faces isolated around??
  - Ctrl + H
- hide selected faces
  - H
- show all faces again
  - Alt + H



## Rotation

- 



## Raycast

```py
result, location, normal, face_index = ground_mesh.ray_cast(
    camera_xyz_local, cam2point_local
)
```






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
    -  transmission
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

## Rotation

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



## Coordinate system

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




## Modifiers

apply modifiers temporarily and get vertices with respect to the world coordinates:

```py
depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = obj.evaluated_get(depsgraph)
vertices = [obj_eval.matrix_world @ v.co for v in obj_eval.data.vertices]
obj_eval.to_mesh_clear()
```





## Animation

- basics
  - add animation properties
  - add key frame
  - choose another time point
  - change the animation properties
  - add key frame
- object / armature
  - animation_data
    - nla_tracks
      - track
        - strips
          - strip
    - action
      - fcurves
        - fcurve
          - key_frame_points



## Camera internals

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



## Misc.

- attributes seems to be duplicated when subdividing the edges


# OpenUSD

## TODO

- https://www.nvidia.com/en-us/learn/learning-path/openusd/

## Glossary

- Prim
    - a node of any 3d thing.
        - e.g. object, camera, light, material, group, animation, ...
- SdfPath
    - a locator to a Prim/Property/... like URL to a web page
- Layer
    - Contains one or more Prims
    - to be saved as a part of USD
- Stage
    - Contains one or more layers
    - Constructs the final scenegraph
- Hydra
    - a modularized rendering framework
- Composition Arc
    - A mechanism to relate a layer or a prim to another
    - (types)
        - Reference
        - SubLayer
        - Payload
        - Inherit
        - Specialize
        - Variant Set
- Property
    - Attribute
    - Relationship

## Modules

### Stage

```python
Usd.Stage.CreateNew()
Usd.Stage.Open()
Usd.Stage.Save()
```

### Hydra

- Hydra scene delegate
    - e.g.
        - a USD scenegraph
- Render index
    - keeps track of changes
- Hydra renderer delegate
    - an actual Hydra rendering implementation
    - e.g.
        - HdTiny
        - HdStrom
            - used by `usdview`
        - HdEmbree
            - uses Embree
                - made by Intel
                - CPU based
            - 

## Examples

A sphere scene with animation:

```usda
#usda 1.0

def Xform "World"
{
    def Camera "MyCamera"
    {
        float focalLength = 50

        float3 xformOp:translate.timeSamples = {
            1: (0, 0, 10),
            10: (5, 0, 10),
            20: (5, 5, 10),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Sphere "MySphere"
    {
        float xformOp:translate.timeSamples = {
            1: 0,
            20: 10,
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float radius = 1
    }
}
```

A simple scene referencing an external OBJ file:

```usda
#usda 1.0
(
    doc = "A simple scene referencing an external OBJ file."
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Xform "MyCubeFromOBJ"
    (
        references = @./assets/meshes/cube.obj@
    )
    {
        float3 xformOp:translate.timeSamples = {
            0: (2.0, 0.0, 0.0),
            50: (2.0, 5.0, 0.0),
        }
        float3 xformOp:scale = (0.5, 0.5, 0.5)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }

    def SphereLight "SphereLight"
    {
        float intensity = 500
    }
}
```

## Use in Python

```bash
pip install usd-core
```

```python
from pxr import Usd, UsdGeom


stage = Usd.Stage.CreateNew("my_scene.usda")
stage.SetFramesPerSecond(24.0)


def setup_cube():
    cube = UsdGeom.Xform.Define(stage, "/World/MyCube")
    translate = cube.AddTranslateOp()
    scale = cube.AddScaleOp()

    translate.Set((0, 0, 0), time=0)
    translate.Set((5, 0, 0), time=10)
    translate.Set((5, 5, 0), time=25)
    scale.Set((2.0, 2.0, 2.0))

    return translate


def setup_camera():
    cam = UsdGeom.Camera.Define(stage, "/World/MyCamera")
    cam.AddTranslateOp().Set((0, 0, 10), time=0)

    focal_length = cam.CreateFocalLengthAttr()
    focal_length.Set(50.0, time=0)
    focal_length.Set(100.0, time=50)

    horizontal_aperture = cam.CreateHorizontalApertureAttr()
    horizontal_aperture.Set(36.0)

    vertical_aperture = cam.CreateVerticalApertureAttr()
    vertical_aperture.Set(24.0)

    return focal_length


cube_translate = setup_cube()
cam_focal = setup_camera()


def print_animation_data():
    print(f"Cube at frame 15: {cube_translate.Get(time=15)}")
    print(f"Camera focal at 30: {cam_focal.Get(time=30)}")
    print(f"Time range: {stage.GetStartTimeCode()} to {stage.GetEndTimeCode()}")
    print(f"FPS: {stage.GetFramesPerSecond()}")


stage.Save()
print_animation_data()
```

## More examples

A pinhole camera and a cube has been exported from Blender:
```usda
#usda 1.0
(
    defaultPrim = "root"
    doc = "Blender v4.4.3"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "root" (
    customData = {
        dictionary Blender = {
            bool generated = 1
        }
    }
)
{
    def Xform "Camera"
    {
        float3 xformOp:rotateXYZ = (63.559303, -0.0000026647115, 46.691948)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]

        def Camera "Camera_001"
        {
            float2 clippingRange = (0.1, 1000)
            float focalLength = 0.38868457
            float horizontalAperture = 0.36
            token projection = "perspective"
            float verticalAperture = 0.2025
        }
    }

    def Xform "Cube"
    {
        def Mesh "Cube" (
            active = true
        )
        {
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 3, 2, 2, 3, 7, 6, 6, 7, 5, 4, 4, 5, 1, 0, 2, 6, 4, 0, 7, 3, 1, 5]
            normal3f[] normals = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
                interpolation = "faceVarying"
            )
            point3f[] points = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
            bool[] primvars:sharp_face = [1, 1, 1, 1, 1, 1] (
                interpolation = "uniform"
            )
            texCoord2f[] primvars:st = [(0.375, 0), (0.625, 0), (0.625, 0.25), (0.375, 0.25), (0.375, 0.25), (0.625, 0.25), (0.625, 0.5), (0.375, 0.5), (0.375, 0.5), (0.625, 0.5), (0.625, 0.75), (0.375, 0.75), (0.375, 0.75), (0.625, 0.75), (0.625, 1), (0.375, 1), (0.125, 0.5), (0.375, 0.5), (0.375, 0.75), (0.125, 0.75), (0.625, 0.5), (0.875, 0.5), (0.875, 0.75), (0.625, 0.75)] (
                interpolation = "faceVarying"
            )
            uniform token subdivisionScheme = "none"
        }
    }

    def DomeLight "env_light"
    {
        float inputs:intensity = 1
        asset inputs:texture:file = @.\textures\color_121212.hdr@
        float3 xformOp:rotateXYZ = (90, 1.2722219e-14, 90)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }
}
```

Not a panoramic camera but just the transform is exported from Blender:

```
#usda 1.0
(
    defaultPrim = "root"
    doc = "Blender v4.4.3"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "root" (
    customData = {
        dictionary Blender = {
            bool generated = 1
        }
    }
)
{
    def Xform "Camera"
    {
        float3 xformOp:rotateXYZ = (63.559303, -0.0000026647115, 46.691948)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
}
```

Define a LiDAR sensor

```usda
#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Xformable "LiDAR_Sensor" (
    "Custom LiDAR Sensor Primitive"
    kind = "component"
)
{
    custom float lidar:angularResolution = 0.1
    custom float lidar:horizontalFov = 360
    custom float lidar:maxRange = 100
    custom float lidar:minRange = 0.1
    custom float lidar:rotationSpeed = 600
    custom string lidar:scanType = "rotating"
    custom float lidar:verticalFov = 30
    double3 xformOp:translate = (0, 0, 2.5)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def Mesh "Ground"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-10, -10, 0), (-10, 10, 0), (10, 10, 0), (10, -10, 0)]
}
```

## usdview

- https://developer.nvidia.com/usd
    - download Pre-Built OpenUSD Libraries and Tools
- https://docs.omniverse.nvidia.com/usd/latest/usdview/quickstart.html
    - run `scripts/useview-gui.bat` by clicking it in file explorer

## References

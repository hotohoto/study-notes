[TOC]

# Godot Engine

## TODO

https://youtu.be/_QHvKMRtJD0
  - area
  - gravity
  - rigidbody player with animation

## Getting started

## Manual 

### Best practices

#### When and how to avoid using nodes for everything

- Object
  - no garbage collection
  - a script can be attached

- RefCounted
  - e.g.
    - FileAccess
- Resource
  - serializable

#### Godot interfaces

#### Godot notifications

- _init()
- _notification()
- _ready()
- _enter_tree()
- _exit_tree()
- _process()
- _physics_process()
- _draw()
- _input()
- _unhandled_input()
- _unhandled_key_input()
- _gui_input()
- _shortcut_input()

#### Data preferences

https://docs.godotengine.org/en/stable/tutorials/best_practices/data_preferences.html

- [1]
  - array
  
    - `Vector<Variant>`
  
    - dictionary
      - `OrderedHashMap<Variant, Variant>`
  
  - object
    - structured
  
- [2]
  - enum with int
  - enum with string
    - https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_exports.html
  
- [3]

  - `Animation`
    - a `RefCounted`
  - `AnimationLibrary`
    - a `RefCounted`
    - a simple animation container
  - `AnimatedTexture`
    - a `Resource`
  - `SpriteFrames`
    - a `Resource`
  - `AnimatedSprite2D`
  - `Sprite2D`
  - `AnimatedSprite3D`
  - `Sprite3D`
  - `AnimationMixer`
    - inherited by `AnimationPlayer` and `AnimationTree`
    - has an `AnimationLibrary`
  - `AnimationPlayer`
    - uses an `AnimatedSprite2D` / `AnimatedSprite3D`
  - `AnimationTree`
    - uses an `AnimationPlayer`

(References)

- (language basics)
  - https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_basics.html
- (variant types)
  - https://docs.godotengine.org/en/stable/contributing/development/core_and_modules/variant_class.html



#### Logic preferences

https://docs.godotengine.org/en/stable/tutorials/best_practices/logic_preferences.html

- Set properties before adding their nodes to the node tree
- loading vs preloading
  - `const` is for `preload`
  - `var` is for `load`
- static vs dynamic
  - use static level for smaller games
  - use dynamic logics for medium/large games



#### Project organization

- https://docs.godotengine.org/en/stable/tutorials/best_practices/project_organization.html
- project: godot-demo-projects
- node: PascalCase
- file/folder: snake_case

#### Version control systems

- https://docs.godotengine.org/en/stable/tutorials/best_practices/version_control_systems.html



### 3D

#### Introduction to 3D

https://docs.godotengine.org/en/stable/tutorials/3d/introduction_to_3d.html

#### Optimization

##### Using MultiMeshInstance3D

https://docs.godotengine.org/en/latest/tutorials/3d/using_multi_mesh_instance.html

- instances GeometryInstance3Ds
  - based on MultiMesh
- faster than MeshInstance3D
- target - the landscape
- source - tree



##### Mesh level of detail (LOD)

##### Visibility ranges (HLOD)

##### Occlusion Culling

##### Resolution scaling

##### Variable rate shading



### Asset pipeline

#### Import process

https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/import_process.html

- automatically imported resources are at `res://.godot/imported/`



### Scripting

#### Core features

##### Using `SceneTree`

https://docs.godotengine.org/en/stable/tutorials/scripting/scene_tree.html

## Contributing

### Engine development

#### Engine architecture

##### Engine core and modules

###### Godot's architecture diagram

![architecture_diagram](https://docs.godotengine.org/en/stable/_images/architecture_diagram.jpg)







## Class Reference

### Nodes

#### GeometryInstance3D

- https://docs.godotengine.org/en/stable/classes/class_geometryinstance3d.html

#### MeshInstance3D(GeometryInstance3D)

- https://docs.godotengine.org/en/stable/classes/class_meshinstance3d.html
- takes a Mesh

#### MultiMeshInstance3D(GeometryInstance3D)

- https://docs.godotengine.org/en/stable/classes/class_multimeshinstance3d.html
- instances a MultiMesh

### Resources

#### Mesh

- contains surfaces
- a surface represents a single material
- https://docs.godotengine.org/en/stable/classes/class_mesh.html

#### MultiMesh

- Provides high-performance drawing of a mesh multiple times using GPU instancing
- https://docs.godotengine.org/en/stable/classes/class_multimesh.html

### Other objects

#### Object

https://docs.godotengine.org/en/stable/classes/class_object.html

- no garbage collection
- a script can be attached

```
var node = Node.new()
print("name" in node)         # Prints true
print("get_parent" in node)   # Prints true
print("tree_entered" in node) # Prints true
print("unknown" in node)      # Prints false
```



#### MainLoop

- defines a single step to process for the game engine's while loop

#### OS

https://docs.godotengine.org/en/stable/classes/class_os.html

#### SceneTree(MainLoop)

https://docs.godotengine.org/en/stable/classes/class_scenetree.html



### Variant types

#### AABB

- A 3D axis-aligned bounding box.



## Godot Engine source codes

- main()
  - (platform/linuxbsd/godot_linuxbsd.cpp)
  - Main::setup()
  - Main::start()
  - os.run()
    - (OS_LinuxBSD::run())
    - while (true)
      - Main::iteration()
        - time_scale = Engine::get_singleton()->get_time_scale()
        - OS::get_singleton()->get_main_loop()->physics_process(physics_step * time_scale)
          - (scene_tree.cpp)
          - (SceneTree::physics_process(double p_time))
            - MainLoop::physics_process(p_time)
              - GDVIRTUAL_CALL(_physics_process, p_time, quit)
                - (calls _physics_process implemented in gdscripts)
            - SceneTree::_process(bool p_physics)
              - SceneTree::_process_group(ProcessGroup *p_group, bool p_physics)
                - (notify events to the nodes)
        - OS::get_singleton()->get_main_loop()->process(process_step * time_scale)
          - (scene_tree.cpp)
          - (SceneTree:process(double p_time))
          - MainLoop::process(p_time)
            - GDVIRTUAL_CALL(_process, p_time, quit)
              - (calls _process implemented in gdscript )
          - SceneTree::_process(bool p_physics)
            - SceneTree::_process_group(ProcessGroup *p_group, bool p_physics)
              - (notify events to the nodes)
  - Main:cleanup()



## Notes

### headless mode

TBD

### offscreen rendering (not implemented)

https://github.com/godotengine/godot-proposals/issues/5790



### gdscripts

- `_process(delta)`
  - `delta`
    - time it took for Godot to complete the previous frame in seconds
- scene can be instantiated as a node
- Scripts attach to a node and extend its behavior



### 2d game reference

https://youtu.be/WEt2JHEe-do

- animation
- set_deffered()
  - set attributes safely.
- export a property
  - expose the property to the editor
- screen size manipulation
- emit_signal
- randomize
  - set a random seed
- randi()
- rand_range()
- path
- yield
  - wait for signal
- get_tree()
  - gets entire nodes tree
- get_tree().call_group("mobs", "queue_free")
- queue_free()
- xxxx.instantiate()
- poisition node
  - only to keep a position
- AudioStreamPlayer

### Physics

- collision layers
- collision masks
- CollisionObject2D (physics objects)
  - props
    - collision_layer
    - collision_mask
  - children
    - Area2D
    - PhysicsBody2D
      - children
        - StaticBody2D
        - RigidBody2D
          - it's hard to check if the body is on the floor
            - contacting/collision is not accurate when it's sliding
            - when using collision signal, it's hard to tell if the contact point is on the floor or on the side of the player
            - when using ray cast, it's hard to tell up to how much distance is far enough to be on the floor.
              - Especially, on the slope, raycast needs to be done multiple times.
        
        - CharacterBody2D
          - more popular than RigidBody2D for the controllable players
          - well supported to tell if it's on the floor and to move sliding slopes
          - difficult to simulate the collision effect?
- (collision shapes)
  - CollisionShape2D
    - props
      - shape: Shape2D
  - CollisionPolygon2D
    - props
      - polygon: PackedVector2Array
- Shape2D
  - children
    - CapsuleShape2D
    - CircleShape2D
    - ConcavePolygonShape2D
    - ConvexPolygonShape2D
    - RectangleShape2D
    - SegmentShape2D
    - SeparationRayShape2D
    - WorldBoundaryShape2D

### 3d game reference

https://youtu.be/YiE9tcoCfhE

- .glb for models
- .tres for what??


### settings

- default project path

### signal

- defined in the signal source script file

### view point navigation

- F : focus on the current object
- shift F: toggle first person view mode
  - move around with WASD
    - press shift to go faster

### terrain

- addon 설치
  - data folder 지정
  - opengl es 3.x 사용



### Reference folder structure

- https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html
  - folder/file: snake_case
  - class: PascalCase

- https://youtu.be/4az0VX9ApcA?si=Qu-jZrRLweoS_anN&t=436
- https://youtu.be/kH5HkKNImXo?si=dBIdpveACGUBzcJW



- assets
  - audio
    - beach_shop_theme.wav
    - dauphin_title_theme.wav
  - credits
    - credits.gd
  - fonts
    - comfortaa
    - custom
    - dive_monitor
    - m5x7.ttf
- common
  - animations
  - collisions
  - lights
  - loot
  - properties
  - resolution_management
    - interface_scaler
      - interface_scaler.gd
      - interface_scaler.tscn
  - shaders
  - shutdowns
  - state_management
    - states
      - chase
      - fear
      - idle
      - ready
      - wander
      - state.gd
    - state_machine.gd
    - state_machine.tscn
  - time_manager
    - time_manager.gd
    - time_manager.tscn
  - tooltips
  - transitions
  - ui
- config
  - preferences.cfg
  - preferences.gd
- entities
  - atacks
  - bosses
  - camera
  - climbing 
    - climbing_bar
    - climbing_hold
    - climbing_node
    - climbing_route
    - climbing_vignette
  - corruption
  - crafting
  - environment
  - fish
  - foraging
  - items
    - consummables
    - equipment
    - forageable
    - placeable
    - tools
      - fishing
      - foraging
        - axes
        - pickaxes
        - foraging_tool.gd
        - foragingTool.tscn
        - foraging_tool_item.gd
      - other
      - weapons
      - tool.gd
      - tool.tscn
      - tool_item.gd
        - (extends Item)
    - item.gd
  - npcs
  - organisms
    - amphibians
    - birds
    - fish
    - fungi
    - invertebrates
      - jellyFish
      - coral_head
      - sand_crab
        - art
          - sand-crab-single.png
          - ...
        - data
          - sand_crab_organism_data.tres
          - sand_crab_research_tier1.tres
          - sand_crab_research_tier2.tres
        - sound
          - aggro
            - sand-crab-aggro.wav
          - hit
            - sand-crab-hit.wav
        - sand_crab_organism.gd
        - sand_crab_organism.tscn
      - tube_coral
  - players
  - ships
  - spawner
  - terrain
- localization
- stages
  - devSites
  - fishermanHut
  - islands
  - ocean
  - tilesets
  - title
  - underwater
- utilities

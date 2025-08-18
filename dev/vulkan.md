# Vulkan

https://kylemayes.github.io/vulkanalia

## 1 Introduction

## 2 Overview (WIP)

### Terms

- display pipeline
    - put geometries - model polygons - into world view screen space
    - put textures
    - apply some effects
    - moved into frame buffer
- offscreen rendering

### Origin of Vulkan

https://kylemayes.github.io/vulkanalia/overview.html

- With Vulkan API we can
    - utilize multi threading
    - control tiled rendering for mobile devices
        - https://en.wikipedia.org/wiki/Tiled_rendering
            - tiles sizes
                - 16x16 or 32x32 usually
            - assign geometries to each tile
                - maybe leveraging modern GPU's supports
            - opposite to immediate mode
    - compile shader codes in a consistent way

### What it takes to draw a triangle

Steps to take in a well-behaved Vulkan program
- step 1
    - create a `VkInstance`
        - describes my application
        - describes APIs to use
    - choose one or more `VkPhysicalDevice`s
- step 2
    - create a logical device called `VkDevice`
        - with `VkPhysicalDeviceFeatures`
    - `VkQueue`
        - for executing operation requests asynchronously
        - there could be separate queue family
- step 3 - window surface and swap chain
    - native library to creating window surfaces
        - GLFW
        - SDL
        - winit
            - written in Rust
    - 

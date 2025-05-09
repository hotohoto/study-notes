# Physically based rendering

https://www.pbr-book.org/4ed/contents



## 1 Introduction

### 1.1 Literate Programming

- tell computer what we want to do rather than telling what computer do
- source codes can contain `<<fragment>>`

### 1.2 Photorealistic Rendering and the Ray-Tracing Algorithm

- make it indistinguishable from a photo graph
  - depends on observer
- it's enough to regard light as particles rather than waves
- Ray-tracer
  - camera
    - "Many rendering systems generate viewing rays starting at the camera that are then traced into the scene to determine which objects are visible at each pixel."
  - Ray-object intersections
  - Light sources
  - Visibility
  - Light scattering at surfaces
  - Indirect light transport
  - Ray propagation

#### 1.2.1 Cameras and film

- viewing volume
  - the region of space that can potentially be imaged on the film
- "Most camera sensors record separate measurements for **three wavelength distributions** that correspond to red, green, and blue colors, which is sufficient to reconstruct a scene’s visual appearance to a human observer."

(notes)

- Humans have three types of cone cells L/M/S
  - that's why we represent colors with three independent channels

#### 1.2.2 Ray-Object Interactions

(TODO)

https://www.pbr-book.org/4ed/Introduction/Photorealistic_Rendering_and_the_Ray-Tracing_Algorithm#RayndashObjectIntersections

## 2 Monte Carlo Integration

## 3 Geometry and Transformations

## 4 Radiometry, Spectra, and Color

## 5 Cameras and Films

## 6 Shapes

## 7 Primitives and Intersection Acceleration 

## 8 Sampling and Reconstruction

## 9 Reflection Models

## 10 Textures and Materials

## 11 Volume Scattering

## 12 Light Sources

## 13 Light Transport I: Surface Reflection

## 14 Light Transport II: Volume Rendering

## 15 Wavefront Rendering on GPUs

## 16 Retrospective and the Future

## A Sampling algorithms

## B Utilities

## C Processing the Scene Description

## References
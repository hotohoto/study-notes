Here’s the full depth map reference in English, cleaned up and structured clearly:

---

# Depth maps overview

## 1. Metric Depth

- The actual physical distance from the camera to the point in the scene.
- Usually in meters (m), but can vary (e.g., centimeters for some LiDARs).
- Directly corresponds to real-world scale.
- Useful for tasks like SLAM, 3D reconstruction, and AR.

## 2. Affine-Invariant Inverse Depth

- Depth values are represented as 1z\frac{1}{z}, invariant to affine transformations.
- Used in:
    - [Depth Anything v1 & v2](https://github.com/DepthAnything/Depth-Anything-V2/issues/93#issuecomment-2239118769)
    - https://github.com/heyoeyo/muggled_dpt/blob/main/.readme_assets/results_explainer.md
    - Models like MiDaS, DPT also follow this principle.

## 3. Z-buffer (not sure)

- Normalized z-value in camera space (not Euclidean distance).
- [Reference: Wikipedia - Z-buffering](https://en.wikipedia.org/wiki/Z-buffering)

## 4. Blender (not sure)

- [Blender Manual – Render Passes](https://docs.blender.org/manual/en/latest/render/layers/passes.html#cycles)
    
- `Z`:
    - Represents Euclidean distance from the camera to the object surface.
    - Not normalized.
    - Suitable for generating ground truth metric depth maps.
- `Mist`:
    - A normalized fog/depth mask used for post-processing effects.
    - Not true depth; interpolated between start and end distance.

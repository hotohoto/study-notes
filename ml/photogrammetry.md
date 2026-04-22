# Photogrammetry

- localization using a matcher model
- infer fundamental matrix
- self-calibration
    - infer intrinsic parameters without checkerboard calibration
- bundle adjustment
- registration

## 3D reconstruction

- fundamental matrix
- essential matrix
- projective reconstruction
- affine reconstruction
- metric reconstruction

## Self-calibration

- find matches using a matcher
    - https://github.com/Vincentqyw/image-matching-webui
    - https://github.com/Parskatt/DeDoDe
- findFundamentalMat()
    - get F for each image pair
- findEssentialMat()
- set an initial intrinsic parameters
    - $c_x = {W \over 2}$
    - $c_y = {H \over 2}$
    - use a random value
        - e.g. $f = \text{width}$
    - use Kruppa's equation
        - requires F values from many images as inputs
- correctMatches()
    - enhance matches using epipolar geometry
- decomposeEssentialMat()
    - get the essential matrix
- recoverPose()
    - get the relative rotation and translation values
- triangulatePoints()
    - make the point cloud
- do bundle adjustment
    - enhance $f$ and points in the cloud

## References

(blogs)
- https://scimad.github.io/2020/09/06/transformation-and-3d-reconstruction

(textbooks)
- 3D Reconstruction
    - https://diposit.ub.edu/server/api/core/bitstreams/47e1cd42-f9e0-4cb6-836d-0439844aca04/content
- Multiple View Geometry in Computer Vision
    - https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A

(models)
- https://github.com/Parskatt/DeDoDe ⭐
- https://github.com/Parskatt/RoMa/
- https://github.com/fraunhoferhhi/RIPE

(Good open projects)
- https://github.com/Vincentqyw/image-matching-webui ⭐
- https://github.com/cvlab-epfl/multiview_calib/ ⭐
- https://github.com/3DOM-FBK/deep-image-matching ⭐
- https://github.com/cvg/Hierarchical-Localization

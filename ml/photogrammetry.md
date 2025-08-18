# Photogrammetry

- localization using a matcher model
- infer fundamental matrix
- self-calibration
    - infer intrinsic parameters without checkerboard calibration
- bundle adjustment
- registration
- 

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

(models)
- https://github.com/Parskatt/DeDoDe ⭐
- https://github.com/Parskatt/RoMa/
- https://github.com/fraunhoferhhi/RIPE
(Good open projects)
- https://github.com/Vincentqyw/image-matching-webui ⭐
- https://github.com/cvlab-epfl/multiview_calib/ ⭐
- https://github.com/3DOM-FBK/deep-image-matching ⭐
- https://github.com/cvg/Hierarchical-Localization

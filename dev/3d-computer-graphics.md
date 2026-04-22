# 3D computer graphics

## 3D model simplification/decimation

- https://meshoptimizer.org/
    - https://github.com/zeux/meshoptimizer
    - https://github.com/gwihlidal/meshopt-rs
    - https://www.npmjs.com/package/gltfpack

## 3D model compression

- https://github.com/google/draco
- https://github.com/donmccurdy/glTF-Transform

```
# Draco (compresses geometry).
gltf-transform draco input.glb output.glb --method edgebreaker

# Meshopt (compresses geometry, morph targets, and keyframe animation).
gltf-transform meshopt input.glb output.glb --level medium
```

- https://d2.naver.com/helloworld/6152907

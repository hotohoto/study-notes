# 3d-tiles

- https://github.com/CesiumGS/3d-tiles/
- https://github.com/CesiumGS/3d-tiles-tools
- open
- heterogeneous
    - photogrammetry
    - BIM/CAD
    - 3d buildings
    - point cloud
- uses glTF as its leaf nodes
- deprecated formats:
    - b3dm / i3dm / pnts / cmpt
- refinement modes
    - REPLACE
    - ADD
- spatial partitioning
    - quadtree
        - 2.5d
    - octree
        - 3d
    - K-d tree
    - implicit tiling
- screen space error (SSE)
    - SSE = (geometricError × viewportHeight) / (distance × 2 × tan(fov/2))
    - geometric error (in meter)
        - Hausdorff distance
            - max distance difference
        - Quadric error
        - Sample spacing

## PROS

- use in top-down

## CONS

- build in bottom-up

## Example workflow

(preprocessing)
- prepare building mesh files for the entire city (10GB)
- partition the space using quadtree
- a single leaf tile contains all the building in the area (several MB)
- a parent tile is the decimated version of the children meshes
- `tileset.json` contains the tree structure
- upload all the static files prepared to the server
- provide the URL of the `tileset.json` to a Cesium client

(client)
- load the `tileset.json`
- for each frame
    - (traverse the tree from the root)
    - for each tile
        - skip if it's out of frustum
        - calculate SSE
        - if SSE > threshold:
            - add the child node to the download queue
        - else:
            - (don't take a look at the descendants of this tile)
    - download the closest tile first
    - evict the tiles not in use from the LRU cache

## Implicit tiling

- don't use
- can reduce the size of `tileset.json`

(before)

```json
{
  "root": {
    "boundingVolume": {...},
    "geometricError": 500,
    "content": {"uri": "0/0/0.glb"},
    "children": [
      {
        "boundingVolume": {...},
        "geometricError": 250,
        "content": {"uri": "1/0/0.glb"},
        "children": [
          {
            "boundingVolume": {...},
            "geometricError": 125,
            "content": {"uri": "2/0/0.glb"},
            "children": [
              ...
            ]
          },
          ...
        ]
      },
      ...
    ]
  }
}
```

(after)

```json
{
  "asset": {"version": "1.1"},
  "geometricError": 500000,
  "root": {
    "boundingVolume": {
      "region": [-3.14, -1.57, 3.14, 1.57, 0, 1000]
    },
    "geometricError": 500000,
    "refine": "REPLACE",
    "content": {
      "uri": "content/{level}/{x}/{y}.glb"
    },
    "implicitTiling": {
      "subdivisionScheme": "QUADTREE",
      "subtreeLevels": 7,
      "availableLevels": 22,
      "subtrees": {
        "uri": "subtrees/{level}/{x}/{y}.subtree"
      }
    }
  }
}
```

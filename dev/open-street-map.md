# OpenStreetMap

- iD
    - the name of the editor
- elements
    - can have tags
    - (types)
        - node
            - point
        - way
            - line
            - area
        - relation
- map feature
    - https://wiki.openstreetmap.org/wiki/Map_features

## Road

- 

## Building

- https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings

## Elevation

- https://wiki.openstreetmap.org/wiki/SRTM
- https://wiki.openstreetmap.org/wiki/Open-Elevation
    - https://open-elevation.com/
    - https://github.com/Jorl17/open-elevation/blob/master/docs/host-your-own.md
- https://wiki.openstreetmap.org/wiki/Altitude

## OSMnx

- `features_from_polygon()`
    - nodes
    - ways including area elements
    - split by osm id
- `graph_from_polygon(..., simplify=False)`
    - nodes for graph
    - ways without area elements
    - split by everything
        - many edges for single osm id
        - each edge has only two points
- `graph_from_polygon(..., simplify=True)`
    - split by junction
- `graph_to_gdfs()`
    - graph to nodes and edges
- `graph_from_gdfs()`
    - nodes and edges to graph
    - (not compatible with the out of `features_from_polygon()`)

## Remarks

## Best places

https://www.bestofosm.org/

- New York
- Estonia
- Mainz
- Amsterdam

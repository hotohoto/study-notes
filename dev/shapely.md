# Shapely tips

## make_valid

```python
polygon = make_valid(polygon)
```

## Robust intersection

- Errors
    - `TopologyException: non-noded intersection`
    - `IllegalArgumentException: mixed-dimension`
- Shapely relies on the C++ **GEOS** library for set operations (intersection, union, difference)
- Do
    - Clean geometries via `buffer(0)` to match the underlying dimensions
        - it makes a point to an empty Polygon
    - snap coordinates using `set_precision(geom, grid_size)`
    - execute `intersection(..., grid_size=grid_size)`.

## Sample points in a polygon

- triangulate the polygon and keep them
    - https://github.com/drufat/triangle
- sample a triangle by area first
- sample a point in the triangle using

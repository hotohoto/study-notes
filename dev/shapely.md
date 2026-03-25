# Shapely tips

## make_valid

```python
polygon = make_valid(polygon)
```

## Sample points in a polygon

- triangulate the polygon and keep them
    - https://github.com/drufat/triangle
- sample a triangle by area first
- sample a point in the triangle using

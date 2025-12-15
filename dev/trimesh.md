# Trimesh

- mesh.vertices
    - vertex coordinates in (v,3)
- mesh.faces
    - face vertices in (f,3)
    - consists of vertex indices
    - for mapping faces to vertices
- mesh.edges
    - face edges in (3f, 2)
    - consists of vertex indices
- mesh.edges_unique
    - edges in (e, 2)
    - consists of vertex indices
    - vertices are sorted in the ascending order
- mesh.edges_unique_inverse
    - edges in (3f,)
    - consists of unique edge indices
    - for mapping from edges to unique edges

# Linear algebra

## linear transformation

- linear map
  - (L, +, ·), (W, +, ·)
  - $L: V \rightarrow W$
  - additivity
    - $L(u + v) = L(u) + L(v)$
  - homogeneity
    - $L(kv) = kL(v)$
    - $k \in F$
  - kernel
    - $\text{ker} L = L^{-1}(0) = \{v \in V | L(v) = 0\}$
  - image
    - $\text{img} L = L(V) = \{L(v) \in W | v \in V\}$
- endomorphism
  - $V = W$
  - ex) $R^3 \rightarrow R^3$
- monomorphism (or injective)
  - $L(w) = L(v) \Rightarrow w = v$
- epimorphism (or surjective)
  - $L(V) = W$
  - 치역(range) = 공역(codomain)
- isomorphism
  - monomorphism + epimorphism
- automorphism
  - isomorphism + endomorphism
- inverse map
  - 2 sided inverse map (left inverse map + right inverse map)
- identity map
  - $L(v) = v$
  - $L = I_v$
- inverse map
  - left inverse map + right inverse map

## matrix similarity

https://en.wikipedia.org/wiki/Matrix_similarity

properties of similar matrices
- same trace
- same determinant
- same rank
- same eigen values

## diagonalization

- https://en.wikipedia.org/wiki/Diagonalizable_matrix

- diagonalizable
  - Unitarily Diagonalizable
    - orthogonally diagonalizable
- defective over real and complex entries
  - non diagonalizable for those entries

- Schur form
  - $T = U^{*}AU$
    - $T$: Upper triangualr
- Jordan normal form or Jordan canonical form
  - $J = P^{−1}AP$
  - whether it's defective or not (over real and complex entries)
- diagonalizable
  - $D = P^{-1}AP$
- diagonalizable by a unitary matrix
  - $D = U^{*}AU$
- orthogonally diagonalizable
  - $D = P^{T}AP$

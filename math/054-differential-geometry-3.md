# Differential geometry - intermediate

## 0. Scope and usage

1. geometric objects (tangent, cotangent, tensor)
2. operators (connection, covariant derivative, exterior derivative)
3. invariants (curvature)
4. global theorem (Stokes)

## 1. Learning objectives

- tangent and cotangent spaces on manifolds
- tensor objects vs coordinate-dependent symbols
- geodesic equations and a metric
- Riemann, Ricci, and scalar curvature
- one-forms, exterior derivative, and Stokes theorem

## 2. Prerequisites check

- multivariable calculus and coordinate change
- linear algebra: dual space, bilinear map, basis change
- basic manifold language: chart, tangent vector, smooth map
- index notation familiarity

Related notes:

- [[020-calculus]]
- [[021-multivariate-calculus]]
- [[031-linear-algebra]]
- [[040-mathematical-analysis]]

## 3. Core narrative

### 3.1 Curvilinear coordinates and metric

- Orthogonal curvilinear coordinates satisfy
    - $g_{ii} = h_i^2$
    - $h_i = \left| {\partial x_i \over \partial q_i} \right|$
- Metric tensor $g$ turns geometric questions into coordinate calculations.
- A Riemannian manifold is a manifold equipped with a positive-definite metric.

One-line summary:
The metric is the bridge from geometry to computation.

### 3.2 Tangent, cotangent, and tensor basics

- Tangent space at $p$: $T_pM$
- Cotangent space: $T_p^*M$
- One-form is an element of $T_p^*M$, a linear map from tangent vectors to real numbers.
- A $(q,r)$ tensor is a multilinear map
    - $T^{(q,r)}: (V^*)^q \times V^r \to \mathbb{R}$
- Tensor operations used repeatedly:
    - tensor product
    - contraction
    - index raising and lowering via metric

One-line summary:
Tensors encode coordinate-independent multilinear structure on tangent and cotangent data.

### 3.3 Connection and covariant derivative

- Affine connection defines how to differentiate tensor fields on manifolds.
- Christoffel symbols are coordinate expressions of a connection, not tensor components.
- Covariant derivative generalizes directional derivative to curved spaces.
- Parallel transport compares vectors in nearby tangent spaces.

Geodesic equation:

$$
{d^{2}x^{\lambda} \over dt^{2}} + \Gamma^{\lambda}_{\mu\nu}{dx^{\mu} \over dt}{dx^{\nu} \over dt}=0
$$

One-line summary:
Connection tells us how vectors and tensors vary along directions on manifolds.

### 3.4 Levi-Civita connection

Given a metric, Levi-Civita connection is the unique connection that is:

- torsion-free
- metric-compatible

Coordinate formula:

$$
\Gamma^m_{jk} = {1 \over 2} g^{im}(\partial_k g_{ij} + \partial_j g_{ki} - \partial_i g_{jk})
$$

One-line summary:
Levi-Civita is the canonical metric-induced connection in Riemannian geometry.

### 3.5 Curvature hierarchy

- Riemann curvature tensor measures noncommutativity of covariant derivatives and holonomy effects.
- Ricci tensor is a contraction of Riemann tensor.
- Scalar curvature is the trace of Ricci tensor, a single-number curvature summary.

Useful memory map:

- Riemann: full local curvature information
- Ricci: averaged directional volume distortion
- Scalar: coarse total curvature indicator

One-line summary:
Curvature is a layered compression from full tensor information to global scalar signal.

### 3.6 Differential forms and Stokes

- Differential $k$-forms are antisymmetric covariant tensor fields.
- Wedge product builds higher-order antisymmetric forms.
- Exterior derivative $d$ satisfies $d^2 = 0$.
- Pullback transports forms through smooth maps.
- Stokes theorem unifies gradient, curl, divergence viewpoints.

General Stokes theorem:

$$
\int_{\partial \Omega} \omega = \int_{\Omega} d\omega
$$

One-line summary:
Differential forms provide a coordinate-light calculus unified by exterior derivative and Stokes theorem.

## 4. Worked example roadmap

- Example A: polar coordinates metric and Christoffel symbols
- Example B: geodesics on plane vs sphere
- Example C: one-form pullback under coordinate map
- Example D: verify a simple Stokes theorem instance in $\mathbb{R}^3$

## 5. Exercises

### Basic

1. Compute metric components in polar coordinates.
2. Show a one-form acts linearly on tangent vectors.
3. Derive one nonzero Christoffel symbol for a curved coordinate system.

### Intermediate

1. Derive geodesic equations on a sphere from its metric.
2. Compute Ricci tensor from a low-dimensional Riemann tensor example.
3. Show $d^2 = 0$ for a sample one-form.

### Challenge

1. State and prove a local-to-global interpretation of Stokes theorem for a simple manifold patch decomposition.

## 6. Glossary and keyword index

### Coordinate and manifold keywords

- section
- inclusion map
- embedded hypersurface
- pullback metric
- tangent space
- tangent bundle
- cotangent space
- manifold
- isometry

### Tensor keywords

- vector: $(1,0)$ tensor
- covector or one-form: $(0,1)$ tensor
- $(0,2)$ tensor, symmetric and antisymmetric cases
- bilinear map, multilinear map
- tensor field
- tensor product, dyadic product
- tensor contraction
- raising and lowering indices
- scalar product, vector product in $\mathbb{R}^3$

### Connection and derivative keywords

- affine connection
- Christoffel symbol
- Levi-Civita connection
- covariant derivative
- second covariant derivative
- Lie bracket
- torsion tensor
- torsion free condition
- differentiation in tensor fields
- parallel transport
- geodesic
- geodesic deviation

### Curvature and geometry keywords

- Riemann curvature tensor
- Ricci tensor
- Ricci scalar
- holonomy
- first fundamental form
- second fundamental form
- Gauss theorema egregium
- Weingarten map or shape operator
- Voss-Weyl formula

### Algebraic and notation keywords

- permutation and parity
- Levi-Civita symbol
- volume tensor
- pseudotensor

## 7. TODO

- write full Example A and Example B
- add one complete proof sketch for metric compatibility
- add one complete proof sketch for Stokes theorem in local chart language
- review notation consistency for index placement

## 8. References

Main:

- [Tensor Calculus by eigenchris](https://youtube.com/playlist?list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)

Extra:

- https://en.wikipedia.org/wiki/Metric_tensor
- https://en.wikipedia.org/wiki/Affine_connection
- https://en.wikipedia.org/wiki/Tensor_product
- https://en.wikipedia.org/wiki/Leibniz_integral_rule
- https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields
- https://www.mathematik.hu-berlin.de/~wendl/pub/connections_chapter6.pdf
- https://www.ese.wustl.edu/~nehorai/Porat_A_Gentle_Introduction_to_Tensors_2014.pdf
- https://einsteinrelativelyeasy.com/

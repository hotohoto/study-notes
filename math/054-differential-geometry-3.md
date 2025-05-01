# Differential geometry - intermediate

## TODO

- section
- inclusion map
  - https://en.wikipedia.org/wiki/Inclusion_map
- embedded hypersurface
  - https://en.wikipedia.org/wiki/Hypersurface
- pullback metric
  - https://en.wikipedia.org/wiki/Pullback_(differential_geometry)
- 엔지니어를 위한 미분기하 5/5
  - (geodesic 공부하거나 영상 보고 나서 보기)
    - [14 Tensor Calculus 15: Geodesics and Christoffel Symbols (extrinsic geometry)](https://youtu.be/1CuTNveXJRc)
    - [Tensor Calculus 16: Geodesic Examples on Plane and Sphere](https://youtu.be/8sVDceI70HM)
  - https://youtu.be/9Nc4sRj7L9g?t=2080



## Tensor

https://chatgpt.com/share/68138f35-020c-8008-80cc-9cc93cf78b6d

- covariant derivative
- contravariant
- vector
  - (1,0) tensor
- covectors
  - (0,1) tensor
  - linear map
    - W: V ➜ R
  - also known as linear form, linear functional, dual vectors or one-form
    - https://en.wikipedia.org/wiki/Linear_form
  - dual space of dual space is the original space
  - notation can be either `w(v)` or `<w, v>`
    - note that this is not a inner product since they are defined in different vector spaces
- (0,2) tensor
  - map
    - T: V × V ➜ R s.t. bilinear
    - T(u,v) = r ∊ R
  - symmetric tensor
    - T(u,v) = T(v,u)
  - antisymmetric
    - T(u,v) = - T(v,u)
  - inner product
    - $g_{ij} = g_{ji}$
    - $\det(g_{ij}) \neq 0$
    - $g(v,v) \ge 0$
    - $g(v,v) = 0$ iif $v = 0$
      - the last one is not necessary for Einstein's general relativity
  - "partial insertion"
    - $T(\cdot, v) \in V^*$
      - $T: V \to V^*$
- (q,r) tensor
  - $T^{(q,r)}: (V^*)^q × (V)^r \to R$
- tangent space
  - $T_pM$
  - the dimension is the same as of the manifold
- tangent bundle
  - $TM$
  - disjoint union of all tangent spaces
  - The tangent bundle is also a manifold.
- cotangent space
  - the dual space of the tangent space
- dual space
- tensor
  - an algebraic object that describes a multilinear relationship between sets of algebraic objects related to a vector space
  - describes a multilinear relationship between sets of algebraic objects related to a vector space
    - not to fix points
  - It doesn't depend on coordinates (or charts).
- linear map
- bilinear map
- multilinear map
- tensor contraction
- lowering of tensors
- raising of tensors
- scalar product
- vector product
  - defined only in $R^3$
- tensor product
- dyadic product
  - a type of tensor product
  - takes 2 vectors
  - returns dyadic
    - a second order tensor
- triad of vector tangent
  - $\vec{g}_i = {\partial \vec{r} \over \partial x^i}$
  - 그레디언트는 이와 비슷하나 x 의 아래 첨자를 씀
- permutation/parity
- Levi-Civita symbol
  - $\varepsilon _{i_{1}i_{2}\dots i_{n}}$
- volume tensor
- pseudotensosr
- Tensor Fields
  - assigns a tensor to each point of a mathematical space (typically a Euclidean space or manifold).
  - generalizes scalar fields or vector fields
- gradient operator
- curvilinear coordinates
- metric tensor
  - rank-2 tensor
  - a type of function
  - input is a pair of tangent vectors at a point of surface (or higher dimensional manifold)
    - $v$, $w$
  - produces a real number scalar $g(v, w)$
  - generalizes many properties of the dot product of vectors in Euclidean space
  - defines length and angle
  - A metric tensor is called positive-definite if it assigns a positive value $g(v, v) > 0$ to every nonzero vector $v$.
- Riemannian manifold
  - Also called a Riemannian space
  - A manifold equipped with a positive-definite metric tensor.
- connection
  - https://en.wikipedia.org/wiki/Connection_(mathematics)
- affine connection
  - a geometric object that connects nearby tangent spaces
  - it's also to define how to differentiate scalars, vectors, even tensors
  - defines a covariant derivative
    - a way of specifying a derivative along tangent vectors of a manifold
    - a generalization of the directional derivative
    - (when it comes to an extrinsic view the covariant derivative is)
      - just the ordinary derivative with the normal component substracted
        - in flat space, it's just the ordinary derivative
  - required for defining directional derivative without fixing a point
  - can be specified by defining Christoffel symbol
  - It's a pretty wide definition, so there exist infinitely many affine connections
  - Christoffel symbols specify a corresponding affine connection
    - In Euclidean space Christoffel symbols are all zero
  - So what are reasonable connections? How can we define them?
    - 👉 Levi-Civita connection
  - defining affine connection is equivalent to
    - defining a Christoffel symbol
    - defining geodesic
    - defining how to do parallel transport
    - defining how to differentiate
  - an affine connection is not a tensor
- Levi-Civita connection $\nabla^{LC}$
  - a kind of affine connection given a metric "tensor"
  - satisfies
    - linearity with respect to the Christoffel symbol "functions" that depend on $P$ since it's a tensor
    - torsion free
      - $\Gamma^k_{ij} = \Gamma^k_{ji}$
      - two parallel transport paths makes a parallelogram when they are done with switched orders
    - metric compatibility
      - ${\partial}_k g_{ij} = \Gamma^l_{ik} g_{jl} + \Gamma^l_{jk} g_{il}$
      - meaning it preserves the metric
      - Leibniz rule
  - $\Gamma^m_{jk} = {1 \over 2} g^{im} ({\partial}_k g_{ij} + {\partial}_j g_{ki} + {\partial}_i g_{jk})$
    - note that $g^{im}$ is an element of the inverse metric tensor.
    - If you have $\Gamma_{12}^{3}$, it means you are looking at
      - how the third component of a vector changes
      - with respect to ...
      - TODO
  - Fundamental theorem of Riemannian geometry
    - For a Riemannian manifold (curved space with a metric), there is a unique connection (=covariant derivative) that is torsion-free and has metric compatibility. And this connection is called the Levi-Civita connection.
- Christoffel symbol
  - an array of numbers describing an affine connection
  - sometimes called the (affine/Levi-Civita) connection coefficients
  - ${\frac {\partial \mathbf {e} _{i}}{\partial x^{j}}}={\Gamma ^{k}}_{ij}\mathbf {e} _{k}=\Gamma _{kij}\mathbf {e} ^{k}$
    -  When the point moves along with the direction of $x^j$, how the basis vector $e_i$ changes in terms of all the current basis.
   - it's not a tensor so it depends on how we define the coordinates (or charts)
- parallel transport (along a curve)
  - a way of transporting geometrical data along smooth curves in a manifold
  - keeps vectors as constant as possible
    - but note that it could be impossible to keep vectors constant on a surface like a sphere
    - in another word, it's impossible to define a constant vector field on a curved surface
  - keeps the length of a vector constant
- differentiation in tensor field
- second covariant derivative
- geodesic
  - a curve representing the shortest path in some sense
    - between two points in a surface

  - a generalization of the notion of a "straight line" 

  - geodesic equation
    - ${\frac {d^{2}x^{\lambda }}{dt^{2}}}+\Gamma _{\mu \nu }^{\lambda }{\frac {dx^{\mu }}{dt}}{\frac {dx^{\nu }}{dt}}=0$
- Lie bracket
  - https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields
  - a vector field can be seen as a derivative operator
  - given two vector fields Lie Bracket generates another vector field indicating whether the tow vector fields as operators can be commutative
- torsion tensor
  - $T(X, Y) := \nabla_X Y - \nabla_Y X - [X, Y]$
  - where $X$ and $Y$ are vector fields
- torsion tensor free
  - $T(X,Y) = 0$
- Riemann curvature tensor
  - $R^d_{abc}$
  - (1 contra, 3 co) - Tensor
  - $a$, $b$
    - differentiation directions on the manifold
  - $c$
    - direction of the vector being transported
  - $d$
    - output component index
- Ricci Tensor
  - The Ricci curvature tensor is the contraction of the first and third indices of the Riemann tensor.
    - ${\displaystyle \underbrace {R_{ab}} _{\text{Ricci}}\equiv \underbrace {R^{c}{}_{acb}} _{\text{Riemann}}=g^{cd}\underbrace {R_{cadb}} _{\text{Riemann}}}$

  - keeps track of how volume change along geodesics.
  - approaches
    - sectional curvature
      - orthonormal basis only
    - volume element derivative
      - any basis
  - properties
    - contractions of Riemann tensor
    - symmetric
    - second Bianchi identity
    - contracted Bianchi identity
    - Einstein's field equations
- Ricci Scalar
  - keeps track of how the size of a ball deviates from standard flat-space size.
  - In a curved space, we can fit a large area in a small boundary.
- holonomy
- geodesic deviation
- isometry
- first fundamental form
  - intrinsic
  - consist of inner product values
    - ${\displaystyle \mathrm {I} (x,y)=x^{\mathsf {T}}{\begin{bmatrix}E&F\\F&G\end{bmatrix}}y}$
    - ${\displaystyle \left(g_{ij}\right)={\begin{pmatrix}g_{11}&g_{12}\\g_{21}&g_{22}\end{pmatrix}}={\begin{pmatrix}E&F\\F&G\end{pmatrix}}}$
    - ${\displaystyle g_{ij}=\langle X_{i},X_{j}\rangle }$
  - positive definite
    - $g_\text{ii} \gt 0$
    - $\det(g_\text{ij}) \gt 0$
  - the inner product on the tangent space of a surface
  - for the two dimensional surfaces defined in $\mathbb{R}^3$ ambient space.
    - so it's not applicable to the general relativity in physics which requires four dimensional space
  - induced canonically from the dot product of $\mathbb{R}^3$
- second fundamental form
  - extrinsic
- Gauss' theorema egregium (Gauss' remarkable theorem)
- Weingarten map or shape operator

## References

(main)

- [Tensor Calculus by eigenchris](https://youtube.com/playlist?list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)


(extra)

- https://en.wikipedia.org/wiki/Metric_tensor
- https://en.wikipedia.org/wiki/Affine_connection
- https://en.wikipedia.org/wiki/Tensor_product
- https://en.wikipedia.org/wiki/Leibniz_integral_rule
- https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields


- https://www.mathematik.hu-berlin.de/~wendl/pub/connections_chapter6.pdf
- [A Gentle Introduction to Tensors](https://www.ese.wustl.edu/~nehorai/Porat_A_Gentle_Introduction_to_Tensors_2014.pdf)
- https://einsteinrelativelyeasy.com/


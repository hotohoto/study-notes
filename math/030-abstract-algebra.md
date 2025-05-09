## algebraic structure

- set
- magma
  - set * single binary operation
  - the binary operation must be closed under *
- semigroup
  - magma with associativity
  - $(x * y) * z = x * (y * z)$
- monoid
  - semigroup with identity element
  - $x * e = e * x = x$
- group
  - (G, *)
  - closed under *
  - inverses
  - there exists an identity e
  - invertible monoid
  - $x * y = y * x = e$ ??
  - associative
  - may not be commutative
  - group multiplication tables
    - or Caley tables
  - https://en.wikipedia.org/wiki/Group_homomorphism
    - (G, +), (H, *)
    - $h: G \to H$
    - $h(u+v) = h(u)*h(v)$
    - $h(e_G) = e_H$
    - $h(u^{-1}) = h(u)^{-1}$
- subgroup
  - https://en.wikipedia.org/wiki/Subgroup
- symmetric group
  - https://en.wikipedia.org/wiki/Symmetric_group
  - n-th symmetric group
  - $S_n = \{f: S \to S | f \text{ is a bijective function}\}$
- abelian group
  - commutative group

- ring
  - 2 binary operations
  - abelian group regarding addition
  - semigroup regarding multiplication
  - supports distributivity
  - $x * (y + z) = z * y + x * z$
  - integer belongs to ring
- commutative ring
  - 곱셈도 교환법칙 성립하는 환
- division Ring
  - 0 이 아닌 모든 원소가 역원을 가지며 원소의 개수가 둘 이상인 환
  - (ex) 사원수(quaternion)
- field
  - 가환환인 나눗셈 환
  - (ex) Rational number, Real number, Complex number
  - can define real numbers axiomatically from the definition of field
- module(가군)
  - ring(환)의 작용이 scalar 곱으로 주어진 abelian group(아벨군)
- vector space
  - also called a linear space
  - 체(field) 의 작용이 scalar 곱으로 주어지는 가군(module)
  - linear space
- metric space
  - vector space with metric (distance)
- normed space
  - metric space with a norm (size)
- banach space
  - normed space which is Cauchy complete
    - Cauchy compete: every Cauchy sequence of points in M has a limit that is also in M
    - Can compute vector length
    - Can compute distance between vectors
- inner product space
  - metric space with an inner-product (similarity)
  - pre-hilbert space
- Hilbert space
  - inner product space which is also banach space
  - inner product is defined.
  - cauchy complete
- Reproducing Kernel Hilbert Space
  - Hilbert space of functions in which point evaluation is a continuous linear functional.
- dual vector space
  - vector space V* of linear maps from a vector space V to its scalar field F
  - the linear maps are called "linear functional" or just "functional"
- $L^p$ space
  - function space
  - p-norm can be computed in finite/infinite dimensions
  - $L^2$ space is the only hilbert space among $L^p$ spaces

## function

- A function is also defined as a set
  - $f = S_f \sub A x B = \{ (a, b) | a \in A, b \in B\}$
  - $\forall a_0 \in A, \exist b_0 \in B$, and $(a_0, b_0) \in S_f$
  - If $(a_1, b_1) \in S_f$ and $(a_1, b_2) \in S_f$, then $b_1 = b_2$
  - and we denote such a fuction f as $f: A \to B$
- https://en.wikipedia.org/wiki/Group_homomorphism

## Notes

- Strictly speaking, substraction and division are not operators.
- 0 is defined as an identity element of the addition operator
- negative integers are defined as inverse elements of the addition operator
- representation theory is related to linear algebra
- Linear algebra assume the flat Euclidean space
- Derivative brings structure rely on non euclidean space into Euclidean space and integral is it's inverse function.
- For example, in linear algebra, if the inner product operator defines the angle in the Euclidean space, where as in statistics fisher information or KL divergence defines the equivalent angle but in non Euclidean space (Refer to information geometry)
- exponential function is defined as a function between 2 groups and their operator to preserve the operators
  - $f(x+y) = f(x) \times f(y)$
- matrix is a function mapping from a vector space to a/another vector space
- axiom of choice
  - when we define a inverse function we need axiom of choice.
- https://en.wikipedia.org/wiki/Galois_theory

## References

- https://en.wikipedia.org/wiki/Algebraic_structure
- https://math.stackexchange.com/a/2361890/634023
- https://en.wikipedia.org/wiki/Semigroup
- https://en.wikipedia.org/wiki/Space_(mathematics)

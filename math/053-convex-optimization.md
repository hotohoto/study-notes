# Convex Optimization

https://convex-optimization-for-all.github.io/

## 01 Introduction
### 01-01 Optimization problems

- $x \in R^n$
  - optimization variable
- $f: R^n \to R$
  - objective function
  - or cost function
- Explicit constraints
  - $g_i: R^n \to R$
    - inequality constraint functions
  - $h_i: R^n \to R$
    - equality constraint functions
- Implicit constraints
  - intersection of domains of all functions above

### 01-02 Convex optimization problems

(Convex optimization problem)
- $f$ and $g_i$ are convex functions
- $h$ are affine functions

(Convex sets)

(Convext functions)

- $f: \mathbb{R}^n \to \mathbb{R}$
- $\text{dom}(f)$ is a convex set
- $f(\theta x + (1 - \theta)y) \le \theta f(x) + (1 - \theta) f(y)$
  - $\forall x, y \in \text{dom}(f)$
  - $0 \le \theta \le 1$

(Relation between a convex set and a convex function)

- epigraph of $f$
  - $f: \mathbb{R}^n \to \mathbb{R}$
  - $\text{epi} f = \{(x, t) \in \mathbb{R}^{n+1}| x \in \text{dom} f, \ f(x) \lt t\}$
- $f$ is a convex function $\iff$ $\text{epi} f$ is a convex set

(Nice property of convex optimization problems)

- a local minimum of a convex function is the global minimum


## 02 Convex Sets

https://convex-optimization-for-all.github.io/contents/chapter02/

### 02-01-01

- line
- line segment
- ray

#### 02-01-02 Affine Set

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/08/02_01_02_Affine_set/

- affine set
  - consists of infinitely many lines
  - there is no boundary
- affine combination
  - weights sum to 1
- affine hull of C
  - minimum affine set that includes C
- affine set and subspace

#### 02-01-03 Convex set

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/08/02_01_03_Convex-set/

- convex set
  - consists of infinitely many segments
- convex combination
  - weights are non-negative and they sum to 1
- convex hull
  - minimum convex set that includes C

#### 02-01-04 Cone

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/08/02_01_04_Convex_cone/

- cone
  - consists of rays start from the origin
- convex cone
  - convex and cone
  - nonnegative homogeneous set
  - consists of infinitely many rays start from the origin
- conic combination
  - weights are all positive
- conic hull
  - minimum convex cone that includes C

### 02-02 Some important examples

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/11/02_02_Some_important_examples/

- trivial ones
  - empty set
  - point
  - line
  - line segment
  - ray

- convex cone
  - norm cone
  - normal cone
  - positive semidefinite cone

#### 02-02-01 Convex set examples

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/11/02_02_01_Convex_sets_examples/

(hyperplanes)

- $\{x\mid a^\top x = b\}$
  - $a \in \mathbb{R}^n$
  - $a \neq 0$
  - $b \in \mathbb{R}$
- also an affine set

(halfspaces)

- $\left\{x\mid\ a^\top x \ge b\right\}$ or $\left\{x\mid\ a^\top x \le b\right\}$
  - $a \in \mathbb{R}^n$
  - $a \neq 0$
  - $b \in \mathbb{R}$

(open halfspaces)

- $\left\{x \mid a^\top x \gt b\right\}$ or $\left\{x\mid a^\top x \lt b\right\}$

(euclidean balls)

- $B(x_c, r) = \left\{x\mid ||x_c - x||_2 \le r \right\} = \left\{x_c + ru\mid ||u||_2 \le 1 \right\}$

(ellipsoids)

- $\mathcal{E}=\left\{x \mid\left(x-x_{c}\right)^\top P^{-1}\left(x-x_{c}\right) \leq 1\right\}$
  - $P$
    - positive definite
    - symmetric

or

- $\mathcal{E}=\left\{x_{c}+A u \mid\|u\|_{2} \leq 1\right\}$
  - $A$
    - non singular squared matrix

- if A or P is positive "semi" definite, it becomes a degenerate ellipsoid.
  - it's still a convex set.

(p-norm balls)

- $\left\{x \mid\|x - x_c||_p \leq r\right\}$
  - is a convex set if $p \ge 1$

(polyhedra)

- $\mathcal{P}=\left\{x \mid a_{i}^\top x \leq b_{i}, i=1, \ldots, m, c_{j}^\top x=d_{j}, j=1, \ldots, p\right\}$
- or $\mathcal{P}=\left\{x \mid a_{i}^\top x \leq b_{i}, i=1, \ldots, m\right\}$
- any affine set (subspaces, hyperplanes, lines) / ray / line segment / halfspace is a polyhedron

(simplexes)

- $C=\operatorname{conv}\left\{v_{0}, \cdots, v_{k}\right\}=\left\{\theta_{0} v_{0}+\cdots+\theta_{k} v_{k} \mid \theta \succeq 0,1^{T} \theta=1\right\}$

- probability simplexes:
  - $C=\operatorname{conv}\left\{e_{1}, \cdots, e_{n}\right\}=\left\{\theta \mid \theta \succeq 0,1^{T} \theta=1\right\}$

#### 02-02-02 Convex cone examples

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/11/02_02_02_Convex_cone_examples/

(Norm cone)

- $C = \{(x, t): \ ||x|| \le t\} \subseteq \mathbb{R}^{n+1}$
  - for a norm $||\cdot||$
- icecream cone = second order norm cone

(Normal cone)

- https://vimeo.com/452229674
- definition
  - $N_C(x) =$
    - if $x \in \text{bd}(C)$
      - $\{g | g^T (y - x) \le 0, \ \forall y \in C \}$
    - else
      - $\emptyset$
- supporting hyperplanes
  - touch set $C$ at a boundary $x$ and have full set $C$ on one side
  - it's said to support $C$ at the boundary $x$
  - definition
    - Hyperplane $\{y: g^T y = r\}$ supports $C$ at $x \in \text{bd}(C)$ if
      - $g^T x = r$
      - $g^T y \le r, \ \forall y \in C$
    - $g$ is called normal vector to $C$ at $x$
- supporting halfspace
  - halfspace made by a supporting hyperplane and containing the set C

(Positive semidefinite cone)

- definition
  - $\mathbb{S}^n_+ = \{ X \in \mathbb{S}^n: X \succeq 0\}$ where
    - $\mathbb{S}^n$: the set of $n \times n$ symmetric matrices
    - $X$: positive semidefinite matrix
- $\theta_1, \theta_2 \gt 0 \ \wedge \ A, B \in \mathbb{S}^n_+ \Longrightarrow \theta_1 A + \theta_2 B \in \mathbb{S}^n_+$
  - so it's a convex cone
- Note that
  - when $X \in \mathbb{S}^2$
    - $X \in \mathbb{S}^2_+ \iff \operatorname{tr}(X) \ge 0 \ \wedge \ \det(X) \ge 0$
  - https://math.stackexchange.com/a/2894056/634023


#### 02-02-02 Operations that preserve convexity

https://convex-optimization-for-all.github.io/contents/chapter02/2021/02/11/02_03_Operations_that_preserve_convexity/


TODO

# Topology

following James R. Munkres

## General Topology
### 2. Topological Space and Continuous Functions

(1)

- open interval $(a, b)$ on $\mathbb{R}$
- open ball
- closed ball
- Closure
  - The closure of a subset S of points in a topological space consists of all points in S together with all limit points of S.

(2)

- neighborhoods
  - Let's say $N \subset \mathbb{R},\ L \in \mathbb{R}$,
  - if $\exists (a, b) \subset N \ s.t.\ L \in (a, b)$
  - then $N$ is neighborhood of $L$
  - I guess this just tells us there are elements next to the point L.
  - this is equivalent to an open interval constrained with a point L is included

- open set
  - Union of open intervals
    - 0 times:
      - empty set is an open set
    - 1 times:
      - equivalent to a single open interval
    - n times
- closed set
  - complement of an open set

(3 family)

- Topology
  - family of sets defining the structure of the topological space
  - geometric object that are preserved under continuous deformations
  - properties(?)
    - dimension
      - allows distinguishing between a line and a surface
    - compactedness
      - allows distinguishing between a line and a circle
    - connectedness
      - allows distinguishing a circle from two non-intersecting circles
- Base
  - A base or basis for the topology τ of a topological space (X, τ) is a family B of open subsets of X such that every open set of the topology is equal to a union of some sub-family of B (this sub-family is allowed to be infinite, finite, or even empty).
  - A base generate a topology so that every open set is a union of basis elements.
  - for example
    - $\mathcal{B}$ is a basis for the Euclidean topology on $\mathbb{R}$.
    - where $\mathcal{B} = \{(a,b) |\ a, b \in \mathbb{R},\ a \lt b\}$
    - and $(a, b)$ represents for an open interval
- Subbase
  - A subbase (or subbasis) for a topological space X with topology T is a subcollection B of T that generates T, in the sense that T is the smallest topology containing B.
  - A subbase generate a topology so that every open set is a union of finite intersections of subbasis elements.
- Standard topology
- Lower limit topology
  - L
- Upper limit topology
  - U
- indiscrete topology
  - $\{X, \phi\}$
- discrete topology
  - powerset of X
- confinite topology
- on $\mathbb{R}$
  - indiscrete topology ⊂ confinite topology ⊂ standard topology ⊂ lower/upper limit topology ⊂ discrete topology
  - both lower and upper limit topologies are not a subset of each other

(4)

- Topological space
  - geometrical space in which closeness is defined
  - set of points along with set of neighborhoods for each point
  - allows limits, continuity, connectedness
  - Definition via neighbourhoods
    - TBD
  - Definition via open sets
    - $(X, \mathcal{T})$
      - $\phi \in \mathcal{T}$
      - $X \in \mathcal{T}$
      - $\forall U_i \in \mathcal{T},\ \bigcap_{i=1}^n U_i \in \mathcal{T} \ (n < \infty)$
      - $\forall U_i \in \mathcal{T},\ \bigcup_i U_i \in \mathcal{T}$
  - $X$ is a set of points
  - $\mathcal{T}$ is a collection of subsets of $X$
  - for example,
    - $X = \{1, 2\}$
    - These are all topological spaces.
      - $(X, \{\phi, X\})$
      - $(X, \{\phi, X, \{1\}\})$
        - Note that this topology is not a σ-algebra, since σ-algebra should include the complement of any element.
      - $(X, \{\phi, X, \{2\}\})$
      - $(X, \{\phi, X, \{1\}, \{2\}\}) = P(X)$
  - $(X, P(X))$ and $(X, {\phi, X})$ are always topological spaces

- Metrizable spaces
  - topological spaces that are homeomorphic to a metric space
- Metric spaces
- Euclidean spaces
### 3. Connectedness and Compactness

### 4. Countability and Separation Axioms

- $T_0$ or Kormogorov spaces
  - 임의의 다른 x, y 에 대해 x또는 y를 둘중 하나만 포함한 openset 이 존재
- $T_1$
  - 임의의 다른 x, y 에 대해 x와 y를 둘중 하나만 포함한 openset 이 각각 존재
- $T_2$ or Hausdorff spaces
  - 임의의 다른 x, y에 대해 x를 포함하는 openset U 와 y를 포함하는 openset V가 서로소이도록 하는 U, V가 존재.
- $T_3$ or regular topological spaces
  - 임의의 closeset C와 C에 포함되지 않은 x에 대해서 C를 포함하는 openset U와 x를 포함하는 openset V가 서로소이도록 하는 U, V가 존재
- $T_4$ or normal (Hausdorff) spaces
  - 임의의 closeset C와 D에 대해서 C를 포함하는 openset U와 D를 포함하는 openset V가 서로소이도록 하는 U, V가 존재
- $T_5$ or completely normal Hausdorff spaces
- $T_6$ or perfectly normal Hausdorff spaces



### 5. The Tychonoff Theorem

### 6. Metrization Theorems and Paracompactness

### 7. Complete Metric Spaces and Function Spaces

### 8. Baire Spaces and Dimension Theory

## Algebraic Topology

### 9. Fundamental Group

### 10. Separation theorem in the plane

### 11. The Seifert-van Kampen Theorem

### 12. Classification of Surfaces

### 13. Classification of Covering Spaces

### 14. Application to Group theory
## etc

- accumulation point = limit point
- derived set
  - A'
- closure
  - $A \cup A' = \bar{A}$
## References

- https://en.wikipedia.org/wiki/Base_(topology)
- https://en.wikipedia.org/wiki/Subbase
- [수학의즐거움 위상수학](https://youtube.com/playlist?list=PL4m4z_pFWq2o4vYP5vqpwqhftOtbzF4lh)

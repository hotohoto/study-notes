# set theory

- [이상엽Math 집합론](https://www.youtube.com/playlist?list=PL127T2Zu76FveA8TGXZU-PSSt7GTMhKp6)
- [수학의즐거움 집합론 기초](https://youtube.com/playlist?list=PL4m4z_pFWq2rboSAR7cvRLcCI36Fb8ruF)

## TODO

- 수학의 즐거움 2번 비디오 보기

## Propositional Calculus and logics

- proposition (명제)

(references)

- https://en.wikipedia.org/wiki/Propositional_calculus

## sets

- subset
- superset
- proper subset = strict subset
- proper superset = strict superset
- family
  - its elements are sets
  - ex) power set
- indexed family
- cartesian product

## relations

- equivalent relation
  - 1 to 1 mapping to partition
  - relation
  - by definition
    - reflexivity
      - https://en.wikipedia.org/wiki/Reflexive_relation
    - symmetry
      - https://en.wikipedia.org/wiki/Symmetric_relation
    - transitivity
      - https://en.wikipedia.org/wiki/Transitive_relation
- equivalent class
  - 동치류
  - set
- quotient set
  - 상집합
  - family of sets
- partition of X
  - not empty
  - disjoint subsets of X
  - covers X
  - family of sets

## functions

- image
- preimage
- domain
- codomain
- range
- mapping types
  - https://en.wikipedia.org/wiki/Bijection
  - injective function
    -  (단사함수)
    -  one-to-one
    -  $f: A \to B$ is an injective function if and only if there exists $g: B \to A$ such that $g \circ f$ is an identity function $i _A$ on $A$
  - surjective function (전사함수)
    - range = codomain
    - onto
    - $f: A \to B$ is an surjective function if there exists $g: B \to A$ such that $f\circ g$ is an identity function $i _B$ on $B$
      - for the reverse
        - since $\forall b \in B$ we need to choose a unique $a \in A$ such that $f(a) = b$. But this is not trivial (especially when $A$ is an infinite set) since $f$ can be non injective.
        - it holds if we assume the axiom of choice (even for the infinite set).
  - bijective function (전단사함수)
    - injective and surjective
  - identity function (항등합수)
  - constant function (상수함수)
  - inverse function (역함수)
    - Any inverse function can exist only for bijective functions
  - composite function (합성함수)
  - inclusion function
  - characteristic function or indicator function
    - https://en.wikipedia.org/wiki/Indicator_function
  - choice function
    - mapping to an element of the given set
- if the domains are different their functions are different as well

### Prepositions

$f(A \cup B) = f(A) \cup f(B)$

$f(A \cap B) \subseteq f(A) \cap f(B)$

$f(f^{-1}(B_1)) \subseteq B_1$

- $B_1 \subseteq B$
- due to codomain which is not range

$f^{-1}(f(A_1)) \supseteq A_1$

- $A_1 \subseteq A$
- due to elements $a_2 \in A$ s.t. $a2 \notin A_1$ and $f(a_2) \in f(A_1)$

## size of set

- `equipotent`
  - if a bijective function exists
- `uncountable`
  - $\#A > \#\mathbb{N}$
  - $\#\mathbb{R} = \#\mathbb{C} = \varsigma$
- `countable`
  - if an injective function to a subset of $\mathbb{N}$ exists
  - categories
    - `denumerable`
      - `countably infinite`
      - $A \approx \mathbb{N}$
      - $\#\mathbb{N} = \#\mathbb{Z} = \#\mathbb{Q} = ℵ_0$
      - there exists an injective function to $\mathbb{N}$
    - `finite`
      - includes empty set $\emptyset$
      - $\#A = k$
        - e.g. $\#\{a,b,c\} = 3$
  - countable is the smallest infinite
- cardinal number formulas
  - $ℵ_0 + ℵ_0 = ℵ_0$
  - $\varsigma + \varsigma = \varsigma$
  - $ℵ_0 + \varsigma = \varsigma$
  - $ℵ_0 ℵ_0 = ℵ_0$
  - $\varsigma \varsigma = \varsigma$
  - $ℵ_0 \varsigma = \varsigma$
  - $\varsigma = ℵ_0 ^ {ℵ_0} = \varsigma^{ℵ_0}$
  - $2^\varsigma = ℵ_0 ^ \varsigma = \varsigma ^ \varsigma$
- Cantor–Bernstein theorem
  - https://en.wikipedia.org/wiki/Cantor%E2%80%93Bernstein_theorem

## Continuum hypothesis

- the first of [Hilbert's 23 problems](https://en.wikipedia.org/wiki/Hilbert%27s_problems)
  - advanced by Cantor
  - $ℵ_0 < ? < \varsigma$

## TODO

- https://youtu.be/0PJ4NJ-PGP0?t=290

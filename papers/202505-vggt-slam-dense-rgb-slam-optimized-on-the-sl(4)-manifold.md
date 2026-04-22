# VGGT-SLAM - Dense RGB SLAM Optimized on the SL(4) Manifold

https://arxiv.org/abs/2505.12549

## Glossary and concepts

### Lie group

- differentiable manifold
    - group multiplication is differentiable
    - taking inverses is differentiable
- group
    - group multiplication is defined
    - satisfies
        - closure
        - identity element exists
        - inverse element exists
        - associativity
$$
SE(n) \subset Sim(n) \subset Aff(n) \subset PGL(n+1)
$$

#### GL(n)

- general linear group
- $\text{det} \neq 0$
    - invertible
- $n \times n$ matrix
- $n^2$ DOF

#### PGL(n)

- = $GL(n) / \{\alpha I \vert \alpha \neq 0\}$
- $A \sim B \Longleftrightarrow B = \alpha A$
    - $(\alpha \neq 0)$

#### SL(n)

- GL(n) and $\text{det} = 1$

---

#### SE(3)

$$
T =
\begin{bmatrix}
R & t \\
0 & 1 
\end{bmatrix}
$$
- $\mathbf{x}^\prime = sR\mathbf{x} + t$
- 6 DOF

#### SO(3)

- $R$
- $\mathbf{x}^\prime = R\mathbf{x}$
- 3 DOF

#### Sim(3)

$$
T =
\begin{bmatrix}
sR & t \\
0 & 1 
\end{bmatrix}
$$
- $\mathbf{x}^\prime = sR\mathbf{x} + t$
- 7 DOF

#### Aff(3)

$$
T =
\begin{bmatrix}
M & t \\
0 & 1 
\end{bmatrix}
$$
- $\text{det}(M) \neq 0$
- $\mathbf{x}^\prime = M\mathbf{x} + t$
- 12 DOF

#### Projective(3)

- := PGL(4)
- an element is called homography

### Etc.

- Submap
    - mapping from a set of adjacent frames
- camera pose DoF
    - extrinsic: 6
    - intrinsic: 4

## 1 Introduction

## 2 Related work

## 3 Method

## 4 Experiments

## 5 Discussion

## References

## A Proof of formula 1

## B Proof of formula 2

## Notes

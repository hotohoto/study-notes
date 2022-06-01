# Linear algebra with applications

- Steven J. Leon
- sixth edition

## 1. Matrices and systems of equations

### 1.2 Row Echelon Form

### 1.4 Elementary Metrices

- row equivalent

### 1.5 Partioned Matrices

## 2. Determinants

### 2.1 The determinant of a matrix

### 2.2 Properties of determinants

### 2.3 Cramer's Rule

## 3. Vector spaces

### 3.1 Definitions and examples

### 3.2 Subspaces

- null space

### 3.3 Linear Independence

### 3.4 Basis and dimension

### 3.5 Change of basis

### 3.6 Row space and column space

- rank
  - dimension of row space
- nullity
  - dimension of null space
- the rank nullity theorem

## 4. Linear transformations

### 4.1 Definition and examples

- $(V, +, ·)$, $(W, +, ·)$
- linear map
  - $L: V \rightarrow W$
    - map from a vector space to a vector space
  - additivity
    - $L(u + v) = L(u) + L(v)$
  - homogeneity
    - $L(kv) = kL(v)$
    - $k \in F$
  - kernel
    - $\operatorname{ker}(L) = L^{-1}(0) = \{v \in V | L(v) = 0\}$
  - image
    - $\operatorname{img}(L) = L(V) = \{L(v) \in W | v \in V\}$
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

### 4.2 Matrix representations of linear transformations

### 4.3 Similarity

$$
B = S^{-1}AS
$$

- We can see it $A$ and $B$ represent the same linear transform $L$
  - where
    - $A$ is with respect to the basis $E$
    - $B$ is with respect to the basis $F$
    - $S$ tranlates a vector represented by the basis $F$ into the same vector represented by the basis $E$
      - columns of S can be seen as $F$ represented by the basis of $E$

https://en.wikipedia.org/wiki/Matrix_similarity

properties of similar matrices
- same trace
- same determinant
- same rank
- same eigenvalues

## 5. Orthogonality

### 5.1 The scalar product in $R^n$
### 5.2 Orthogonal subspaces

- orthogonal subspaces
- orthogonal complement
- fundamental subspaces theorem
  - $N(A) = R(A^\top)$
  - $R(B^\top)^\perp = R(A)^\perp$

### 5.3 Least squares problems
### 5.4 Inner product spaces
### 5.5 Orthonormal sets
### 5.6 The Gram-Schmidt orthogonalization process
### 5.7 Orthogonal polynomials

#### Orthogonal sequences

Inner product may be defined as

$$\langle p, q \rangle = \int_a^b p(x) q(x) w(x) dx$$

where $w(x)$ is a postive continuous function. The interval can be taken either open or closed and may be finite or infinite. And $\int_a^b p(x) w(x) dx$ converges for every $p \in P$.

(Theorem 5.7.2)

Let $p_0, p_1, ...$ be a sequnce of orthogonal polynomials. Let $a_i$ denote the lead coefficient of $p_i$ for each i, and define p_{-1}(x) to be the zero pynomial. Then


$$
\alpha_{n+1} p_{n+1} (x) = (x - \beta_{n+1}) p_n (x) - \alpha_n \gamma_n p_{n-1} (x) \qquad (n \ge 0)
$$

where

- if $n = 0$
  - $\alpha_0 = \gamma_0 = 1$
- if $n > 0$

$$\alpha_n = {a_{n-1} \over a_n}$$
$$\beta_n = {(p_{n-1}, x p_{n-1}) \over (p_{n-1}, p_{n-1})}$$
$$\gamma_n = {(p_n, p_n) \over (p_{n-1}, p_{n-1})}$$

if we choose $a_i$ to be $1$,

$$
p_{n+1} (x) = (x - \beta_{n+1}) p_n (x) - \gamma_n p_{n-1} (x)
$$

#### Classical orthogonal polynomials

##### Legendre polynomials

inner product:

$$\langle p, q \rangle = \int_{-1}^1 p(x) q(x) dx$$

recursion relation:

$$(n+1) P_{n+1} = (2n + 1) x P_n(x) - nP_{n-1}(x)$$

the first five polynomials of the sequence:

- $p_0(x) = 1$
- $p_1(x) = x$
- $p_2(x) = {1 \over 2}(3x^2 - 1)$
- $p_3(x) = {1 \over 2}(5x^3 - 3x)$
- $p_4(x) = {1 \over 8}(35x^4 -30x^2 + 3)$

##### Chebyshev polynomials $T_n(x)$

inner product:

$$\langle p, q \rangle = \int_{-1}^1 p(x) q(x) (1-x^2)^{-1/2}dx$$

properties:

$$T_n(\cos \theta) = \cos n\theta$$
$$\cos(n+1) \theta = 2 \cos \theta \cos n\theta - \cos(n-1)\theta$$

recursion relation:

$$T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x) \qquad (n \ge 1)$$

the first four polynomials of the sequence:

- $T_0(x) = 1$
- $T_1(x) = x$
- $T_2(x) = 2x^2 - 1$
- $T_3(x) = 4x^3 - 3x$


##### Jacobi polynomials

(generalization of Legendre and Chebyshev polynomials)

$$\langle p, q \rangle = \int_{-1}^1 p(x) q(x) (1-x)^{\lambda}(1+x)^{\mu}dx \qquad (\lambda, \mu \gt -1)$$

##### Hermite polynomials $H_n(x)$

inner product:

$$\langle p, q \rangle = \int_{-\infty}^{\infty} p(x) q(x) e^{-x^2}dx$$

recursion relation:

$$H_{n+1}(x) = 2xH_n(x) - 2n H_{n-1}(x)$$

the first four polynomials of the sequence:

- $H_0(x) = 1$
- $H_1(x) = 2x$
- $H_2(x) = 4x^2 - 2$
- $H_3(x) = 8x^3 - 12x$

##### Laguerre polynomials $L_n(x)$

inner product:

$$\langle p, q \rangle = \int_0^{\infty} p(x) q(x) x^{\lambda}e^{-x}dx$$

recursion relation:

$$(n+1) L_{n+1}^{\lambda}(x) = (2n + 1 - x)L_n(x) - 2n L_{n-1}(x)$$

the first four polynomials of the sequence :

- $L_0^{(0)}(x) = 1$
- $L_1^{(0)}(x) = 1 - x$
- $L_2^{(0)}(x) = {1 \over 2}x^2 - x + 2$
- $L_3^{(0)}(x) = {1 \over 6}x^3 + 9 x^2 - 18x + 6$

## 6. Eigenvalues

### 6.1 Eigenvalues and eigenvectors

characteristics polynomial
$$P_A(\lambda) = det(A - \lambda I)$$

#### Complex Eigenvalues

#### The product and sum of the eigenvalues

$$
\prod \lambda_i = det(A)
$$

$$
\sum \lambda_i = \operatorname{tr}(A)
$$

#### Similar matrices

They have the same eigenvalues.

### 6.2 System of linear differenal equations

#### Complex eigenvalues
#### Highorder systems

### 6.3 Diagonalization

(Theorem 6.3.1)

- If $\lambda_1, \lambda_2, ..., \lambda_k$ are distinct eigenvalues of an $n \times n$ matrix A with corresponding eigenvectors $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_k$,
- then, $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_k$ are linearly independent

(Theorem 6.3.2)

- An $n \times n$ matrix $A$ is diagonalizable if and only if $A$ has $n$ linearly independent eigenvectors.

(diagonalizable matrix)

#### The exponential of a matrix

For any $n \times n$ matrix $A$, we can define the matrix exponential $e^A$ as follows.

$$
e^A = I + A + {1 \over 2!} A^2 + {1 \over 3!} A^3 + \cdots
$$

If $X$ diagonalizes $A$ then,

$$
D = X^{-1}AX
$$
$$
A = XDX^{-1}
$$

$$
e^A = e^{XDX^{-1}} = X(I + A + {1 \over 2!} A^2 + {1 \over 3!} A^3 + \cdots) X^{-1} = Xe^DX^{-1}
$$

#### personal notes: diagonalizable matrix on wikipedia

- https://en.wikipedia.org/wiki/Diagonalizable_matrix

- diagonalizable
  - unitarily diagonalizable
    - orthogonally diagonalizable
- defective over real and complex entries
  - non diagonalizable for those entries

- Schur form
  - $T = U^HAU$
    - $T$: upper triangular
- Jordan normal form or Jordan canonical form
  - $J = P^{−1}AP$
  - whether it's defective or not (over real and complex entries)
- diagonalizable
  - $D = P^{-1}AP$
- diagonalizable by a unitary matrix
  - $D = U^HAU$
- orthogonally diagonalizable
  - $D = P^{T}AP$

### 6.4 Hermitian matrices

#### Complex Inner Products

- $\langle\mathbf{z}, \mathbf{z}\rangle \ge 0$
- $\langle\mathbf{z}, \mathbf{w}\rangle = \overline{\langle\mathbf{w}, \mathbf{z}\rangle}$
- $\langle\alpha\mathbf{z} + \beta\mathbf{w}, \mathbf{u}\rangle = \alpha\langle \mathbf{z}, \mathbf{u}\rangle + \beta\langle \mathbf{w}, \mathbf{u}\rangle$

#### Hermitian matrices

(some rules)

- $(A^H)^H = A$
- $(\alpha A + \beta B)^H = \overline\alpha A^H + \overline\beta B^H$
- $(AC)^H = C^H A^H$

(definition)

A matrix $M$ is said to be **Hermitian** if $M = M^H$.

(Theorem 6.4.1)

The eigenvalues of a Hermitian matrix are all real. Futhermore, eigenvectors belonging to distinct eigenvalues are orthogonal.

(definition)

An $n \times n$ matrix $U$ is said to be **unitary** if its column vectors form an orthonormal set in $C^n$.

(Corollary 6.4.2)

If the igenvalues of a Hermitian matrix $A$ are distinct, then there exists a unitary matrix $U$ that diagonalizes $A$.

(Theorem 6.4.3 - Schur's Theorem)

For each $n \times n$ matrix $A$, there exists a unitary matri $U$ such that $U^H A U$ is upper triangular.

(Theorem 6.4.4 - Spectral Theorem)

If $A$ is Hermitian, then there exists a unitary matrix $U$ that diagonalizes $A$.

(Corollary 6.4.5)

If $A$ is a real symmetric matrix, then there is an orthogonal matrix $U$ that diagonalizes $A$, that is, $U^T A U = D$, where $D$ is diagonal.

#### Normal Matrices

(Definition)

A matrix $A$ is said to be **normal** if $AA^H = A^H A$.

(Theorem 6.4.6)

A matrix $A$ is normal if and only if $A$ possesses a complete orthonormal set of eigenvectors.

### 6.5 The singular value decomposition

### 6.6 Quadratic forms

### 6.7 Positive definite matrices

### 6.8 Nonnegative matrices

## 7. Numerical linear algebra

## 8. Iterative methods

## 9. Jordan canonical form



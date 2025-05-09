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

- basis vectors are linearly independent by definition

### 3.5 Change of basis

### 3.6 Row space and column space

- N(A)
  - null space of A
- R(A)
  - range of A which is actually the column space
- rank
  - dimension of row space
- nullity
  - dimension of null space
- the rank nullity theorem
  - rank(A) + nullity(A) = n

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

#### Application 1: information retrieval revisited

- represent a dataset of documents as a normalized bag of words matrix
- calculate cosine similarity between each document and the search text

#### Application 2: statistics - correlation and covariance matrices

- where
  - $X$ is $n \times p$ matrix centered to the mean of each feature
  - $U$ is $n \times p$ matrix normalized for each feature
- correlation matrix
  - $C = U^TU$
- covariance matrix
  - $S = {1\over n - 1} X^TX$

#### Application 3: psychology - factor analysis and principal component analysis

(Refer to 6.5 - application 4)

### 5.2 Orthogonal subspaces

- orthogonal subspaces
- orthogonal complement
- fundamental subspaces theorem
  - $N(A) = R(A^T)^\perp$ and $N(A^T) = R(A)^\perp$
  - whrer
    - $R(A)$ is the range of $A$ which is the column space of $A$

### 5.3 Least squares problems
### 5.4 Inner product spaces

- Frobenius norm
  - $||A||_F = (\langle A, A\rangle)^{1/2} = (\sum\limits_{i=1}^{m} \sum\limits_{j=1}^{n} a_{ij}^2) ^{1/2} = (\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_n^2) ^{1/2}$
    - where
      - $A \in R^{m \times n}$
      - $\sigma_i$ is a singular value
        - Refer to the lemma 6.5.2

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

(conjugate transpose or Hermitian transpose)

- $\mathbf{z}^H = \overline{\mathbf{z}}^T$

(complex inner products)

- $\langle\mathbf{z}, \mathbf{z}\rangle \ge 0$
- $\langle\mathbf{z}, \mathbf{w}\rangle = \overline{\langle\mathbf{w}, \mathbf{z}\rangle}$
- $\langle\alpha\mathbf{z} + \beta\mathbf{w}, \mathbf{u}\rangle = \alpha\langle \mathbf{z}, \mathbf{u}\rangle + \beta\langle \mathbf{w}, \mathbf{u}\rangle$

(standard inner product on $C^n$)

- $\langle \mathbf{z},\mathbf{w} \rangle = \mathbf{w}^H \mathbf{z}$
- $||\mathbf{z}|| = (\mathbf{z}^H\mathbf{z})^{1/2}$

(Hermitian matrices)

A matrix $M$ is said to be **Hermitian** if $M = M^H$.

(Some rules)

- $(A^H)^H = A$
- $(\alpha A + \beta B)^H = \overline\alpha A^H + \overline\beta B^H$
- $(AC)^H = C^H A^H$

(Theorem 6.4.1)

The eigenvalues of a Hermitian matrix are all real. Furthermore, eigenvectors belonging to distinct eigenvalues are orthogonal.

(definition)

An $n \times n$ matrix $U$ is said to be **unitary** if its column vectors form an orthonormal set in $C^n$.

- Note that $U^H = U^{-1}$.

(Corollary 6.4.2)

If the eigenvalues of a Hermitian matrix $A$ are distinct, then there exists a unitary matrix $U$ that diagonalizes $A$.

- Note that actually it hold even if the eigenvalues are not distinct. See 6.4.4 spectral theorem.

(Theorem 6.4.3 - Schur's Theorem)

For each $n \times n$ matrix $A$, there exists a unitary matri $U$ such that $U^H A U$ is upper triangular.

(Theorem 6.4.4 - Spectral Theorem)

If $A$ is Hermitian, then there exists a unitary matrix $U$ that diagonalizes $A$.

- Note that eigenvectors corresponding to the same eigenvalue need not be orthogonal to each other. However, since every subspace has an orthonormal basis, orthonormal bases can be found for each eigenspace, so an orthonormal basis of eigenvectors can be found.

(Corollary 6.4.5)

If $A$ is a real symmetric matrix, then there is an orthogonal matrix $U$ that diagonalizes $A$, that is, $U^T A U = D$, where $D$ is diagonal.

#### Normal Matrices

(Definition)

A matrix $A$ is said to be **normal** if $AA^H = A^H A$.

(Theorem 6.4.6)

A matrix $A$ is normal if and only if $A$ possesses a complete orthonormal set of eigenvectors.

### 6.5 The singular value decomposition

(Theorem 6.5.1 - The SVD Theorem)

If $A$ is an $m \times n$ matrix, then $A$ has a singular value decomposition.

$$A = U \Sigma V^T$$

(observations)

- The singular values $\sigma_1, ..., \sigma_n$ are unique.
  - Note that the matrices $U$ and $V$ are not unique
- $V$ diagonalizes $A^T A$
- $U$ diagnoalizes $AA^T$
- $v_j$'s are called right singular vectors
- $u_j$'s are called left singular vectors
- $A \mathbf{v}_j = \sigma_j \mathbf{u}_j$
  - where
    - $j = 1, ..., n$
- If $A$ has rank $r$,
  - then
    - (i) $\mathbf{v}_1, ..., \mathbf{v}_r$ form an orthonormal basis for $R(A^T)$.
    - (ii) $\mathbf{v}_{r+1}, ..., \mathbf{v}_n$ form an orthonormal basis for $N(A)$.
    - (iii) $\mathbf{u}_1, ..., \mathbf{u}_r$ form an orthonormal basis for $R(A)$.
    - (iv) $\mathbf{u}_{r+1}, ..., \mathbf{u}_m$ form an orthonormal basis for $N(A^T)$.
  - r = (the number of its nonzero singular values)
    - Note that it doesn't apply to the number of eigenvalues.
  - $A = U_1 \Sigma_1 V_1^T$
    - where
      - $U_1 = (\mathbf{u}_1, \mathbf{u}_2, ..., \mathbf{u}_r)$
      - $V_1 = (\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_r)$
    - called the compact form of the singular value decomposition of $A$

(Lemma 6.5.2)

If $A$ is an $m \times n$ matrix and $Q$ is an $m \times m$ orthogonal matrix. then,
$$ ||QA||_F = ||A||_F$$
.

(Theorem 6.5.3)

Let $A = U\Sigma V^T$ be an $m \times n$ matrix, and let $\mathcal{M}$ denote the set of all $m \times n$ matrices of rank $k$ or less, where $0 \lt k \lt \operatorname{rank}(A)$. If $X$ is a matrix in $M$ satisfying

$$||A - X||_F = \min\limits_{S \in \mathcal{M}} ||A - S ||_F$$
then
$$||A - X||_F = (\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_n^2) ^{1/2}$$
In particular, if $A' = U \Sigma'V^T$, where

$$
\Sigma' =
\left[
\begin{array}{c c c|c}
\sigma_1 & & & \\
& \ddots & & O \\
& & \sigma_k & \\
\hline
& O & & O
\end{array}
\right]
=
\left[
\begin{matrix}{}
\Sigma_k & O \\
O & O
\end{matrix}
\right]
$$

then
$$||A - A'||_F = (\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_n^2) ^{1/2} = \min\limits_{S \in \mathcal{M}} ||A - S ||_F$$

#### Application 1: numerical rank

(definition)

The numerical rank of an $m \times n$ matrix is the number of singular values of the matrix that are greater than $\sigma_1 \max(m, n)\epsilon$, where $\sigma_1$ is the largest singular value of $A$ and $\epsilon$ is the machine epsilon.

#### Application 2: digital image processing

$$A = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + \cdots + \sigma_n \mathbf{u}_n\mathbf{v}_n^T$$

$$A_k = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + \cdots + \sigma_k \mathbf{u}_k\mathbf{v}_k^T$$

- (The total storage for $A_k$) = $k(2n + 1)$.

#### Application 3: information retrieval - latent semantic indexing

- $\operatorname{argmin}\limits_{i} [Q^T x]_i$
  - the number of scalar multiplication: $mn$
- $\operatorname{argmin}\limits_{i} [Q_1^T x]_i = \operatorname{argmin}\limits_{i} [V_1 \Sigma_1 U_1^T x]_i$
  - where
    - $\sigma_{r+1} = \cdots = \sigma_{n} = 0$
  - the number of scalar multiplication: $r(m + n + 1)$

#### Application 4: psychology - principal component analysis (PCA)

- Given a $n \times p$ data matrix $X$ centered to each column mean
- finding orthonormal basis $\mathbf{y}_1, ..., \mathbf{y}_r$  that span $R(X)$ where $r \le p$ as maximize the variance of $\mathbf{y}_i$ one by one from $\mathbf{y}_1$.
- $\mathbf{y}_i = X\mathbf{v}_i = \sigma_i \mathbf{u}_i$
- note that
  - $\operatorname{var}(\mathbf{y}_1) = {1 \over n - 1}(X \mathbf{v}_1)^T X\mathbf{v}_1 = \mathbf{v}_1^T S \mathbf{v}_1$
  - $S = {1 \over n - 1} X^T X$
    - covariance matrix
  - $X = U_1 \Sigma_1 V_1^T = U_1W$
    - $U_1$: the principle hidden features
      - $n \times r$
    - $W$: representation of observed features as linear combinations of the principle hidden features
      - $r \times p$

### 6.6 Quadratic forms

- quadratic equation
  - $ax^2 + 2bxy + cy^2 + dx + ey + f = 0$
- quadratic form
  - $\mathbf{x}^TA\mathbf{x}$

#### Conic Sections

- imaginery conic
- degenerate conic
- non degenerate conic
  - standard position
    - circle
      - $x^2 + y^2 = r^2$
    - ellipse
      - ${x^2 \over \alpha^2} + {y^2 \over \beta^2} = 1$
    - hyperbola
      - ${x^2 \over \alpha^2} - {y^2 \over \beta^2} = 1$
      - ${y^2 \over \alpha^2} - {x^2 \over \beta^2} = 1$
    - parabola
      - $x^2 = \alpha y$
      - $y^2 = \beta x$
  - non standard position

TODO

### 6.7 Positive definite matrices

### 6.8 Nonnegative matrices

## 7. Numerical linear algebra

## 8. Iterative methods

## 9. Jordan canonical form


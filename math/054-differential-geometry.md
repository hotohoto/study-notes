# Differential geometry

## TODO

- make kappa and k clear in the notes

- 숙명여대 서검교 교수님
  - 1학기 - 미분기하학
    - https://youtube.com/playlist?list=PL85AYQZ4ks4JIO8pUgNAOXDDb7upeA0Et
  - 2학기 - 현대기하학
    - https://youtube.com/playlist?list=PL85AYQZ4ks4JFInW_Zcs5M8PFkI72BAkY
- Elementary Differential Geometry
  - https://play.google.com/books/reader?id=9nT1fOwATf0C&pg=GBS.PA12
  - ex 1.2.4
- [수학아빠sk... 어디교수님?](https://youtube.com/playlist?list=PL0ApUgH_3J1X3HTC9CX3r1dgJCgFy2V4W)
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


## Parameterized differentiable curve

https://youtu.be/4V8I02AhWsQ

- curve
  - how to define
    1. a subset
    2. path
      - a position dependent on t
      - a parameterized differentiable curve is a differentiable ($C^\infty$) map
        - $\alpha: (a, b) \subset R \to R^3$
          - open interval is used
  - differential geometry is about easy curves which is differentiable
- tangent vector
  - $\alpha'(t) = (x'(t), y'(t), z'(t)) \in R^3$
    - tangent vector of $\alpha$ at $t \in I$
- a plane curve
  - If $\exists$ a plane P s.t. $\alpha(I) \subset P$ then $\alpha$ is called a plane curve or planar curve
- rigid motion
  - translation
  - rotation
  - reflection
  - [glide reflection](https://www.regentsprep.org/glide-reflection/)
- examples
  - helix
    - $\alpha(t) = (a\cos{t}, a\sin{t}, bt)$
    - $t \in R$
    - $a, b \gt 0$
  - $\alpha(t) = (t^3, t^2)$
    - is a parameterized differenible curve
    - there is a cusp at the origin ($t=0$)
      - not smooth there
  - $\alpha(t) = (t, |t|)$
    - is not a parameterized differentiable
- discussion
  - if $\alpha''(t) = 0$ then what can be said about $\alpha$?
- Prop
  - $\alpha: I \to R^3$, a parameterized $C^\infty$ curve with $\alpha'(t) \neq 0, \forall t \in I$ then
    - $|\alpha(t)| = \text{const} \neq 0 \iff \alpha(t) \perp \alpha'(t)$
      - moving on a sphere

## Regular curves and arc length

https://youtu.be/VAry7h0jFyo

- regular curve
  - def
    - A curve $\alpha: I \to R^3$
    - parameterized $C^\infty$ curve
    - $\alpha'(t) \neq 0$
    - $\forall t \in I$
  - arc length
    - $s(t) = \int_{t_0}^{t}{|\alpha'(t)|dt}$
    - where curve $\alpha: I \to R^3$ is regular
    - note that, by the fundamental theorem of calculus,
      - ${ds \over dt} = |\alpha'(t)|$

(recap)

- cross product
  - $u \times v$
  - $u = (u_1, u_2, u_3) \in R^3$
  - $v = (v_1, v_2, v_3) \in R^3$
  - properties
    - $u \times v = - v \times u$
    - $(au + bw) \times = au \times v + bw \times v$
    - $u \times v = 0 \in R^3 \iff$ $u$ and $v$ are linearly dependent
    - $u \times v \cdot w = det(u, v, w) = \begin{vmatrix} u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \\ w_1 & w_2 & w_3 \end{vmatrix}$
      - $u \times v \cdot u = 0$
      - $u \times v \cdot v = 0$
    - $\{u, v, u \times v\}$ is a basis of $R^3$ if $u \times v \neq 0$
    - $(u \times v)\cdot(x \times y) = \begin{vmatrix} u \cdot x & v \cdot x \\ u \cdot y & v \cdot y \end{vmatrix}$
    - $(u \times v) \times w = (u \cdot v) v - (v \cdot w) u$
    - ${d \over dt}(u(t) \times v(t)) = u'(t) \times v(t) + u(t) \times v'(t)$


## Curvature and torsion of a curve

https://youtu.be/v2NpwCOZbqA

(The local theory of curves parameterized by arc length)

- $\alpha: \to R^3$, parameterized by arc length $s$
  - $s(t) = \int_{t_0}^{t} |\alpha'(u)|du$
- observe
  - $|\alpha'(s)| = 1$
    - This is an important observation!
- $|\alpha''(s)|$ measures the rate of change of the angle which neighboring tangents make with the tangents at s
  - e.g. for a curcle to be parameterized arc length it's like this.
    - $\alpha(s) = (r \cos{s \over r}, r\sin{s \over r}, 0)$
    - $\alpha'(s) = (-\sin{s \over r}, \cos{s \over r}, 0)$
    - $|\alpha'(t)| = 1$
    - $\alpha''(s) = (-{1 \over r}\cos{s \over r}, -{1 \over r}\sin{s \over r}, 0)$
    - $|\alpha'(t)| = {1 \over r}$
      - Note that $r$ is in inverse proportion to the curvature.
- curvature
  - of curve $\alpha$ parameterized by arc length
  - definition
    - $k(s) := |\alpha''(s)|$
- osculating plane
  - definition:
    - the plane determined by $\alpha'(s)$ and $\alpha''(s)$.
- $\alpha(s)$ singular point of order 1
  - $\iff \alpha''(s) = 0 \iff k(s) = 0$
- $\alpha(s)$ singular point of order 0
  - $\iff \alpha'(s) = 0$
    - not regular curve
- $T(s)$
  - unit tangent vector (field) of $\alpha$ at s
- $N(s)$
  - normal unit vector (field)
  - a unit vector in the direction of $\alpha''(s)$
  - notes
    - $\alpha''(s) = |\alpha''(s)|N(s)$ = k(s)N(s)
    - $N(s) = B(s) \times T(s)$
- $B(s)$
  - $T(s) \times N(s)$
  - binormal unit vector (field) at $s$
  - i.e. normal to the osculating plane
  - notes
    - $B'(s) = (T(s) \times N(s))' = T'(s) \times B(s) + T(s) \times B'(s) = T(s) \times N'(s)$
      - so
        - $B'(s) \perp T(s)$
        - $B'(s) \perp N'(s)$
    - $<B(s), B(s)>' = 2<B(s), B'(s)> = 0$
      - so
        - $B'(s) \perp B(s)$
    - thus
      - $B'(s) := \tau(s)N(s)$
        - since there is no $T$ or $B$ component.
      - $\tau(s)$
        - torsion of $\alpha$ at $s$
        - the measure of how the osculating plane is turning as $s$ varies
- Frenet-Serret frame
  - or TNB frame
  - or orthonomal frame
  - $\{T, N , B\}$


## A reparameterization of a curve

https://youtu.be/Fh3byc6gROM

(Reparameterization of a curve)

- $\alpha: (a, b) \to R^3$, regular
- $g: (c, d) \to (a, b)$ one-to-one and onto s.t. $g$ and $g^{-1}$: $(a, b) \to (c, d)$ are $c^\infty$, then $g$ is called a reparameterization of $\alpha$.
  - $a \lt t \lt b$
  - $c \lt r \lt d$
  - $t = g(r)$
  - $r = g^{-1}(t)$
- Note
  - Sometimes $\beta = \alpha \circ g$ is called a reparameterization of $\alpha$

### Regularity of $\beta$

$$
\beta: \text{regular} \iff {d\beta \over dr} \neq 0
$$

$$
{d\beta \over dr} = {d(\alpha \circ g) \over dr} = {d\alpha(t) \over dt}{dg(r) \over dr}
$$

$$
\beta: \text{regular} \iff \alpha: \text{regular} \wedge {dg \over dr} \neq 0 \iff \alpha: \text{regular}
$$

because

$$
g \circ g^{-1}(t) = t \Longrightarrow {dg \over dr}{dg^{-1} \over dt} = 1 \Longrightarrow {dg \over dr} \neq 0
$$

### arc length reparameterization

- $a \lt t \lt b$
- $0 \lt s \lt L$
- $t = t(s)$
- $s = s(t) = \int_a^t |\alpha'(t)|dt$
- $\alpha(t) = \beta(s(t)) = \beta(s) = \alpha(t(s))$
- $|\beta'(s)| = 1$

## Frenet-Serret Formula

https://youtu.be/UzMRb4JxQRs

### Euler's observation

- $\alpha(s) = (x(s), y(s), 0)$
- a plane curve
- unit speed
  - $|T(s)| = 1$
- $\alpha'(s) = (x'(s), y'(s), 0) = T(s)$
- $\theta$ : angle between the x-axis and the tangent vector field to $\alpha$ (which is $T(s)$)
- $\Rightarrow x'(s) = <T(s), (1,0,0)> = \cos \theta(s)$
- $T(s) = (\cos \theta(s), \sin \theta(s), 0)$
- $T'(s) = \alpha''(s) = (-\sin \theta(s) \theta'(s), \cos \theta(s) \theta'(s), 0)$
- $\text{(curvature of s)} = \kappa(s) = k(s) = |T'(s)| = |\theta'(s)|$
- $\Rightarrow$ The curvature of a plane curve is the rate of the change of the angle which the tangent vector field makes with the x-axis or any fixed direction.

### Orthonormality of {T, N, B}
(Lemma)

- $\alpha(s)$: parameterized by arc length
- $\Rightarrow$ For any $s$ with $k(s) \neq 0$, ${T, N, B}$ is an orthonormal set.

(pf)

- $T(s) = \alpha'(s)$
  - since $\alpha$ is parameterized by arc length
- $|T(s)| = |\alpha'(s)| = 1$
- $\Rightarrow \ <T, T> = 1$
- $\Rightarrow \ <T, T'> = 0$
- $\alpha''(s) = T'(s) = k(s)N(s)$
- $\Rightarrow \ <T, N> = 0$
- $|N| \triangleq 1$
- $B \triangleq T \times N$
- $\Rightarrow \ <B, T> = 0, \ <B, N> = 0, \ |B| = 1$
- Q.E.D.

We're going to introduce moving coordinates.

### Frenet-Serret Formula

Thm. $\alpha$: unit speed, $k(s) \neq 0$

$$
\left(\begin{array}{c}
T^{\prime} \\
N^{\prime} \\
B^{\prime}
\end{array}\right)=\left(\begin{array}{ccc}
0 & k & 0 \\
-k & 0 & -\tau \\
0 & \tau & 0
\end{array}\right)\left(\begin{array}{l}
T \\
N \\
B
\end{array}\right)
$$

$\iff$

- $T' = kN$
- $N' = -kT -\tau B$
- $B' = \tau N$

(such a matrix is said to be skew-symmetric)

(pf)

- $T' \triangleq kN$
- $N' = aT + bN + cB$
  - $a = <N', T> = -k$
    - $<T, N> = 0$
    - $<T', N> + <T, N'> = 0$
    - $\Rightarrow <T, N'> = - <kN, N> = -k$
  - $b = <N', N> = 0$
  - $c = <N', B> = -\tau$
    - $<N, B> = 0$
    - $\Rightarrow <N', B> + <N, B'> = 0$
    - $\Rightarrow <N', B> = - <N, B'> = - <N, \tau N> = -\tau$
- $B' \triangleq \tau N$

### A characterization of a plane curve

https://youtu.be/A8C6G-VI5nI

- Surprisingly, $T'''$ is a linear combination of $T$, $N$, and $B$.

Corollary

$\alpha(s)$ is a plane curve $k \neq 0$
- $\iff B(s) \equiv \text{const.}$
- $\iff \tau(s) \equiv 0$

(planes containing TNB framework)
- The osculating Plane at $\alpha(s)$
  - containing $T$, $N$
  - $\perp B$
- The normal plane
- containing $N$, $B$
  - $\perp T$
- The rectifying plane
  - containing $T$, $N$
  - $\perp N$

### A characterizaiton of a helix

https://youtu.be/A40S18IAF_Y


(helix)

- a regular curve $\alpha$ is a helix if $\exist$ a unit vector $u$ s.t. $<T, u> = |T||u|\cos{\theta} \equiv$ constant
- such $u$ is called the axis of the helix
- such $\theta$ is called the pitch of the helix

(ex) right circular helix
- $(r\cos{t}, r\sin{t}, ht)$
  - $h \gt 0$
  - $r \gt 0$
- with the axis $(0, 0, 1)$

Any regular plane curve is a helix

(1802) Lancret Theorem

- $\alpha(s)$ is a unit speed curve $k(s) \neq 0$
- $\alpha$ is a helix $\iff \exist$ constant $c$ s.t. $\tau = ck$
  - $c = -\cot{\theta}$ where $\theta$ is the pitch of the helix

### Property of a spherical curve

https://youtu.be/5goF_kbBUfE

Prep.
- if
  - $\alpha: I \to \mathbb{R}^3$
    - parameterized by arch length
  - $\{\alpha(I)\} \subset$ (the sphere of radious $r$ and center $p$)
- then
  - 1️⃣ $k \neq 0$
  - 2️⃣ if $\tau \neq 0$, then $\alpha - p = - \rho N  + \rho^\prime \sigma B$
    - where
      - $\rho = {1 \over k}$
        - a radious of curvature
      - $\sigma = {1 \over \tau}$
        - a radious of torsion

Proof
- 1️⃣
  - $|\alpha(s) - p|^2 = r^2$
  - $\Rightarrow \ <\alpha - p, \alpha - p>= r^2$
  - $\Rightarrow \ 2<\alpha - p, (\alpha - p)^\prime> = 0$
    - $\alpha^\prime = T$
  - $\Rightarrow \ <\alpha - p, \alpha^\prime> = 0$
  - $\Rightarrow \ <\alpha - p, T> = 0$
    - $\therefore \alpha - p \perp T$
  - $\Rightarrow \ <\alpha - p, T>^\prime = 0$
  - $\Rightarrow \ <T, T> + <\alpha - p, k N> = 0$
  - $\Rightarrow \ k <\alpha - p, N> = - 1$
  - $\Rightarrow \ k \neq 0$
- 2️⃣
  - Assume $\tau \neq 0$
  - $\alpha - p = aT + bN + cB$
  - $= <\alpha - p, T>T + <\alpha - p, N>N + <\alpha - p, B>B$
  - $= -\rho N + <\alpha - p, B>B$
    - $\because$
      - $\alpha - p \perp T$
      - $<\alpha - p, N> = - {1 \over k} = -\rho$
  - $= -\rho N + \rho^\prime \sigma B$
    - $\because$
      - $<\alpha - p, N> = -\rho$
      - $\Rightarrow \ <T, N> + <\alpha -p, -kT -\tau B> = -\rho^\prime$
      - $\Rightarrow \ <T, N> - k <\alpha -p, T> -\tau <\alpha - p, B> = -\rho^\prime$
      - $\Rightarrow \ 0 + 0 -\tau<\alpha -p, B> = -\rho^\prime$
      - $\Rightarrow \ <\alpha -p, B> = \rho^\prime \sigma$
        - $\because \tau \neq 0$

Remark
- $r^2 = <\alpha - p, \alpha - p> = \rho ^2 + (\rho^\prime \sigma)^2$
  - $\because$ 2️⃣

S.S. Chern said "Differentiate."

### Characterization of curves by curvature and torsion

https://youtu.be/5goF_kbBUfE?t=1626

- $k \equiv 0$
  - $\iff$ $\alpha$ is a straight line
- $k \neq 0, \tau \equiv 0$
  - $\iff$ $\alpha$ is a plane curve
- ${k \over \tau} = \text{const}$
  - $\iff$ $\alpha$ is a helix
- $k = \text{const} \gt 0, \tau \equiv 0$
  - $\iff$ $\alpha$ is a circle
  - TODO proof
- $k = \text{const}, \tau \equiv \text{const} \neq 0$
  - $\iff$ $\alpha$ is a circular helix
  - TODO proof

### Picard–Lindelöf theorem

- https://youtu.be/5goF_kbBUfE?t=2175
- https://en.wikipedia.org/wiki/Picard%E2%80%93Lindel%C3%B6f_theorem

(version 1)

An initial value problem (IVP) is given as follows.

- $y^\prime = f(x, y(x)) = f(x, y)$
- $y(x_0) = y_0$
- $y: \mathbb{R} \to \mathbb{R}$

if
- $f(x, y)$ is continuous in $R = \{|x-x_0| \le a, |y-y_0| \le b\}$
    - where $a, b \gt 0$
- and $f(x, y) \le \sup\limits_{R}|f(x, y)| := K$

$\Rightarrow$
  - the IVP has a unique solution $y = y(x)$ in $|x - x_0| \le \beta$
    - where $\beta := \min\{a, {b \over K}\}$

(version 2)

An initial value problem (IVP) is given as follows.

- ${d\alpha \over dt} = F(t, \alpha(t)) = F(t, \alpha)$
- $\alpha(t_0) = P$
- $\alpha: \mathbb{R} \to \mathbb{R}^3$

if
- $F(t, \alpha)$ is continuous in $R = \{|x-x_0| \le a, |\alpha-P| \le b\}$
    - where $a, b \gt 0$
- and $F(t, \alpha) \le \sup\limits_{R}|F(t, \alpha)| := K$

$\Rightarrow$
  - the IVP has a unique solution $\alpha = \alpha(t)$ in $|t - t_0| \le \beta$
    - where $\beta := \min\{a, {b \over K}\}$

(version 3)

It can be extended to system equations.

### Isometry map

https://youtu.be/5goF_kbBUfE?t=2934

- definition:
  - $M: \mathbb{R}^3 \to \mathbb{R}^3$ is an isometry map
    - if $|M(p) - M(q)| = |p - q|$
      - $\forall p, q \in \mathbb{R}^3$.
- also called a rigid motion
- a mapping that preserves distances
- examples:
  - Refection
  - Rotation
  - Translation

### Fundamental theorem of the local theory of curves

https://youtu.be/5pyegExDAss
https://youtu.be/zY0-L3mzJUY

Question:

- Given two scalar valued functions or scalar fields, is there a curve that the curvature and the torsion of it are those two functions?
- If so, how many such curves are there?

Answer:

- Yes, and it's unique.
- Any regular curve $\alpha(s)$ with $k(s) > 0$ is completely determined up to rigid motions by $k(s)$ and $\tau(s)$
  - where $k(s) = 0$ it's trivial.
- More precisely
  - given
    - $0 \in (a, b) = I$
      - Actually 0 can be any number we set.
    - $\bar{k}(s) \gt 0$, $\bar{\tau}(s)$ s.t $C^\infty$ on $I$
    - $P$: a fixed point in $\mathbb{R}^3$
    - $\{D, E, F\}$: orthonormal basis of $\mathbb{R}^3$ at $P$
  - $\Rightarrow$
    - $\exist!$ regular curve $\alpha: I \to \mathbb{R}^3$ s.t.
      - 1
        - the parameter is arc length from $\alpha(0)$
      - 2
        - $\alpha(0) = P$
        - $T(0) = D$
        - $N(0) = E$
        - $B(0) = F$
      - 3
        - $k(s) = \bar{k}(s)$
        - $\tau(s) = \bar{\tau}(s)$
      - Note that $!$ means "uniquely"

(Proof)

- Existence:
  - step 1
    - Consider the system of ODE
      - $u^\prime_j(s) = a^1_j(s) u_1(s) + a^2_j(s) u_2(s) + a^3_j(s) u_3(s)$
        - $= \sum\limits_{i=1}^3 a^i_j(s) u_i(s)$
        - where
          - $j \in \{1, 2, 3\}$
          - $
              (a^i_j)=\left(\begin{array}{ccc}
              0 & \bar{k} & 0 \\
              -\bar{k} & 0 & -\bar{\tau} \\
              0 & \bar{\tau} & 0
              \end{array}\right)
            $
          - $u_1(0) = D$
          - $u_2(0) = E$
          - $u_3(0) = F$
    - By ODE theory (Picard–Lindelöf theorem) this system has a unique solution.
  - step 2
    - Claim: $\{u_j\}$  is a moving frame of $\alpha$ with $\bar{k}$ and $\bar{\tau}$
      - (proof of claim)
        - I. $\{u_i\}$ is an orthonormal set?
          - $P_{ij}(s) := <u_i(s), u_j(s)>$
          - $P_{ij}(s)^\prime := <u_i(s)^\prime, u_j(s)> + <u_i(s), u_j(s)^\prime>$
          - $P_{ij}(s)^\prime=<a^k_i u_k, u_j> + <u_i, a^k_j u_k>$
            - Einstein summation convention
          - $P_{ij}(s)^\prime=a^k_i P_{kj} + a^k_j P_{ik}$
          - We can see this as a system of differential equation again
            - with initial values:
              - $P_ij(0) = \delta_{ij}$
                - where
                  - $\because D, E, F$ are orthonormal basis at $P$
                  - $\delta_{ij}$ is the Kronecker-delta function
                    - 0: $i \neq j$
                    - 1: $i = j$

          - By ODE theory (Picard–Lindelöf theorem) this system has a unique solution.
            - and $\delta_{ij}$ is the only solution
            - note:
              - $a^k_i \delta_{kj} + a^k_j \delta_{ik}$
              - $= a^j_i + a^i_j$
              - $= 0 = (\delta_{kj})^\prime$
            - $\therefore \{u_j\}$ is an orthonormal set
        - II.
          - Let $\alpha(s) = P + \int^s_0u_1(\sigma)d\sigma$
            - for $s \in (a, b)$
          - then
            - $\alpha \in C^\infty(I, \mathbb{R}^3)$
              - $\because \alpha^{\{n\}}$ is a linear combination of $u_1$, $u_2$, $u_3$ no matter how many times it is differentiated
                - $\alpha(s)^\prime = u_1(s)$
                - $\alpha(s)^{\prime\prime} = u_1^\prime(s) = \bar{k}u_2(s)$
                - ...
            - $\alpha$ is regular and unit speed
              - $|\alpha(s)^\prime| = 1 \gt 0$
                - $\because$ I
        - III. wants $k=\bar{k}$, $\tau=\bar{\tau}$, $u_1=T$, $u_2=N$, $u_3=B$
          - $\alpha^\prime  = T = u_1$
            - $\because$ II
          - $T^\prime = kN = u_1^\prime = \bar{k}u_2(s) = 1$
          - $\Longrightarrow k = \bar{k}$
            - $\because$
              - $k > 0$ and $\bar{k} > 0$
              - $N$ and $u_2$ are unit vectors
          - $\Longrightarrow u_2 = N$
          - $<u_1 \times u_2, u_3> = \pm 1$
            - $\because \{u_j\}$ is an orthonormal set
          - At $s=0$
            - $<u_1(0) \times u_2(0), u_3(0)> = <D \times E, F> = 1$
            - $<u_1 \times u_2, u_3> \equiv 1$
              - $\because <u_1 \times u_2, u_3>$ is continuous
          - $\Longrightarrow \{u_j\}$ is right-handed
          - $\Longrightarrow u_3 = B$
          - $\tau N = B^\prime = u_3^\prime = \bar{\tau} u_2$
          - $\Longrightarrow \tau = \bar{\tau}$
            - $\because$ $N$ and $u_2$ are unit vectors
- Uniqueness
  - observation
    - Arc length is invariant under rigid motions.
      - i.e.
        - given a rigid motion $M: \mathbb{R}^3 \to \mathbb{R}^3$,
        - $\int_a^b|\alpha^\prime(t)|dt = \int_a^b|(M \circ \alpha)^\prime(t)|dt$
  - Suppose $\exist$ two curves $\alpha(s)$ and $\beta(s)$ s.t.
    - $k_\alpha(s) = k_\beta(s)$
    - $\tau_\alpha(s) = \tau_\beta(s)$
    - orthonomral basis are given as
      - $\{T_\alpha, N_\alpha, B_\alpha\}$
      - $\{T_\beta, N_\beta, B_\beta\}$
    - $\forall s \in I$
  - At a given $s = s_0 \in I$, $\exist$ a rigid motion M s.t.
    - $\alpha(s_0) = M \circ \beta(s_0)$
    - $T_\alpha(s_0) = M \circ T_\beta(s_0)$
    - $N_\alpha(s_0) = M \circ N_\beta(s_0)$
    - $B_\alpha(s_0) = M \circ B_\beta(s_0)$
  - So we may assume that
    - $\alpha(s_0) = \beta(s_0)$
    - $T_\alpha(s_0) = T_\beta(s_0)$
    - $N_\alpha(s_0) = N_\beta(s_0)$
    - $B_\alpha(s_0) = B_\beta(s_0)$
  - Frenet-Serret Formula
    - $
      \left(\begin{array}{c}
      {T_\alpha}^{\prime} \\
      {N_\alpha}^{\prime} \\
      {B_\alpha}^{\prime}
      \end{array}\right)=\left(\begin{array}{ccc}
      0 & k & 0 \\
      -k & 0 & -\tau \\
      0 & \tau & 0
      \end{array}\right)\left(\begin{array}{l}
      T_\alpha \\
      N_\alpha \\
      B_\alpha
      \end{array}\right)
      $
    - $
      \left(\begin{array}{c}
      {T_\beta}^{\prime} \\
      {N_\beta}^{\prime} \\
      {B_\beta}^{\prime}
      \end{array}\right)=\left(\begin{array}{ccc}
      0 & k & 0 \\
      -k & 0 & -\tau \\
      0 & \tau & 0
      \end{array}\right)\left(\begin{array}{l}
      T_\beta \\
      N_\beta \\
      B_\beta
      \end{array}\right)
      $
  - Observe
    - ${1\over2}(|T_\alpha - T_\beta|^2 + |N_\alpha - N_\beta|^2 + |B_\alpha - B_\beta|^2)^\prime$
    - $=<T_\alpha - T_\beta, {T_\alpha}^\prime - {T_\beta}^\prime> + <N_\alpha - N_\beta, {N_\alpha}^\prime - {N_\beta}^\prime> + <B_\alpha - B_\beta, {B_\alpha}^\prime - {B_\beta}^\prime>$
    - $=k<T_\alpha - T_\beta, {N_\alpha} - {N_\beta}> - k <N_\alpha - N_\beta, {T_\alpha} - {T_\beta}> -\tau <N_\alpha - N_\beta, {B_\alpha} - {B_\beta}> + \tau<B_\alpha - B_\beta, {N_\alpha} - {N_\beta}>$
    - $=0$
    - $\Longrightarrow |T_\alpha - T_\beta|^2 + |N_\alpha - N_\beta|^2 + |B_\alpha - B_\beta|^2 = const. = 0$
      - $\because$ the equality assumptions at $s_0$
    - $\Longrightarrow$
      - $T_\alpha = T_\beta$
      - $N_\alpha = N_\beta$
      - $B_\alpha = B_\beta$
  - $\Longrightarrow \alpha^\prime = T_\alpha = T_\beta = \beta^\prime$
  - $\Longrightarrow (\alpha - \beta)^\prime = 0$
  - $\alpha = \beta$
    - $\because$ the equality assumption at $s_0$

(Remark)

- For the plane curves, we can fix $N$ with respect to $T$ among two possible choices, then $k$ becomes an oriented/signed curvature.
- But, in $\mathbb{R}^3$ there are infitely many choices of $N$.


### An explicit representation of a helix

https://youtu.be/y3C8sxGFTuc
...

TODO





## Misc.

- cross ratio
  - ǁa-cǁǁb-dǁ/ǁa-dǁǁb-cǁ
- Manifolds
  - topological space which is locally Euclidean space and metrizable
- Curvature
  - related to absolute value the second order derivative of a function
  - κ
- Torsion
  - How sharply it is twisting
  - τ
  - curve β is planar ⇔ τ = 0
  - curve β is a circle ⇔ τ = 0 and κ > 0 and κ is a constant
- Osculating plane
- Osculating circle
- Gaussian curvature
  - $K=\kappa_{1}\kappa_{2}$
  - $\kappa_{1}$ and $\kappa_{2}$ are principal curvatures.
    - pick a point P
    - there exists tangent space $T_{p}M$ which is $\mathbb{R}^2$ space
    - but it may not be an Euclidean space
      
      - the inner product can be different from the dot product
      
      - P corresponds to the origin in the tagent space
    - consider normal vector N on the P
    - slice the surface with a hyperplane containing the normal vector N
      - there are going to infintely many curves
    - $\kappa_{1}$ and $\kappa_{2}$ going to be the max and min curvature of those curves
      - these can be found by using the spectral theorem.
  - when Gaussian curvature is a constant over the space
    - Sphere $K \gt 0$
    - Euclidean space $K = 0$
    - Half plane $K \lt 0$
- flatness of $\mathbb{R}^n$ space
  - inner product can be calculated by an identity matrix
- convex
  - the second derivative is positive
- concave
  - the second derivative is negative

## Gauss–Bonnet theorem

$$
\int_M K\;dA+\int_{\partial M}k_g\;ds=2\pi\chi(M)
$$

- $K$: Gaussian curvature
- $M$: compact 2 dimensional Riemannian manifold with boundary $\partial M$
- $k_g$: geodesic curvature of $\partial M$
- $dA$: element of Area
- $ds$: the line element along the boundary of $M$
- $\chi(M)$: Euler characteristic of $M$
  - $\chi = V - E + F$
  - $V$
    - number of vertices (corner)
  - $E$
    - number of edges
  - $F$
    - number of faces

- (sum of inner angles of triangles made from geodesics) = 180° + Surface integral of Gaussian curvature for the triangle


## Gauss's Theorema Egregium

Gauss's remarkable theorem

The theorem is that Gaussian curvature can be determined entirely by measuring angles, distances and their rates on a surface, without reference to the particular manner in which the surface is embedded in the ambient 3-dimensional Euclidean space)

- Thanks to "multiplication" of two principal curvatures when calculating Gaussian curvature
- So Gaussian curvature is "intrinsic".



(notes)

- many second dimensional surfaces can be classified by the number of genera
  - https://en.wikipedia.org/wiki/Genus_(mathematics)
- thrid dimensional surfaces can be classified/decomposed as 8 elementary types.
  - Geometrization Conjecture
  - related to Poincaré conjecture

## Tensor

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
- Gradient operator
- Curvilinear coordinates
- Metric tensor
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
- Connection
  - https://en.wikipedia.org/wiki/Connection_(mathematics)

- Affine connection
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
  - A pretty wide definition so there exist infinitely many affine connections
  - Christoffel symbols specify a corresponding affine connection
    - In Euclidean space Chrisotffel symbols are all zero
  - So what are reasonable connections? How can we define them?
    - 👉 Levi-Civita connection
  - defining affine connection is equivalent to
    - defining a Christoffel symbol
    - defining geodesic
    - defining how to do parallel transport
    - defining how to differentiate
  - an affine connection is not a tensor
- Levi-Civita connection
  - a kind of affine connection given a metric tensor
  - satisfies
    - linearity and Leibniz rule
    - torsion free
      - $\Gamma^k_{ij} = \Gamma^k_{ji}$
      - two parallel transport paths makes a parallelogram when they are done with switched orders
    - metric compatibility
      - ${\partial}_k g_{ij} = \Gamma^l_{ik} g_{jl} + \Gamma^l_{jk} g_{il}$
      - meaning it preserves the metric
  - $\Gamma^m_{jk} = {1 \over 2} g^{im} ({\partial}_k g_{ij} + {\partial}_j g_{ki} + {\partial}_i g_{jk})$
  - Fundamental theorem of Riemannian geometry
    - For a Riemannian manifold (curved space with a metric), there is a unique connection (=covariant derivative) that is torsion-free and has metric compatibility. And this connection is called the Levi-Civita connection.
- Christoffel symbol
  - an array of numbers describing an affine connection
  - sometimes called the (affine/Levi-Civita) connection coefficients
  - ${\frac {\partial \mathbf {e} _{i}}{\partial x^{j}}}={\Gamma ^{k}}_{ij}\mathbf {e} _{k}=\Gamma _{kij}\mathbf {e} ^{k}$
    -  When the point moves along with the direction of $x^j$, how the basis vector $e_i$ changes in terms of all the current basis.
   - depends on how we define the inner product
- Parallel transport (along a curve)
  - a way of transporting geometrical data along smooth curves in a manifold
  - keeps vectors as constant as possible
    - but note that it could be impossible to keep vectors constant on a surface like a sphere
    - in another word, it's impossible to define a constant vector field on a curved surface
  - keeps the length of a vector constant
- Differentiation in Tensor Field
- Second covariant derivative

- Geodesic

- Lie bracket
  - https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields
  - a vector field can be seen as a derivative operator
  - given two vector fields Lie Bracket generates another vector field indicating that the tow vector fields as operators can be commutative

- Torsion tensor
- Torsion free tensor

- Riemann curvature tensor

- Ricci Tensor
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

- Isometry

- First fundamental form
- Second fundamental form
- Gauss' theorema egregium (Gauss' remarkable theorem)

- Weingartenm map or shape operator

## References

(main)

- [Elementary Differential Geometry](https://play.google.com/books/reader?id=9nT1fOwATf0C)
- [Tensor Calculus by eigenchris](https://youtube.com/playlist?list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx)


(extra)

- http://www.kocw.net/home/cview.do?mty=p&kemId=1197788&ar=relateCourse

- https://en.wikipedia.org/wiki/Metric_tensor
- https://en.wikipedia.org/wiki/Affine_connection
- https://en.wikipedia.org/wiki/Tensor_product
- https://en.wikipedia.org/wiki/Leibniz_integral_rule
- https://en.wikipedia.org/wiki/Lie_bracket_of_vector_fields


- http://www.math.uchicago.edu/~may/VIGRE/VIGRE2008/REUPapers/Halper.pdf
- https://www.mathematik.hu-berlin.de/~wendl/pub/connections_chapter6.pdf
- [A Gentle Introduction to Tensors](https://www.ese.wustl.edu/~nehorai/Porat_A_Gentle_Introduction_to_Tensors_2014.pdf)


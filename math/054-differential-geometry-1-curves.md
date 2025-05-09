[TOC]

# Differential geometry - elementary

## TODO

- make kappa and k clear in the notes
  - kappa
  - k
- 숙명여대 서검교 교수님: 1학기 - 미분기하학
  - https://youtube.com/playlist?list=PL85AYQZ4ks4JIO8pUgNAOXDDb7upeA0Et
- Elementary Differential Geometry
  - https://play.google.com/books/reader?id=9nT1fOwATf0C&pg=GBS.PA12
  - ex 1.2.4
- [수학아빠sk... 어디교수님?](https://youtube.com/playlist?list=PL0ApUgH_3J1X3HTC9CX3r1dgJCgFy2V4W)



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

## A characterization of a plane curve

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

## A characterization of a helix

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

## Property of a spherical curve

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

## Fundamental theorem of the local theory of curves

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


## An explicit representation of a helix

https://youtu.be/y3C8sxGFTuc?si=JZuXeM6ya2hs8R3t

(e.g.)

- Let $\alpha(s)$ be a helix with $k \gt 0$ $(\tau = ck)$ for some constant $c$
- Find $\alpha(s)$ explicitly



(sol.)

- Reparametrize $\alpha$ given by a parameter $t$

$$
t(s) = \int_0^s k(\sigma)\mathrm{d}\sigma
$$

(This is a nice idea! 👍)
$$
t^\prime = {\mathrm{d}t \over \mathrm{d}s} = k(s) \gt 0
$$

$$
\Longrightarrow t: \text{1-1}
$$

$$
\left(\begin{array}{c}
T^{\prime} \\
N^{\prime} \\
B^{\prime}
\end{array}\right)
=\left(\begin{array}{ccc}
0 & k & 0 \\
-k & 0 & -\tau \\
0 & \tau & 0
\end{array}\right)
\left(\begin{array}{l}
T \\N \\B
\end{array}\right)
$$

$$
=\left(\begin{array}{ccc}0 & k & 0 \\-k & 0 & -ck \\0 & ck & 0\end{array}\right)\left(\begin{array}{l}T \\N \\B\end{array}\right)
$$

(so far derivations are w.r.t. $s$, and $t$ is just a variable yet.)

1️⃣
$$
T^\prime
= {\mathrm{d}T \over \mathrm{d}s}
= {\mathrm{d}T \over \mathrm{d}t}{\mathrm{d}t \over \mathrm{d}s}
= {\mathrm{d}T \over \mathrm{d}t}k
$$

$$
\Longrightarrow {dT \over \mathrm{d}t} = N
$$

2️⃣
$$
N^\prime = {\mathrm{d}N \over \mathrm{d}s} = {\mathrm{d}N \over \mathrm{d}t}{\mathrm{d}t \over \mathrm{d}s} = {\mathrm{d}N \over \mathrm{d}t}k
$$

$$
\Longrightarrow {\mathrm{d}N \over \mathrm{d} t} = -T -cB \\
$$

3️⃣
$$
B^\prime = {\mathrm{d}B \over \mathrm{d}s} = {\mathrm{d}B \over \mathrm{d}t}k
$$

$$
\Longrightarrow {\mathrm{d}B \over \mathrm{d}t} = cN
$$

Thus, 
$$
{\mathrm{d}^2N \over \mathrm{d} t^2} = {\mathrm{d} \over \mathrm{d}t}(-T - cB) = -N -c^2N = - (1 + c^2)N 
$$
Let
$$
\omega ^2 = (1 + c^2)
$$
Then,
$$
N = \cos(\omega t)C_1 + \sin(\omega t) C_2
$$
for some fixed "vectors": $C_1$, $C_2$, $C_3$, and $C_4$.
$$
{\mathrm{d}T \over \mathrm{d}t} = N \\

\Longrightarrow 
T = {1\over \omega}
\left(
\sin(\omega t)C_1 - \cos(\omega t)C_2 + C_3
\right)
= \alpha^\prime(s) \\

\Longrightarrow
\alpha(s) = {1\over \omega} 
\left(
\int_0^s \sin(\omega t(\sigma)) \mathrm{d}\sigma \cdot C_1
- \int_0^s \cos(\omega t(\sigma)) \mathrm{d}\sigma \cdot C_2
+ C_3 s + C_4
\right)
$$
We've found the curve except the constant vectors.

Let's determine them.
$$
{\mathrm{d}N \over \mathrm{d}t}
= {\mathrm{d}^2T \over \mathrm{d}t^2}
= - \omega \sin(\omega t)C_1 + \omega\cos(\omega t)C_2
$$
Also,
$$
\begin{align}
0 \equiv & \langle N, {\mathrm{d}N \over \mathrm{d}t}\rangle \\
= & \langle
\cos(\omega t) C_1 + \sin(\omega t) C_2, 
- \omega \sin(\omega t)C_1 + \omega\cos(\omega t)C_2
\rangle \\
= & \omega( - \Vert C_1 \Vert ^2 + \Vert C_2\Vert^2)\cos(\omega t)\sin(\omega t)
+ \omega \langle C_1, C_2 \rangle (\cos^2(\omega t) - \sin^2 (\omega t))
\end{align}
$$
(At $t=0$, $\langle C_1, C_2 \rangle = 0$)
$$
\therefore
0 = ( - \Vert C_1 \Vert ^2 + \Vert C_2\Vert^2) \\
\Longrightarrow
\Vert C_1 \Vert = \Vert C_2 \Vert
$$

$$
\begin{align}
1 \equiv & \Vert N \Vert ^2 \\
= & \langle N, N \rangle \\
= & \langle \cos(\omega t) C_1 + \sin(\omega t) C_2, \cos(\omega t) C_1 + \sin(\omega t) C_2 \rangle \\
= & \cos^2(\omega t)\Vert C_1 \Vert ^2 + \sin^2(\omega t)\Vert C_2 \Vert ^2 \\
= & \Vert C_1 \Vert ^2
\end{align}
$$

$$
\therefore \Vert C_1 \Vert = \Vert C_2 \Vert = 1
$$

Similarly,
$$
\begin{align}
0 \equiv & \langle T, N \rangle \\
= & {1\over \omega} \langle
\sin(\omega t)C_1 - \cos(\omega t) C_2 + C_3,
\cos(\omega t) C_1 + \sin(\omega t) C_2
\rangle \\
\end{align}
$$
(At $t = 0$, $\langle C_3, C_1 \rangle = 0$)
$$
0 \equiv \langle C_3, sin(\omega t) C_2 \rangle \\
\therefore \langle C_3, C_2 \rangle = 0
$$
And then,
$$
\begin{align}
1 = &\Vert T \Vert ^2 \\
 = &{1 \over \omega^2 } (1 + \Vert C_3 \Vert^2)
\end{align}
$$

$$
\omega^2 = 1 + c^2 = 1 + \Vert C_3 \Vert^2 \\
\therefore \Vert C_3 \Vert = \vert c \vert
$$

$$
\Longrightarrow C_3 = \pm c (C_1 \times C_2 )
$$

Let's decide the sign.
$$
\begin{align}
cT = & c (N \times B) \\
= & N \times cB \\
= & N \times -({\mathrm{d}N \over \mathrm{d}t} + T) \\
= & \left( \cos(\omega t)C_1 + \sin(\omega t) C_2 \right)
\times
-\left(
- \omega \sin(\omega t)C_1 + \omega\cos(\omega t)C_2 + {1\over \omega}
\left(
\sin(\omega t)C_1 - \cos(\omega t)C_2 + C_3
\right)
\right) \\

= & \left( \cos(\omega t)C_1 + \sin(\omega t) C_2 \right)
\times
\left(
\left(\omega - {1\over\omega} \right)
\sin(\omega t)C_1
- \left(\omega - {1\over\omega} \right)
\cos(\omega t)C_2
- {1 \over \omega} C_3
\right) \\

\end{align}
$$
Also,
$$
cT = {c\over \omega}
\left(
\sin(\omega t)C_1 - \cos(\omega t)C_2 + C_3
\right)
$$
To decide the sign of $C_3$ we only need to compare the coefficients of $C_3$ and $C_1 \times C_2$.
$$
- \left(\omega - {1\over\omega} \right) (C_1 \times C_2) = {c \over \omega} C_3 \\
(- \omega ^2 + 1)(C_1 \times C_2) = c C_3 \\ 
- c ^2(C_1 \times C_2) = c C_3 \\ 
- c(C_1 \times C_2) = C_3 \\ 
$$
And, we observe
$$
\alpha(0) = {1 \over \omega} C_4
$$
.

Therefore, with respect to an orthonormal basis $\left\{ A, B, A\times B \right\}$, 
$$
\alpha(s) = {1 \over \omega}
\left(
\int_0^s \sin(\omega t(\sigma)) \mathrm{d}\sigma,
- \int_0^s \cos(\omega t(\sigma)) \mathrm{d}\sigma,
-cs
\right)
+ \alpha(0)
$$
where
$$
t(\sigma) = \int_0^\sigma k(y)\mathrm{d}y
$$
and
$$
\omega^2 = 1 + c^2
$$
.

Q.E.D.



## Local canonical form of a curve

- https://youtu.be/ol5oYW--UTc?si=NAyu23iznwz9zSxY
- (canonical can mean kind of natural)

$\alpha(s): I \to \mathbb{R}^3$

- parameterized by arc length s.t. $k(s) \gt 0$

By Taylor expansion,

$\alpha(s) = \alpha(0) + s\alpha^\prime(0) + {s^2 \over2!}\alpha^{\prime\prime}(0) + {s^2 \over2!}\alpha^{\prime\prime\prime}(0) + R(s)$

where $\lim \limits_{s \to 0} {R\over s^3} = 0$.

Note that,

- $\alpha^\prime = T$

- $\alpha^{\prime\prime} = kN$

- $\alpha^{\prime\prime\prime} = k^\prime N + kN^\prime = k^\prime N -k^2T - k\tau B$



TODO https://youtu.be/ol5oYW--UTc?si=R9oosH9Vde9Vwxi_&t=2429



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
      
      - P corresponds to the origin in the tangent space
    - consider normal vector N on the P
    - slice the surface with a hyperplane containing the normal vector N
      - there are going to infinitely many curves
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



## References

(main)

- [Elementary Differential Geometry](https://play.google.com/books/reader?id=9nT1fOwATf0C)


(extra)

- [세종대학교 오장헌 미분기하학 1](http://www.kocw.net/home/cview.do?mty=p&kemId=1197788&ar=relateCourse)


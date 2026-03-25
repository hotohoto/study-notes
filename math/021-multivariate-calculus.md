# Multivariate Calculus

- Fundamental theorem of calculus
- Fundamental theorem of line integral
- Green theorem
- Stoke theorem
- Gauss' divergence theorem

---

https://namu.wiki/w/%EB%AF%B8%EB%B6%84%ED%98%95%EC%8B%9D

- dx is a matrix or linear transform, especially covector
  - $dx: \mathbb{R}^2 \to R$
  - 미분형식 (differential form)
  - projection
- exterior derivative
- wedge product

## Divergence

## Jacobian

- https://youtu.be/wCZ1VEmVjVo?si=DaHMG2YYeVP3T_0J
- https://angeloyeo.github.io/2020/07/24/Jacobian.html

(keywords)
- derivative
    - "scaling factor near the point a"
    - "scaling factor near the point (a, b)"
- integral
    - density
    - mass
- linear map
    - origin is fixed
    - determinant
- general functions
    - $f(x) = x^2$
    - $f(x, y) = (x^2 - y^2, 3xy)$
- jacobian matrix
    - when the general functions are approximated "around" the point (a,b)
    - it's the linear map
    - (a,b) is fixed
- jacobian determinant
- chain rule
- changing variables in integration
    - infinitesimal length also changes
        - how much?
            - scale factor
    - injective g

$$
{\begin{gathered}
    \int_{g(a)}^{g(b)} f(x) dx = \int_a^b f(g(u))g^\prime(u)du \\
    \iint_{g(D)} f(x,y) xdy = \iint_D f(g(u, v)) \text{abs}{\vert J \vert} du dv
\end{gathered}}
$$

# Generalized Linear Models

- https://youtube.com/playlist?list=PLmM_3MA2HWpYoG_q69ZzbCSwvriLuimoO

## Background

##### Overdispersed exponential family

$$
f(y ; \theta, \phi) = \exp({y\theta - b(\theta) \over \phi/w} - c(y, \phi))
\tag{1}
$$

where

- $\theta$
  - the canonical parameter
- $\phi \gt 0$
  - known dispersion parameter
- $w \gt 0$
  - known weights
- $b$
  - a smooth function (log partition)
- $c$
  - doesn't depend on $\theta$

Notes

$$
\int _{-\infty}^\infty
f(y;\theta, \phi) \mathrm{d}y = 1
\tag{2}
$$

$$
{\partial^k \over \partial \theta^k} \int_{-\infty}^{\infty} f(y|\theta, \phi) \mathrm{d}y
= \int_{-\infty}^{\infty} {\partial^k \over \partial \theta^k} f(y|\theta, \phi) \mathrm{d}y
= 0 \\

\Rightarrow

\int_{-\infty}^{\infty} f^\prime (y;\theta, \phi) \mathrm{d}y
= 0 \\

\Rightarrow

\int_{-\infty}^{\infty} f^{\prime\prime} (y;\theta, \phi) \mathrm{d}y
= 0

\tag{3}
$$
where

- $k$
  - number of differentiation



Let $l = \ln f$ 

- which is the log likelihood

then,
$$
l^\prime = {f^\prime \over f}
\tag{4a}
$$

$$
l^{\prime\prime} = {f^{\prime\prime} \over f} - \left({f^\prime \over f} \right)^2
\tag{4b}
$$

where

- the differentiations are with respect to $\theta$



$$
\mathbb{E}(l^\prime) = \int l^\prime f \mathrm{d}y = \int f^\prime \mathrm{d}y = 0
\tag{5}
$$
where

- the differentiations are with respect to $\theta$



$$
\mathbb{E}[l^{\prime\prime}] \\
= \int f^{\prime\prime} - \left({f^{\prime} \over f}\right)^2 f \mathrm{d}y \\
= \int f^{\prime\prime} \mathrm{d}y - \int \left({f^{\prime} \over f}\right)^2 f \mathrm{d}y \\
= 0 - \mathbb{E}[(l^\prime)^2] \\
= - \operatorname{Var}[{l^\prime}]
\tag{6}
$$



##### Log likelihood

$$
l = {y\theta - b(\theta) \over \phi/w} - c(y, \phi)
\tag{7}
$$

$$
l^\prime = {y - b^\prime(\theta) \over \phi/w}
\tag{8}
$$

$$
l^{\prime\prime} = - {b^{\prime\prime}(\theta) \over \phi/w}
\tag{9}
$$



##### Mean

$$
0 = \mathbb{E}[l^{\prime}] = \mathbb{E}\left[{y - b^\prime(\theta) \over \phi/w}\right]\\
\Rightarrow \\
\mathbb{E}\left[ y \right] = b^{\prime}(\theta)
\tag{10}
$$

- by Eq. (5) and Eq. (8)



##### Variance

$$
\operatorname{Var} \left[ l^\prime \right]
= - \mathbb{E}\left[l^{\prime\prime}\right] \\

\operatorname{Var} \left[ {y - b^\prime(\theta) \over \phi/w} \right]
= - \mathbb{E}\left[- {b^{\prime\prime}(\theta) \over \phi/w}\right] \\

{1 \over (\phi/w)^2} \operatorname{Var}\left[ y\right]
= {b^{\prime\prime}(\theta) \over \phi/w} \\

\operatorname{Var}\left[ y \right]
= b^{\prime\prime}(\theta) \cdot {\phi \over w}

\tag{11}
$$



## Canonical link function

##### Density

$$
f(y ; \theta, \phi) = \exp({y\theta - b(\theta) \over \phi/w} - c(y, \phi))\tag{1}
$$



##### Link function

$$
\mu = \mathbb{E}(y) = b^\prime(\theta) = g^{-1}(\theta)
\tag{12}
$$

$$
g(\mu) = \theta = \mathbf{x}^{T}\boldsymbol{\beta}
\tag{13}
$$

where

- $g$ is called the canonical link function
- we assume $b^\prime$ is invertible



##### Theorem 1

$b^\prime$ is strictly  increasing which implies that it has an inverse function

(note)

- We will not consider $\theta$'s that have a distribution concentrated at one point
  - meaning the variance is 0 and $\theta$ equals to a certain value

(proof)

By Eq. (11),
$$
\operatorname{Var}\left[y\right] = b^{\prime\prime}(\theta){\phi \over w} \gt 0 \\
\Rightarrow
b^{\prime\prime}(\theta) \gt 0
$$
since 

- it's the variance, and so, it's non-zero
- and we ignore the equal case as mentioned in the note above

Let $\theta _1 \lt \theta _2$.

- By the "mean value theorem", there exists $c \in (\theta _1, \theta _2)$ s.t. $b^{\prime\prime} (c) = {b^\prime(\theta _2) - b^\prime(\theta _1) \over \theta _2 - \theta _1} \gt 0$.
- Thus, $b^\prime(\theta _2) \gt b^\prime(\theta _1)$.

Therefore $b^\prime$ is strictly increasing.

Q.E.D.



##### Convex function

(Definition)

A differentiable function of one variable, $f$, is convex if and only if the graph lies above all of it's tangent line.
$$
f(x) \ge f^{\prime}(x_0)(x - x_0) + f(x_0)
$$

##### Theorem 2

$b$ is strictly convex

(Proof)

By theorem 1, $b^\prime$ is strictly increasing.

Let $\theta _1 \lt \theta _2$,

- then by the mean value theorem, there exists $c \in (\theta _1, \theta _2)$ s.t. $b^{\prime} (c) = {b(\theta _2) - b(\theta _1) \over \theta _2 - \theta _1} \gt 0$

$$
\Rightarrow 
b(\theta_2) = b^\prime(c)(\theta_2 - \theta_1) + b(\theta_1) \\
\gt b^\prime(\theta_1)(\theta_2 - \theta_1) + b(\theta_1) \\
$$

which means $b(\theta _2)$ is strictly greater than the tangent line at $\theta _1$.

Since $\theta _1$ and $\theta _2$ are arbitrary, the theorem follows.

Q.E.D



##### (Summary)

Properties of the log partition function $b$

- $b$ is strictly convex
- $b^\prime$ is strictly increasing
- $b^{\prime\prime}$ is strictly positive



## Likelihood, score, and fisher information

##### Density

$$
f(y_i ; \theta, \phi) = \exp({y_i\theta_i - b(\theta_i) \over \phi/w} - c(y_i, \phi))
\tag{14}
$$

##### Joint density / likelihood

TODO

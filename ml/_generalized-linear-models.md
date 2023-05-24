# Generalized Linear Models

- https://youtube.com/playlist?list=PLmM_3MA2HWpYoG_q69ZzbCSwvriLuimoO

## Background

### Overdispersed exponential family

$$
f(y | \theta, \phi) = \exp({y\theta - b(\theta) \over \phi/w} - c(y, \phi))
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

(1)
$$
\int _{-\infty}^\infty
f(y|\theta, \phi) \mathrm{d}y = 1
$$
(2)
$$
{\partial^k \over \partial \theta^k} \int_{-\infty}^{\infty} f(y|\theta, \phi) \mathrm{d}y
= \int_{-\infty}^{\infty} {\partial^k \over \partial \theta^k} f(y|\theta, \phi) \mathrm{d}y
= 0 \\

\Rightarrow

\int_{-\infty}^{\infty} f^\prime (y|\theta, \phi) \mathrm{d}y
= 0, \\

\Rightarrow

\int_{-\infty}^{\infty} f^{\prime\prime} (y|\theta, \phi) \mathrm{d}y
= 0
$$
where

- $k$
  - number of differentiation

(3)


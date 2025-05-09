## Probability notations

- $(\Omega, \mathcal{F}, P)$
  - Probability Space
  - $\Omega$
    - Sample space
  - $\mathcal{F}$
    - Set of events (or Event space)
    - $\sigma$-field
  - $P$
    - $\mathcal{F} \to [0, 1]$
  - Family of Borel sets $\mathcal{F} = \mathcal{B}(\mathbb{R})$ is a σ-field on $\mathbb{R}$
  
- $A$, $B$, $C$, ...
  - events
- $\xi$ (or $X$)
  - random variable
  - $\xi: \Omega \to \mathbb{R}$
  - $\{\xi \in \mathcal{B}\} \triangleq \{\omega \in \Omega: \xi(\omega) \in \mathcal{B}\} = \xi^{-1}(\mathcal{B})$
- $x$
  - particular realization of random variable
- $P(A)$
  - probability of an event
- $F(x)$ or $F_X(x)$ or $F_\xi$
  - CDF
  - distribution function
  - non decreasing
  - right continous
  - (for discrete random variable)
    - $F(x) = P(X \le x)=\sum _{x_{i}\leq x}\operatorname {P} (X=x_{i})=\sum _{x_{i}\leq x}p(x_{i})$
  - (for continuous random variable)
    - $F(x) = P(X \le x)=\sum _{x_{i}\leq x}\operatorname {P} (X=x_{i})=\int_{-\infty}^x f(t)\,dt$
- joint cumulative distribution function
- marginal cumulative distribution function
- $p(x)$
  - PMF
  - $P(X=x) = p(x)$
- $f(x)$ or $f_X(x)$
  - PDF
  - $P(a \le X \le b) = \int_a^bf(x)\,dx$
  - PDF may not exist, if CDF is not differentiable
- $E[X]$
  - $\sum xP(X=x)$
  - expectation
  - (for discrete random variable)
    - $\sum xp(x)$
  - (for continuous random variable)
    - $\int_{-\infty}^{\infty}xf(x)\,dx$
- Covariance of X and Y
  - $\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$
  - if $\text{Cov}(X, Y) = 0$ then $X$ and $Y$ are uncorrelated
    - $X$ and $Y$ are independent $\to$ $X$ and $Y$ are uncorrelated

## Formulas

### Law of total expectation
- the law of iterated expectations
- Adam's law

$$
E[X]=E[E[X∣Y]]=E_Y[E_{X∣Y=y}[X∣Y]]
$$

- https://en.wikipedia.org/wiki/Law_of_total_expectation
- https://juandiegodc.com/src/content/adam-n-eve

### Law of total variance
- variance decomposition formula
- law of iterated variances
- Eve's law

$$
Var[X]=E[Var[X∣Y]] + Var[E[X∣Y]]
$$

- https://en.wikipedia.org/wiki/Law_of_total_variance
- https://juandiegodc.com/src/content/adam-n-eve

## conditional probability

## conditional expectation

### P(a|b) P(a;b) 는 어떤 차이인가

f(x;θ) is the density of the random variable X at the point x, with θ being the parameter of the distribution. f(x,θ) is the joint density of X and Θ at the point (x,θ) and only makes sense if Θ is a random variable. f(x|θ) is the conditional distribution of X given Θ, and again, only makes sense if Θ is a random variable. This will become much clearer when you get further into the book and look at Bayesian analysis.

--------
확률은 집합에서 정의된다
확률변수가나온경우 그것에 대응하는 집합에 대한 확률을 의미한다
컨디션 값은 항상 정해진 사건이다


## Continuous probability distributions

### Beta distribution

- special case of Dirichlet distribution
- $P(X) \geq 0$  where $x$ is in $[0,1]$
- $\alpha$
  - ...
- $\beta$
  - ...

### Gamma distribution

- sum of exponential distribution
- X, Y ~ Gamma and W = X / X + Y, T = X + Y, then W ~ Beta, T ~ Gamma and W and T are independent

### Dirichlet distribution

### Normal distribution

### Multivariate normal

PDF

$$
{\displaystyle f_{\mathbf {X} }(x_{1},\ldots ,x_{k})={\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}
$$

Conditional distribution

$$
(\boldsymbol{x_1} | \boldsymbol{x_2} = \boldsymbol{a}) \sim N(\bar{\boldsymbol{\mu}}, \bar{\boldsymbol\Sigma})
$$

$$
\bar{\boldsymbol\mu} =
\boldsymbol\mu_1 + \boldsymbol\Sigma_{12} \boldsymbol\Sigma_{22}^{-1}
\left(
 \mathbf{a} - \boldsymbol\mu_2
\right)
$$

$$
{\overline {\boldsymbol {\Sigma }}}={\boldsymbol {\Sigma }}_{11}-{\boldsymbol {\Sigma }}_{12}{\boldsymbol {\Sigma }}_{22}^{-1}{\boldsymbol {\Sigma }}_{21}
$$

Marginal distribution

- just drop the corresponding terms from $\mathbb{\mu}$ and $\Sigma$

The PDF maximum for multivariate normal distributions decreases as the dimension increases

$X \sim \mathcal{N}(\mathbb{0}_d, I_d)$


| d | $f_X(\mathbb{0}_d)$ |
|---|---------------------|
| 1 | 0.398942280         |
| 2 | 0.159154943         |
| 3 | 0.063493636         |
| 4 | 0.025330296         |
| 5 | 0.010105326         |
| 6 | 0.004031442         |
| 7 | 0.001608313         |
| 8 | 0.000641624         |
| 9 | 0.000255971         |

2-dimensional multivariate normal distribution 에서 단면에 상수를 곱하면 conditional distribution이 나오는지..?

- 안됨
- sigma != 1 인 일반적인 multivariate 에 대해서 그 단면과 conditional distribution 과 marginal distribution 은 다 다르며 비례 관계가 없음.

### Student's t-distribution

### F-distribution

### $\chi^2$-distribution

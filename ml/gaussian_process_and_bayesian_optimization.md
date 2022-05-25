# Gaussian process and Bayesian optimization

## TODO

- fix negative log likelihood in the code

## Questions
- Why is it Bayesian?
  - what is objective
    - expansive to calculate
    - actual training for example
  - surrogate function?
    - we assumes we know the results from all the point
  - acquisition function
    - tells us which point is good to explore next time
- how to make the covariance matrix psd/sd??
  - kernel can be either positive semidefnite or positive definite
  - covariance matrix is at least positive semidefinite
- Why linear kernel ends up with linear mean function? with variance 0
  - there is no covariance??
  - there is varaince??
- pros and cons

## answered

- What is the meaning of clicking an observed data point in the blog?
- What if margin is not there? in MPI
- what if covariance is greater than 1
  - the target value is likely to be similar
- what if covariance is less than 0
  - the target value is likely to be oposite acrossing the mean line
- How to center the random process to have zero mean?
  - Maybe we can just standardize the observed data
  - including not only the target values but also dependent variable values
- What if we have many hyperparameters to optimize?
  - Is it related to the fact that kernel is defined as $R^n \times  R^n \to R$

## Multivariate normal distributions

- a.k.a. Multivariate Gaussian distribution
- marginal distribution is normal distribution
- conditional distribution is normal distribution

## Gaussian process

- $T$
  - index set
  - $t_1, t_2, \cdots, t_k$
- $X$
  - $X_{t_1}, X{t_2}, \cdots, X{t_k}$
  - A stochastic process indexed by $t \in T$
  - Usually the mean is centered to be zero.
  - $\sim N(0,K)$
- $K(x, x^\prime)$
  - a covariance function or kernel
  - Inputs $x$ and $x^\prime$ are going to be two indices $t$ and $t^\prime$
  - types
    - stationarity
      - $X_t$ is not depending on the index $t$
      - but it can depends on $X_{t+3}$ for example.
    - isotropy
    - smoothness
    - periodicity

### Gaussian process regression
- $X_N$ is the observed input data $\{x_1, x_2, \cdots, x_N\}$. This is the index set for the Gaussian processes $Y_N$ and $T_N$.
- $Y_{N}$ is the latent gaussian process $\{y_1, y_2, \cdots, y_N\}$ corresponding to the observed dataset consisting of $X_N$ and $T_N$.
- $T_{N}$ is the gaussian process corresponding to the target data to be realized as
  - Incorporates gausssian noise in the diagonal components of the covariance matrix.
  - $T_N \sim N(0, K_N + (\beta I_N)^{-1})$
  - $t_n = y_n + e_n$
  - $e_n \sim N(0, \beta^{-1})$
- $cov_N$
  - $K + (\beta I)^{-1}$
  - covariance matrix for $T_N$
- $t_{N+1} | T_{N} \sim N(k^t K_N^{-1} T_N,\ c-k^\mathsf{T}K^{-1}k)$

- $K_{N+1}$
$$
K_{N+1} =
\begin{bmatrix}
K_{N} & k\\
k^\mathsf{T} & c
\end{bmatrix}
$$
- $k = \begin{bmatrix}k(x_{N+1}, x_1),\ k(x_{N+1}, x_2), \cdots,\ k(x_{N+1}, x_N)\end{bmatrix}^\mathsf{T}$
- $c = k(x_{N+1}, x_{N+1}) + \beta$
- kernel function can be parameterized as $k(x, x^\prime; \theta)$ and the parameters $\theta$ can be trained - by using the gradient descent method for example.


## Bayesian optimization

### summary

- when a global optimization problem is challenging?
  - black box
  - non-convex
  - non-linear
  - noisy
  - computationally expensive objective function
- objective function
  - input is a sample
  - output i
  - cost returns
- surrogate objective function
  - probablilstic model of objective function
  - Bayesian approximation of the objective function that can be sampled efficiently.
  - P(f|D) = P(D|f) * P(f)
    - where
      - f: an objective function
      - D: {xi, f(xi), … xn, f(xn)}
      - (x1, x2, …, xn): samples
    - posterior = likelihood * prior
  - A regression model which is usually a random forest or a Gaussian Process
- acquisition function
  - Technique by which the posterior is used to select the next sample from the search space.
  - types
    - MPI(Maximum Probability of Improvement)
    - MEI(Maximum Expected Improvement)
    - LCB(Lower Confidence Bound)
- algorithm
  1. Select a sample by optimizing the acquisition function.
  2. Evaluate the sample with the objective function.
  3. Update the data and, in turn, the surrogate Function.
  4. Go to 1.


## References

- https://en.wikipedia.org/wiki/Gaussian_process
- https://youtube.com/playlist?list=PLbhbGI_ppZIRPeAjprW9u9A46IJlGFdLn
- https://distill.pub/2019/visual-exploration-gaussian-processes/
- https://machinelearningmastery.com/what-is-bayesian-optimization/
- https://distill.pub/2020/bayesian-optimization/

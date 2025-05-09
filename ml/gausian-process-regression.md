# Gausian Process Regression

## Questions

- What is the difference between Bayesian regression and Gausian Process Regression?

## TODO

- Derive marginal/conditional distribution of Multivariate Gausian distribution
  - Bishop 2장 PRML
- summarize GP and GPR
- code review
  - http://katbailey.github.io/post/from-both-sides-now-the-math-of-linear-regression/
  - http://katbailey.github.io/post/gaussian-processes-for-dummies/

## Summary

Random Process

- X(t, w)
  - when it draws, it will give function
  - if t is fixed X becomes random variable
  - if w is fixed X becomes deterministic function

Gausian Process

- one of random process

Bayesian linear regression

- final purpose
  - P(t_{n+1}|T_N, t_{n+1}) = P(t_{n+1}|T_{N+1})
- noise ~ N(0, betha^-1 I)
- conjugate distribution
  - The distribution where p(θ|x) and p(θ) in the same probability distribution familiy in Bayesian probability setting, and we call the prior a conjugate prior. For example Gaussian distribution. Posterior distribution can be derived analytically in Bayesian linear regression.

Gausian Process Regression

- Bayesian linear regression with the normal distribution as a prior
- assumption
  - W ~ N(0, alpha^-1 I) as a prior
- then
  - Y ~ N
  - T ~ N
- good for continuous domain
- we define a kernel and our covariance will consist of the kernel
  - kernel
    - defines similarity
    - should be symmetric
    - should be positive semi definite
      - it's difficult to prove if it's PSD
      - but we can derive kernels from the other known kernels
- prior
  - p(W) or p(T)
- posterior
  - p(W|X,Y)
- lazy learning
  - we could say GPR inference will all data points those have been given before.
- provides not only prediction (mean) but also the uncertainty (variance)

https://wiseodd.github.io/techblog/2017/01/01/mle-vs-map/

MLE(Maximum Likelihood Estimation)

- In the problem of finding θ for hypothesis function in ML,
- MLE is a special case of MAP, where the prior P(θ) is uniform!
- MLE tries to find best θ that best explains the training data
- MLE can overfit

MAP(Maximum A Posteriori)

- θ_MAP = argmax_θ P(X|θ)|P(θ)

Hyper-parameter tunning

- kernel hyper parameter
- noise precision

## GPLVM (Gaussian Process Latent Variable Model)

- mapping to higher dimension to lower dimension like PCA

## Variational Inference

## MISC

variation can be considered as precision

PD or PSD consider symmetric matrices only

### kernel trick

- Gausian Kernel = Rasidual Basis Function = Squared Exponential Kernel
- kernel methods
  - SVM
- kernel
  - usually(?) it sends 2 vectors to higher dimension and does inner product.

### Representer theorem

- https://en.wikipedia.org/wiki/Representer_theorem
- Empirical Risk Minimzation
  - https://en.wikipedia.org/wiki/Empirical_risk_minimization
- Reproducing kernel Hilbert space
  - https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space

# Information geometry

applies the techniques of differential geometry to study probability theory and statistics

## Concepts

### missing data imputation using EM/Bayesian formula

https://youtu.be/z4qOY4i2LQY

- selection bias
  - Berkson's bias
  - selective survival bias
  - non-reponse bias
  - incidence-prevalence bias
  - loss-to-follow-up bias
  - volunteer bias
- information bias
  - non differential misclassification
  - differential misclassification
  - reacall bias
  - memory decay bias
  - measurement bias
  - false labeling
  - reverse bias
  - interview
  - publication bias
  - hawthorne effect
  - ascertainement bias
  - time bias
- imputation in statistics = denoising in machine learning
- 대표본 근사??

(Mean score theorem)

Fisher 1922

(Louis' formula)

1982

(Bayesian vs Frequentist)

- Bayesian
  - model
    - f(latent, θ| data)
  - computation
    - Data augmentation
  - prediction
    - I-step
  - parameter update
    - P-step
  - parameter estimation
    - Posterior mode
  - Imputation
    - Multiple imputation
  - Variance estimation
    - Rubin's formula
- Frequentist
  - model
    - f(latent| data, θ)
  - computation
    - EM algorithm
  - prediction
    - E-step
  - parameter update
    - M-step
  - parameter estimation
    - MLE
  - Imputation
    - Fractional imputation
  - Variance estimation
    - Linearization or Resampling

(Small area estimation)

Fay-Herriot model approach
- 1979 Estimates of income for small places: an application of James–Stein procedures to census data
- [Multivariate Fay–Herriot models for small area estimation](https://doi.org/10.1016/j.csda.2015.07.013)

### fisher information matrix

(likelihood definition)

- probability for discrete random variable
- density for continuous random variable

(empirical distribution)

- we can construct CDF as a step function using given examples

(score function)

- function of parameters $\theta$
- gradient of likelihood when only one data example is given.
- mean of score function at the true parameter is zero

(fisher information matrix)

- function of parameters
- covariance matrix of score
  - evaluated at the distribution corresponding to the true parameter
- negative expected hessian of log likelihood
  - defines the curvature of log likelihood
- hessian of KL divergence
  - KL-divergence is not a distance metric because it's not symmetric

(observed fisher information)

The observed information matrix at $\theta^*$ is defined as

$
\mathcal{J}(\theta^*)
= - \left.
  \nabla \nabla^T
  \ell(\theta)
\right|_{\theta=\theta^*}
$

$
=-\left.\left({\begin{array}{cccc}{\tfrac {\partial ^{2}}{\partial \theta _{1}^{2}}}&{\tfrac {\partial ^{2}}{\partial \theta _{1}\partial \theta _{2}}}&\cdots &{\tfrac {\partial ^{2}}{\partial \theta _{1}\partial \theta _{p}}}\\{\tfrac {\partial ^{2}}{\partial \theta _{2}\partial \theta _{1}}}&{\tfrac {\partial ^{2}}{\partial \theta _{2}^{2}}}&\cdots &{\tfrac {\partial ^{2}}{\partial \theta _{2}\partial \theta _{p}}}\\\vdots &\vdots &\ddots &\vdots \\{\tfrac {\partial ^{2}}{\partial \theta _{p}\partial \theta _{1}}}&{\tfrac {\partial ^{2}}{\partial \theta _{p}\partial \theta _{2}}}&\cdots &{\tfrac {\partial ^{2}}{\partial \theta _{p}^{2}}}\\\end{array}}\right)\ell (\theta )\right|_{\theta =\theta ^{*}}
$

- where
  - $\ell (\theta |X_{1},\ldots ,X_{n})=\sum _{i=1}^{n}\log f(X_{i}|\theta )$
  - i.i.d. assumed
- the negative of the second derivative (the Hessian matrix) of the "log-likelihood".
- It's a sample based version of the Fisher information.
- references
  - https://en.wikipedia.org/wiki/Observed_information

$$

$$

(applications)

- natural gradient descent
  - uses inverse of Fisher information matrix
  - limit
    - we should be able to calculate FIM with respect to the current parameter.
    - we cannot calculate FIM even if we assume the distribution
      - even empirical FIM could be calculated properly only when we know the true parameters of the model
      - we want to find the true parameters, since we don't know them.
    - But the approximation of FIM would be still useful.
      - ADAM uses components related to FIM to some extent
    - Even empirical fisher information and it's inverse are difficult to calculate/store when there are many parameters in deep learning
- Cramér–Rao bound
  - A lower bound on the variance of unbiased estimators of a deterministic (fixed, though unknown) parameter,
  - stating that the variance of any such estimator is at least as high as the inverse of the Fisher information.
  - An unbiased estimator which achieves this lower bound is said to be (fully) efficient.

(References)

- [수학의 즐거움 - Statistics and Information Geometry](https://youtube.com/playlist?list=PLMSC8mGmVhvBQ5jZRjzY3KkIEEokLMZhH)
- [The Fisher Information](https://youtu.be/pneluWj-U-o)
- [수학의 즐거움 - 직장인과 문과생을 위한 수학교실](https://youtu.be/4s06EgHHRrA)
- https://www.quora.com/What-is-an-intuitive-explanation-of-Fisher-information/answer/Duane-Rich
- https://wiseodd.github.io/techblog/2018/03/11/fisher-information/
- https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/
- https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound

### Bregman divergence

$$
D_{F}(p,q)=F(p)-F(q)-\langle \nabla F(q),p-q\rangle
$$

- if $F(p)=\sum _{i}p(i)\log p(i)$ it becomes KL-divergence.

(Reference)

- https://en.wikipedia.org/wiki/Bregman_divergence

## References

- https://en.wikipedia.org/wiki/Information_geometry
- https://en.wikipedia.org/wiki/Statistical_manifold

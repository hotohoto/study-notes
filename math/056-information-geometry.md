# Information geometry

## Big picture

(Three layers)

- $(\Omega, \mathcal{F}, P)$
  - the stage where outcomes $\omega$ are drawn
- $L^2(\Omega, P)$
  - the flat Hilbert space of random variables
  - conditional expectation and projection geometry live here
- $\mathcal{P}$
  - the curved manifold of probability distributions
  - Fisher information and KL divergence live here
  - at each point, the tangent-space picture is locally $L^2$-like

(Key distinctions)

- $X : \Omega \to \mathbb{R}$ is a function in $L^2(\Omega, P)$.
- A distribution is not a point of $L^2(\Omega, P)$ as a random variable but a pushforward measure.
- Different random variables can share one distribution.
- Fixing $P$ defines the inner product, norm, distance, and angle in $L^2(\Omega, P)$.
- Covariance, orthogonality, and conditional expectation require a shared $(\Omega, \mathcal{F}, P)$.
- Curvature lives in $\mathcal{P}$, not in $L^2$.

(Tangent-space link)

- At a fixed distribution $P$,

$$
T_P\mathcal{P}
\cong
\{f \in L^2(\Omega, P) : E_P[f] = 0\}.
$$

- Equivalently,

$$
L^2(\Omega, P) = \mathbb{R} \oplus T_P\mathcal{P}.
$$

- $\oplus$
    - direct sum
- $\mathbb{R}$
    - the subspace of constant functions
- $T_P\mathcal{P}$
    - identified with the mean-zero subspace of $L^2(\Omega, P)$
    - $X - E[X]$ lives here
    - score functions live here, so they must have mean zero
    - much of the geometry behind variance, covariance, and Fisher information happens here

## Core ideas

### What a random variable really is

- $X$ is a function, not a density.
- $\omega$ is an outcome.
- Randomness comes from how $\omega$ is sampled under $P$.
- $f_X$ and $X$ are different functions with different domains.
- Same law does not imply same random variable: $X$ and $1-X$ may be distributionally identical but distinct on $\Omega$.
- A point in $L^2(\Omega, P)$ is an equivalence class modulo almost-sure equality.

### Expectation as an inner product

Expectation defines the inner product

$$
\langle X, Y \rangle = \mathbb{E}[XY].
$$

Properties:

- linearity
- symmetry
- positive-definiteness modulo almost-sure equality

Norm:

$$
\|X\| = \sqrt{\mathbb{E}[X^2]}.
$$

Consequences:

- orthogonality: $\mathbb{E}[XY] = 0$
- if both means are zero, orthogonality = uncorrelatedness
- Cauchy-Schwarz:

$$
|\mathbb{E}[XY]| \le \sqrt{\mathbb{E}[X^2] \mathbb{E}[Y^2]}.
$$

- correlation coefficient = cosine of the angle between centered variables

### Unification with function spaces

Same $L^2$ structure, different measure:

$$
\langle f, g \rangle = \int f g \, dx
\quad \text{vs.} \quad
\langle X, Y \rangle = \int X Y \, dP.
$$

- Lebesgue measure vs probability measure
- Fourier series vs Karhunen-Loeve expansion
- Parseval identities vs variance decomposition

### Conditional expectation basics

Conditional expectation $\mathbb{E}[X \mid Y]$ is the mean of $X$ given $Y$, hence a function of $Y$.

The tower property says:

$$
\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X].
$$

Interpretation:

- averaging conditional means recovers the global mean
- with nested sigma-algebras, repeated conditioning keeps the coarser information

Its geometric interpretation is:

$$
\mathbb{E}[X \mid Y]
= \operatorname{Proj}_{L^2(\sigma(Y))}(X).
$$

$\mathbb{E}[X \mid Y]$ is the orthogonal projection of $X$ onto the closed subspace of square-integrable $\sigma(Y)$-measurable functions.

Equivalent views:

- least-squares approximation of $X$ by functions of $Y$
- minimizer of

$$
\mathbb{E}[(X - g(Y))^2]
$$

over all admissible $g(Y)$

- residual orthogonality:

$$
X - \mathbb{E}[X \mid Y] \perp h(Y)
$$

for every square-integrable $h(Y)$

Consequences:

- tower property = projection of a projection
- variance decomposition = Pythagorean theorem
- unified view of regression, Kalman filtering, martingales, ML estimators

### Statistical manifold

- A parametric family $p(x \mid \theta)$ can be viewed as a manifold.
- Coordinates: parameters $\theta$.
- Local metric: Fisher information.

### Likelihood and score

- Discrete case: pmf.
- Continuous case: density.
- For i.i.d. samples $X_1, \dots, X_n$,

$$
\ell(\theta \mid X_1, \dots, X_n) = \sum_{i=1}^{n} \log f(X_i \mid \theta).
$$

- score = gradient of log-likelihood
- under regularity conditions, $E_\theta[\nabla_\theta \log p(X \mid \theta)] = 0$, so the score is a mean-zero tangent vector

### Fisher information matrix

Fisher information is the central metric object in information geometry.

- covariance of the score
- negative expected Hessian of log-likelihood
- local quadratic structure of the statistical model
- second-order term of KL divergence near a reference distribution

For one observation,

$$
\mathcal{I}(\theta)
= \mathbb{E}_{X \sim p(x \mid \theta)}
\left[
\nabla_\theta \log p(X \mid \theta)
\nabla_\theta \log p(X \mid \theta)^T
\right].
$$

Under regularity conditions,

$$
\mathcal{I}(\theta)
= - \mathbb{E}_{X \sim p(x \mid \theta)}
\left[
\nabla_\theta^2 \log p(X \mid \theta)
\right].
$$

Uses:

- NGD uses $\mathcal{I}(\theta)^{-1}$ instead of the Euclidean metric
- Cramér-Rao: variance lower bound for unbiased estimators
- large Fisher information = parameters are easier to estimate
- geometrically: Fisher is the Riemannian metric on $\mathcal{P}$
- computation happens in coordinates; meaning lives on the manifold
- tangent-space view: Fisher gives the local inner product

### Observed Fisher information

Observed information = sample analogue of Fisher information:

$$
\mathcal{J}(\theta^*)
= - \left.
\nabla \nabla^T \ell(\theta)
\right|_{\theta = \theta^*}.
$$

- negative Hessian of log-likelihood at a chosen parameter
- sample-dependent, unlike expected Fisher information
- used for asymptotic variance estimates and standard errors

Reference:

- https://en.wikipedia.org/wiki/Observed_information

Limitations:

- exact computation is often expensive
- storage/inversion is often intractable in deep learning
- practical methods use diagonal, blockwise, or empirical approximations

### Missing data, EM, and Bayesian views

Missing-data problems are a good comparison point for frequentist and Bayesian views.

- Frequentist view
  - Model: $f(\text{latent} \mid \text{data}, \theta)$
  - Computation: EM algorithm
  - Prediction/update cycle: E-step and M-step
  - Parameter estimate: MLE
  - Imputation: fractional imputation or model-based completion
  - Variance estimation: linearization or resampling
- Bayesian view
  - Model: $f(\text{latent}, \theta \mid \text{data})$
  - Computation: data augmentation
  - Prediction/update cycle: imputation step and posterior step
  - Parameter estimate: posterior mean or posterior mode
  - Imputation: multiple imputation
  - Variance estimation: Rubin's rules

Intuition:

- imputation in statistics is analogous to denoising / latent reconstruction in ML
- EM alternates latent estimation and parameter update
- Bayesian methods keep uncertainty in both latent variables and parameters

Biases that motivate careful missing-data modeling:

- Selection bias
  - Berkson's bias
  - selective survival bias
  - non-response bias
  - incidence-prevalence bias
  - loss-to-follow-up bias
  - volunteer bias
- Information bias
  - non-differential misclassification
  - differential misclassification
  - recall bias
  - memory decay bias
  - measurement bias
  - false labeling
  - reverse causation bias
  - interviewer bias
  - publication bias
  - Hawthorne effect
  - ascertainment bias
  - time-window bias

Related references:

- Fisher (1922)
- Louis' formula (1982)
- https://youtu.be/z4qOY4i2LQY

### EM as alternating projection

EM can be viewed as alternating projection under KL-type geometry.

- Data manifold $D$: joint distributions consistent with the observed data $x$
- Model manifold $M$: distributions of the form $p_\theta(x, z)$

EM alternates between them:

- E-step: move toward $D$ via the posterior over latent variables
- M-step: move back to $M$ by maximizing over $\theta$
- explains monotone improvement and KL-Pythagorean convergence structure

### Submanifolds

- A submanifold is a lower-dimensional manifold inside a larger one.
- Examples:
  - the sphere inside $\mathbb{R}^3$
  - the Gaussian family inside the space of all distributions
  - the set of distributions representable by a neural network
- each point has its own tangent space
- projection, orthogonality, and NGD remain meaningful on restricted model families
- neural-network training can be viewed as search on a structured submanifold of distribution space

### Small area estimation

Fay-Herriot is a classic small-area-estimation example with hierarchical shrinkage.

- 1979: Estimates of income for small places: an application of James-Stein procedures to census data
- [Multivariate Fay-Herriot models for small area estimation](https://doi.org/10.1016/j.csda.2015.07.013)

### Bregman divergence

$$
D_F(p, q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle
$$

- general template for many discrepancy measures
- if $F(p) = \sum_i p(i) \log p(i)$, it yields KL divergence
- KL divergence is not a metric: not symmetric

## Resources

- [수학의 즐거움 - Statistics and Information Geometry](https://youtube.com/playlist?list=PLMSC8mGmVhvBQ5jZRjzY3KkIEEokLMZhH)
- [The Fisher Information](https://youtu.be/pneluWj-U-o)
- [수학의 즐거움 - 직장인과 문과생을 위한 수학교실](https://youtu.be/4s06EgHHRrA)
- [Intuitive explanation of Fisher information](https://www.quora.com/What-is-an-intuitive-explanation-of-Fisher-information/answer/Duane-Rich)
- [Fisher information blog post](https://wiseodd.github.io/techblog/2018/03/11/fisher-information/)
- [Natural gradient blog post](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/)
- [Cramér-Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound)
- [Video tutorial](https://t.co/ONvjDsH8Ag)
- [Elementary tutorial](https://t.co/cIA6FAsF4p)
- [Short overview](https://t.co/kvzw9awIOe)
- [Hands-on coding with geomstats](https://t.co/ITfDzdyo5h)
- [Information geometry portal](https://franknielsen.github.io/IG/)

## References

- https://en.wikipedia.org/wiki/Information_geometry
- https://en.wikipedia.org/wiki/Statistical_manifold
- https://en.wikipedia.org/wiki/Bregman_divergence

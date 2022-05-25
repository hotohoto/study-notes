# PRML

## Personal Questions

TODO

(1.1)

- how to interpret over-fitting from the perspective of MLE
- By adopting a Bayesian approach, the over-fitting problem can be avoided. ??
  - We shall see that there is no difficulty from a Bayesian perspective in employing models for which the number of parameters greatly exceeds the number of data points.
  - Indeed, in a Bayesian model the effective number of parameters adapts automatically to the size of the data set.

## 🐢 1 Introduction

- generalization
  - The ability to categorize correctly new examples that differ from those used for training
- unsupervised learning
  - clustering
  - density estimation
  - dimension reduction
    - useful for visualization
- useful studies
  - probability theory
  - decision theory
  - information theory

### 🐢 1.1 Example: Polynomial Curve Fitting

- RMS error
  - $E_\text{RMS} = \sqrt{2 E(w^*) / N} = \sqrt{ \sum\limits_{n=1}^N 2 \times {1 \over 2N} (y(x_n, w^*) - t_n)^2 }$
- shrinkage methods (in statistics literatures)
  - weight decay (in the context of neural nets)
  - examples
    - ridge regression
    - lasso
- training data
  - training set
    - used to determine weights
  - validation set
    - also called a hold-out set
    - used to determine hyperparameters

### 🐢 1.2 Probability Theory

- random variable
- sum rule
- product rule
- joint probability
- marginal probability
- conditional probability
- Bayes' theorem
  - prior probability
  - posterior probability
- independent random variables

#### 🐢 1.2.1 Probability densities

- probability density
  - can be seen as derivative of cumulative distribution function
- change of variable of probability density function using the Jacobian factor
- cumulative distribution function
- joint probability density

#### 🐢 1.2.2 Expectations and covariances

- expectation
- conditional expectation
- variance
- covariance
  - of 2 random variables
- covariance matrix
  - of 2 random vectors

#### 🐢 1.2.3 Bayesian probabilities

- frequentist
  - considers
    - model parameters as fixed
      - MLE finds the best parameters(estimates) as a fixed value
    - there are multiple datasets
      - bootstrap is used to obtain error bars on the estimates
        - by considering the distribution of possible data sets
- Bayesian
  - considers
    - model parameters as a random variable
      - there is uncertainty
    - there is a single observed dataset
  - pros
    - we can be less extreme when number of samples are small
  - cons
    - not effective if a prior is poorly chosen
      - usually just for convenience
    - the evidence term is difficult to calculate
      - MCMC and high computing power addressed this
        - and opened the door to Bayesian methods
  - MCMC
    - very flexible
      - can be applied to a wide range of models
    - but computationally intensive
    - alternatives
      - variational Bayes
      - expectation propagation

#### 🐢 1.2.4 The Gaussian distribution

- Under the normal distribution assumption, MLE will fit like the below.
  - $\mu_\text{ML} = {1 \over N}\sum\limits_{n=1}^{N}x_n$
  - $\sigma_\text{ML} = {1 \over N}\sum\limits_{n=1}^{N}(x_n - \mu_\text{ML})^2$
    - This is the biased sample variance that underestimates the variance.
    - N may not be a big problem when it's a large number.
    - This will lead to overfitting since it's not fitting with respect to the true $\mu$

#### 🐢 1.2.5 Curve fitting re-visited

- $p(\mathbf{w}|\mathsf{x}, \mathsf{t}, \alpha, \beta) \propto p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta)p(\mathbf{w}|\alpha)$

- $
\argmax\limits_{w} p(\mathbf{w}|\mathsf{x}, \mathsf{t}, \alpha, \beta) \\
= \argmax\limits_{w} p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta)p(\mathbf{w}|\alpha) \\
= \argmax\limits_{w} [\log p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta) + \log p(\mathbf{w}|\alpha)] \\
= \argmin\limits_{w} [ - \log p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta) - \log p(\mathbf{w}|\alpha)] \\
= \argmin\limits_{w} [ {\beta \over 2} \sum\limits_{n=1}^{N} \{y(x_n, \mathbf{w}) - t_n\}^2 + {\alpha \over 2} \mathbf{w}^T\mathbf{w}] \\
= \argmin\limits_{w} [ \sum\limits_{n=1}^{N} \{y(x_n, \mathbf{w}) - t_n\}^2 + \lambda  \mathbf{w}^T\mathbf{w}] \\
$
- assuming
  - $p(t|x, \mathbf{w}, \beta) = \mathcal{N}(t|y(x|\mathbf{w}), \beta^{-1})$
  - $p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta) = \prod\limits_{n=1}^N \mathcal{N}(t_n|y(x_n|\mathbf{w}), \beta^{-1}) = \prod\limits_{n=1}^N ({\beta \over 2\pi})^{1/2}\exp(\beta(t_n - y(x_n|\mathbf{w}))^2)$
    - $\because$ i.i.d.
  - $p(\mathbf{w}| \alpha) = \mathcal{N}(\mathbf{w}| \mathbf{0}, \alpha^{-1}\mathbf{I}) = ({\alpha \over 2 \pi})^{(M+1)/2} \exp {\{- {\alpha \over 2} \mathbf{w}^T\mathbf{w}\}}$
- where
  - the training data is $\{\mathsf{x}, \mathsf{t}\}$
    - $\mathsf{x} = (x_1, ..., x_N)^T$
    - $\mathsf{t} = (t_1, ..., t_N)^T$
  - $\alpha$: precision of the prior distribution of $\mathbf{w}$
  - $\beta$: precision of the distribution of the target $t$
  - $M$: the order of the polynomial hypothesis function
  - $\lambda = \alpha /\beta$

#### 🐇 1.2.6 Bayesian curve fitting

$$
p(t|x, \mathsf{x}, \mathsf{t}) = \int p(t|x, \mathsf{w}) p(\mathbf{w}|\mathsf{x},\mathsf{t}) d\mathbf{w} \\
= \int
  p(t|x, \mathsf{w}, \beta) \
  {
    p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta)p(\mathbf{w}|\alpha)
    \over
    p(\mathsf{t}|\mathsf{x}, \beta)
  }
  d\mathbf{w} \\
= \int {
  p(t|x, \mathsf{w}, \beta) p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta)p(\mathbf{w}|\alpha)
  \over
  \int p(\mathsf{t}|\mathsf{x}, \mathbf{w}, \beta)p(\mathbf{w}|\alpha)d\mathbf{w}
} d\mathbf{w} \\
= \int {
  \mathcal{N}(t|y(x|\mathbf{w}), \beta^{-1}) \prod\limits_{n=1}^N \mathcal{N}(t_n|y(x_n|\mathbf{w}), \beta^{-1})\mathcal{N}(\mathbf{w}|\mathbf{0}, \alpha^{-1}\mathbf{I})
  \over
  \int \prod\limits_{n=1}^N \mathcal{N}(t_n|y(x_n|\mathbf{w}), \beta^{-1})\mathcal{N}(\mathbf{w}|\mathbf{0}, \alpha^{-1}\mathbf{I})d\mathbf{w}
} d\mathbf{w}

$$

TODO
- 그냥 수식적는 걸로 정리하고 TODO 남겨 놓고 chapter 3 까지 읽은 다음 다시 읽기
- evidence term 은 joint distribution 의 marginalization 으로 표현 가능
- conjugate prior table 에서 likelihood 가 normal 이고 prior가 normal이면 posterior도 normal
  - 그렇다고 joint distribution 도 nromal 이라고 할수 있나? yes
- $p(\mathbf{w}|\mathsf{x}, \mathsf{t}, \alpha, \beta)$ 를 먼저 구해야 계산이 깔끔해질듯..
-

---

🐢 1.3 Model Selection
🐢 1.4 The Curse of Dimensionality
🐢 1.5 Decision Theory
🐢 1.5.1 Minimizing the misclassification rate
🐢 1.5.2 Minimizing the expected loss
🚀 1.5.3 The reject option
🐇 1.5.4 Inference and decision
🐢 1.5.5 Loss functions for regression
🐢 1.6 Information Theory
🐢 1.6.1 Relative entropy and mutual information

🐢 2 Probability Distributions
🐢 2.1 Binary Variables
🐢 2.1.1 The beta distribution
🐢 2.2 Multinomial Variables
🐢 2.2.1 The Dirichlet distribution
🐢 2.3 The Gaussian Distribution
🐢 2.3.1 Conditional Gaussian distributions
🐢 2.3.2 Marginal Gaussian distributions
🐢 2.3.3 Bayes’ theorem for Gaussian variables
🐢 2.3.4 Maximum likelihood for the Gaussian
🐇 2.3.5 Sequential estimation
🐇 2.3.6 Bayesian inference for the Gaussian
🐇 2.3.7 Student’s t-distribution
🚀 2.3.8 Periodic variables
🐢 2.3.9 Mixtures of Gaussians
🐢 2.4 The Exponential Family
🐢 2.4.1 Maximum likelihood and sufficient statistics
🐇 2.4.2 Conjugate priors
🐇 2.4.3 Noninformative priors
🐢 2.5 Nonparametric Methods
🐢 2.5.1 Kernel density estimators
🐢 2.5.2 Nearest-neighbour methods

🐢 3 Linear Models for Regression
🐢 3.1 Linear Basis Function Models
🐢 3.1.1 Maximum likelihood and least squares
🚀 3.1.2 Geometry of least squares
🐇 3.1.3 Sequential learning
🐇 3.1.4 Regularized least squares
🚀 3.1.5 Multiple outputs
🐢 3.2 The Bias-Variance Decomposition

---

🐇 3.3 Bayesian Linear Regression
🐇 3.3.1 Parameter distribution
🐇 3.3.2 Predictive distribution
🐇 3.3.3 Equivalent kernel

---

🚀 3.4 Bayesian Model Comparison
🚀 3.5 The Evidence Approximation
🚀 3.5.1 Evaluation of the evidence function
🚀 3.5.2 Maximizing the evidence function
🚀 3.5.3 Effective number of parameters
🐇 3.6 Limitations of Fixed Basis Functions

🐢 4 Linear Models for Regression
🐢 4.1 Discriminant Functions
🐢 4.1.1 Two classes
🐢 4.1.2 Multiple classes
🐢 4.1.3 Lest squares for classification
🐢 4.1.4 Fisher’s linear discriminant
🐢 4.1.5 Relation to least squares
🐢 4.1.6 Fisher’s discriminant for multiple classes
🐢 4.1.7 The perceptron algorithm
🐢 4.2 Probabilistic Generative Models
🐢 4.2.1 Continuous inputs
🐢 4.2.2 Maximum likelihood solution
🐢 4.2.3 Discrete features
🐢 4.2.4 Exponential family
🐢 4.3 Probabilistic Discriminant Models
🐢 4.3.1 Fixed basis functions
🐢 4.3.2 Logistic regression
🐢 4.3.3 Interative reweighted least squares
🐢 4.3.4 Multiclass logistic regression
🚀 4.3.5 Probit regression
🚀 4.3.6 Canonical link functions
🐇 4.4 The Laplace Approximation
🐇 4.4.1 Model comparison and BIC
🐇 4.5 Bayesian Logistic Regression
🐇 4.5.1 Laplace approximation
🐇 4.5.2 Predictive distribution

🐢 5 Neural Networks
🐢 5.1 Feed-forward Networks Functions
🐢 5.1.1 Weight-space symmetries
🐢 5.2 Network Training
🐢 5.2.1 Parameter optimization
🐢 5.2.2 Local quadratic approximation
🐢 5.2.3 Use of gradient information
🐢 5.2.4 Gradient descent optimization
🐢 5.3 Error Backpropagation
🐢 5.3.1 Evaluation of error-function derivatives
🐢 5.3.2 A simple example
🐢 5.3.3 Efficiency of backpropagation
🚀 5.3.4 The Jacobian matrix
🚀 5.4 The Hessian Matrix
🚀 5.4.1 Diagonal approximation
🚀 5.4.2 Outer product approximation
🚀 5.4.3 Inverse Hessian
🚀 5.4.4 Finite differences
🚀 5.4.5 Exact evaluation of the Hessian
🚀 5.4.6 Fast multiplication by the Hessian
🐇 5.5 Regularization in Neural Networks
🐇 5.5.1 Consistent Gaussian priors
🐇 5.5.2 Early stopping
🚀 5.5.3 Invariances
🚀 5.5.4 Tangent propagation
🚀 5.5.5 Training with transformed data
🚀 5.5.6 Convolutional networks
🚀 5.5.7 Soft weight sharing
🚀 5.6 Mixture Density Networks
🚀 5.7 Bayesian Neural Networks
🚀 5.7.1 Posterior parameter distribution
🚀 5.7.2 Hyperparameter optimization
🚀 5.7.3 Bayesian neural networks for classification

🐢 6 Kernel Methods
🐢 6.1 Dual Representaions
🐢 6.3 Constructing Kernels
🐢 6.3 Radial Basis Function Networks
🐢 6.3.1 Nadaraya-Watson model
🚀 6.4 Gaussian Processes
🚀 6.4.1 Linear regression revisited
🚀 6.4.2 Gaussian processes for regression
🚀 6.4.3 Learning the hyperparameter
🚀 6.4.4 Automatic relevance determination
🚀 6.4.5 Gaussian processes for classification
🚀 6.4.6 Laplace approximation
🚀 6.4.7 Connection to neural networks

🐢 7 Sparse Kernel Machines
🐢 7.1 Maximum Margin Classifiers
🐢 7.1.1 Overlapping class distributions
🐇 7.1.2 Relation to logistic regression
🐇 7.1.3 Multiclass SVMs
🐇 7.1.4 SVMs for regression
🐇 7.1.5 Computational learning theory
🚀 7.2 Relevance Vector Machines
🚀 7.2.1 RVM for regression
🚀 7.2.2 Analysis of sparsity
🚀 7.2.3 RVM for classification

🐇 8 Graphical Models
🐇 8.1 Bayesian Networks
🐇 8.1.1 Example: Polynomial regression
🐇 8.1.2 Generative models
🐇 8.1.3 Discrete variables
🐇 8.1.4 Linear-Gaussian models
🐇 8.2 Conditional Independence
🐇 8.2.1 Three example graphs
🐇 8.2.2 D-separation
🐇 8.3 Markov Random Fields
🐇 8.3.1 Conditional independence properties
🐇 8.3.2 Factorization properties
🐇 8.3.3 Illustration: Image de-noising
🐇 8.3.4 Relation to directed graphs
🐇 8.4 inference in Graphical Models
🐇 8.4.1 Inference on a chain
🐇 8.4.2 Trees
🐇 8.4.3 Factor graphs
🐇 8.4.4 The sum-product algorithm
🐇 8.4.5 The max-sum algorithm
🚀 8.4.6 Exact inference in general graphs
🚀 8.4.7 Loopy belief propagation
🚀 8.4.8 Learning the graph structure

🐢 9 Mixture Models and EM
🐢 9.1 K-means Clustering
🐢 9.1.1 Image segmentation and compression
🐢 9.2 Mixtures of Gaussians
🐢 9.2.1 Maximum likelihood
🐢 9.2.2 EM for Gaussian mixtures
🐢 9.3 An Alternative View of EM
🐢 9.3.1 Gaussian mixtures revisited
🐢 9.3.2 Relation to K-means
🚀 9.3.3 Mixtures of Bernoulli distributions
🚀 9.3.4 EM for Bayesian linear regression
🐇 9.4 The EM Algorithm in General

🐇 10 Approximate Inference
🐇 10.1 Variational Inference
🐇 10.1.1 Factorized distributions
🐇 10.1.2 Properties of factorized approximations
🐇 10.1.3 Example: The univariate Gaussian
🐇 10.1.4 Model comparison
🐇 10.2 Illustration: Variational Mixture of Gaussians
🐇 10.2.1 Variational distribution
🐇 10.2.2 Variational lower bound
🐇 10.2.3 Predictive density
🚀 10.2.4 Determining the number of components
🚀 10.2.5 Induced factorizations
🚀 10.3 Variational Linear Regression
🚀 10.3.1 Variational distribution
🚀 10.3.2 Predictive distribution
🚀 10.3.3 Lower bound
🚀 10.4 Exponential Family Distributions
🚀 10.4.1 Variational message passing
🚀 10.5 Local Variational Methods
🚀 10.6 Variational Logistic Regression
🚀 10.6.1 Variational posterior distribution
🚀 10.6.2 Optimizing the variational parameters
🚀 10.6.3 Inference of hyperparameters
🚀 10.7 Expectation Propagation
🚀 10.7.1 Example: The clutter problem
🚀 10.7.2 Expectation propagation of graphs

🐇 11 Sampling Methods
🐇 11.1 Basis Sampling Algorithms
🐇 11.1.1 Standard distributions
🐇 11.1.2 Rejection sampling
🚀 11.1.3 Adaptive rejection sampling
🚀 11.1.4 Importance sampling
🚀 11.1.5 Sampling-importance-resampling
🚀 11.1.6 Sampling and EM algorithm
🐇 11.2 Markov Chain Monte Carlo
🐇 11.2.1 Markov chains
🐇 11.2.2 The Metropolis-Hastings algorithm
🐇 11.3 Gibbs Sampling
🚀 11.4 Slice Sampling
🚀 11.5 The Hybrid Monte Carlo Algorithm
🚀 11.5.1 Dynamical systems
🚀 11.5.2 Hybrid Monte Carlo
🚀 11.6 Estimating the Partition Function

🐢 12 Continuous Latent Variables
🐢 12.1 Principal Component Analysis
🐢 12.1.1 Maximum variance formulation
🐢 12.1.2 Minimum-error formulation
🐢 12.1.3 Applications of peA
🐢 12.1.4 PCA for high-dimensional data
🚀 12.2 Probabilistic p e A
🚀 12.2.1 Maximum likelihood peA
🚀 12.2.2 EM algorithm for peA
🚀 12.2.3 Bayesian peA
🚀 12.2.4 Factor analysis
🐇 12.3 Kernel PCA
🚀 12.4 Nonliear Latent Variable Models
🚀 12.4.1 Independent component analysis
🚀 12.4.2 Autoassociative neural networks
🚀 12.4.3 Modelling nonlinear manifolds

🐇 13 Sequential Data
🐇 13.1 Markov Models
🐇 13.2 Hidden Markov Models
🐇 13.2.1 Maximum likelihood for the HMM
🐇 13.2.2 The forward-backward algorithm
🚀 13.2.3 The sum-product algorithm for the HMM
🚀 13.2.4 Scaling factors
🐇 13.2.5 The Viterbi algorithm
🚀 13.2.6 Extensions of the hidden Markov model
🐇 13.3 Linear Dynamical Systems
🐇 13.3.1 Inference in LDS
🐇 13.3.2 Learning in LDS
🚀 13.3.3 Extensions of LDS
🚀 13.3.4 Particle filters

🐇 14 Combining Models
🐇 14.1 Bayesian Model Averaging
🐇 14.2 Committees
🐇 14.3 Boosting
🐇 14.3.1 Minimizing exponential error
🐇 14.3.2 Error functions for boosting
🐢 14.4 Tree-based Models
🚀 14.5 Conditional Mixture Models
🚀 14.5.1 Mixtures of linear regression models
🚀 14.5.2 Mixtures of logistic models
🚀 14.5.3 Mixtures of experts

## References

- https://dominhhai.github.io/en-us/2017/12/ml-prml/

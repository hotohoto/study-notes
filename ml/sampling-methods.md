# Sampling Methods

- It might be difficult to sample computationally for the range of either PMF or PDF is close to 0.

pros

- direct

cons

- in many case it doesn't seem to be possible

## Rejection sampling

- assumptions
  - easy to tell if it's accepted or not
  - difficult to sample y directly from Y somehow
- X ~ target distribution where its PMF or PDF is f(x)
- Y ~ proposal distribution where its PMF or PDF is g(y). $M \cdot g(y)$ envelopes f(y)
- algorithm
  - sample y from Y
  - sample u from Unif(0,1)
  - if $u < {f(y) \over M \cdot g(y)}$
    - if this holds, accept y
    - if not, reject y and return to the first step (don't care)
- M ~ average number of iteration to get 1 sample
- examples
  - Finding conditional probability in a Bayesian network, reject all the samples that doesn't hold the condition when we samples following the topological order.
    - X ~ conditional probability we're finding
    - Y ~ unconditional probability we're sampling on
  - sample dots over square in 2D space and accept samples only in the circle
    - X ~ circle area as PDF
    - Y ~ square area as PDF
- pros
  - i.i.d.
- cons
  - difficult for human to adjust M
  - rejects too much
  - ...(TODO)

### Importance Sampling

- an approximation method to calculate expectation of $f(x)$ according to $P$ where $x \sim P$
  - sample states from a different distribution
  - "importance" seems to mean "weights" for summing up values
- How?
  - where $X:\Omega \to {\mathbb {R}}$ be a random variable in some probability space $(\Omega ,{\mathcal {F}},P)$.
  - $E(f(X); P) = \int f(x)p(x)dx = \int f(x){p(x)\over q(x)}q(x)dx \approx {1 \over n} \sum\limits_i f(x_i){p(x_i)\over q(x_i)}$
    - where $x_i \sim q$
- pros
    - It can lower the variance of the estimation of $E(f(X); P)$
      - it depends on how we pick $q$
    - It can sample states even if $P$ is difficult to sample from
    - It's useful when estimating properties of a particular distribution
- References
  - https://towardsdatascience.com/importance-sampling-introduction-e76b2c32e744

### Annealed Importance Sampling (AIS)

- Motivation
  - The importance sampling is good when the proposal distribution approximates the target distribution well.
  - It's difficult to pick a good proposal distribution.
- Key idea
  - move from a tractable distribution to a distribution of interest via sequence of intermediate distributions
  - alternate between MCMC transitions and importance sampling updates
  - average the final samples with weights
    - weight is accumulated during the process above for each sample
- References
  - [Annealed Importance Sampling](https://arxiv.org/abs/physics/9803008)
  - [Introduction to Annealed Importance Sampling](https://agustinus.kristia.de/techblog/2017/12/23/annealed-importance-sampling/)

## MCMC(Markov Chain Monte Carlo)

- transit into the other assignment with the transition probability table
- some assignment occurs more than the other
- problem of autocorrelated samples is inherent
- MCMC can be a good choice when it's difficult to come up with a proposal distribution of reject sampling or importance sampling.

domains where sampling methods like MCMC is popular:

- machine learning
- bayesian statistics
- bioinformatics
- statistical mechanics

### Gibbs sampling (basic version) (Skip)

- special case of MH algorithm
- sample a feature $x_{t,i}$ one by one rather than sampling all the features $x_t$ together
- set the proposal distribution q from the conditional distribution of p
  - then the detailed balance condition is automatically held
  - needs to analyze probability mass function which we're interested in
  - it's applicable when the joint distribution is not known explicitly or is difficult to sample from directly,
  - but when the conditional distribution of each variable is known and is easy (or at least, easier) to sample from.
- when calculating the conditional probability of p we can consider only its Markov Blanket
- can be used for LDA to calculate expectation phase of EM algorithm
  - assign latent variable by sampling Z which has $\pi(z)$ of a Markov Chain as its probability distribution

- can be used as an optimization method instead of EM algorithm
  - pros
    - sampling methods can get out of saddle points more efficiently than EM algorithm.
      - so some times the model performance is better than EM algorithm
  - cons
    - takes more time to train usually
      - that's why this is not that popular

## Another example (TODO)

Generative vs discriminative
p(x,y) vs p(y|x)

application to bayesian machine learning (TODO)

- example problem and idea
  - mixture model
  - em

Gibbs sampling is particularly well-adapted to sampling the posterior distribution of a Bayesian network, since Bayesian networks are typically specified as a collection of conditional distributions.

## Q&A

### Questions

- Why is gibbs sampling is better than naive sampling if both need to analyze $p(x)$
- What proposal distribution does Gibbs Sampling use?
- about autocorrelation...

### Possibly Answered

- What is the best sampling method for the most problems?
  - Nothing like that, it's case by case
- What is discrete time continuous state markov chain like?
  - if we can define $p_{i,j}$ such that i, j is continuous values it defines NxN matrix that is infinitely large.
- Does gibbs sampling resolve i.i.d. problem of MCMC method to some extent?
  - I don't think so. 2 consequent samples still can be affected by the proposal distribution
- What if MH algorithm found a proposed sample at p(x) = 0.
  - it will be rejected
- How can we tell the transition matrix when we don't know the stationary distribution exactly?
  - Actually the transition matrix can be different from the original.

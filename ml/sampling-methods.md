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

### Importance Sampling (skip)

- estimating properties of a particular distribution
- only having samples generated from a different distribution than the distribution of interest.
- variance reduction technique

- Let X : Ω → R $X:\Omega \to {\mathbb {R}}$ be a random variable in some probability space $(\Omega ,{\mathcal {F}},P)$.
- assumption
  - sampling from P is difficult
  - or we want to lower the variance of the estimation of $E[X;P]$
- accumulate importance or weight from each sample
- replace probability function
- examples
  - Finding conditional probability in a Bayesian network, reject all the samples that doesn't hold the condition when we samples following the topological order.
    - P ~ conditional probability we're finding
    - L ~ unconditional probability we're sampling on

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

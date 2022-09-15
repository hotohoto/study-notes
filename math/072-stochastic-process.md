# Basic Stochastic Processes

## Terminology

- probability 1
  - almost surely
- Banach space
  - a complete normed vector space
  - has a metric
  - length and distance of vectors converge to a well defined limit
- σ-field
  - or σ-algebra
- adapted process
  - a stochastic process that cannot see into the future
  - definition
    - given
      - $(\Omega, \mathcal{F}, P)$
        - a probability space
      - $I$
        - an index set with a total order $le$
      - $\mathbb{F} = (\mathcal{F}_i)_{i \in I}$
        - a filtration of the sigma σ-algebra $\mathcal{F}$
      - $(S, \Sigma)$
        - a measurable space
      - $X: I \times \Omega \to S$
        - a stochastic process
    - the process $X$ is said to be adapted to the filtration $(\mathcal{F}_i)_{i \in I}$
    - if the random variable $X_i: \Omega \to S$ is a $(\mathcal{F}_i, \Sigma)$-measurable function for each $i \in I$
- stopping time
  - also Markov time, Markov moment, optional stopping time, or optional time
- martingale
  - a basic definition
    - $E(|X_n|) < \infty$
    - $E(X_{n+1} | X_1, ..., X_n) = X_n$
  - https://en.wikipedia.org/wiki/Martingale_(probability_theory)
  - local martingale
- Brownian motion
  - or Wiener process
  - standard Brownian motion
    - definition
      - Given a probability space $(\Omega, \mathcal{F}, P)$,
      - a process $W=(W_t)_{t \ge 0}$ is called a standard Brownian motion
      - if
        - it is a Gaussian process
          - i.e. for any $t_1, ..., t_n$, $(W_1, ..., W_n)$ is a Gaussian random vector
        - $E(W_t)=0$, $E(W_s W_t) = min(s,t)$ for all $s, t \ge 0$
    - properties
      - $W_0 = 0$
      - $\forall t \ge s \ge 0$, $W_t - W_s$ is independent from $\sigma(W_u|u \le s)$
      - $\forall h \gt 0$, $(W_{t+h} - W_t)_{t \ge 0}$ is a standard Brownian motion.
        - $W_{t+h} - W_t \sim N(0, h)$
    - has the Markov property
    - a martingale
  - general Brownian motion with drift
    - has the Markov property
    - but not a martingale
- `<x>` quadratic variation
- `<x, y>` cross quadratic variation
- diffusion process
  - a solution to a stochastic differential equation
    - continuous time Markov process with almost sulrely continuous sample paths
    - e.g.
      - Brownian motion
      - reflected Brownian motion
      - Ornstein-Uhlenbeck processes

## 1. Review of Probability

- a probability space $(\Omega, \mathcal{F}, P)$
  - Ω
    - a sample space
    - the set of all possible outcomes
  - $\mathcal{F}$
    - an event space
    - an event is a set of outcomes in the sample space
  - $P$
    - (less informally speaking,) $P$ assigns an event in the event space a probability
    - A probability measure $P$
      - a function
      - $P: \mathcal{F} \to [0, 1]$
      - s.t.
        - $P(\Omega) = 1$
        - if $A_1, A_2, ...$ are pairwise disjoin sets (that is, $A_i \cap A_j = \emptyset$ for $i \neq j$) belonging to $\mathcal{F}$, then $P(A_1 \cup A_2 \cup \cdots) = P(A_1) + P(A_2) + \cdots$
- σ-field $\mathcal{F}$
  - a family of subsets of Ω
    - the empty set $\emptyset$ belongs to $\mathcal{F}$
    - if $A$ belongs to $\mathcal{F}$, then so does the complement $\Omega \setminus A$
    - if $A_1, A_2, ...$ is a sequence of sets in $\mathcal{F}$ then their union $A_1 \cup A_2 \cup ...$ also belongs to $\mathcal{F}$
- borel sets $\mathcal{F}$
  - the smallest σ-field containing all intervals in $\mathbb{R}$
- Lebesques measure
- random variable
- integrable random variable

## 2. conditional expectation

- conditioning on an event
- conditioning on a discrete random variable
- conditioning on an arbitrary random variable
- conditioning on a σ-field
- general properties

## 3. Martingales in Discrete Time

- sample path
- filtration
  - totally ordered collection of subsets
  - a family of σ-algebras that are ordered non-decreasingly
  - definition
    - $(\Omega, \mathcal{A}, P)$
      - a probability space
    - $I$
      - an index set with a tortal order
    - $\mathbb{F}: (\mathcal{F}_i)_{i \in I}$
      - is called a filtration if $\mathcal{F}_k \subseteq \mathcal{F}_l$ for all $k < l$
      - then $(\Omega, \mathcal{A}, \mathbb{F}, P)$ is called a filtered probability space.
- martingale
  - an integrable sequence of random variables
- submartingale
- supermartingale
- stopping time
  - a random variable $\tau$
  - the number of rounds played before quitting the game
- first hitting time

## 4. Maringale inequality and convergence

## 5. Markov Chain

## 6. Stochastic process in continuous time

- Poisson process
- Brownian motion
- Wiener process

## 7. Itô stchastic calculus

## Personal Notes

### Auto correlation

- Auto correlation
  - in [-1, 1]
  - also known as serial correlation
- Partial auto correlation
  - the correlation that results after removing the effect of any correlations due to the terms at shorter lags.
  - useful for selecting `p` for AR/ARIMA models

(Reference)

- https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

### Stationarity

- strictly stationary process (strong stationary process)
- wide-sense stationary process (weak stationary process)
- ergodicity
  - $\lim\limits_{N \to \infty} {1 \over N} \sum\limits^{N} Y_t = E[Y_t]$
  - $\lim\limits_{N \to \infty} {1 \over N} \sum\limits^{N} Y_t Y_{t+k} = E[Y_t Y_{t+k}]$
- non-stationary process

## References


# Markov chain

- stochastic process
  - sequence of random variables evolving over time
  - here random variables are dependent rather than i.i.d
- jumping from state to state at discrete time $n$

(assuming there are discrete and finite many states)

(Homogeneous) Markov property:

$$
P(X_{n+1} = j|X_{n} = i, X_{n-1} = i_{n-1}, X_{n-2} = i_{n-2}, ..., X_0 = i_0)\\ = P(X_{n+1} = j | X_n = i)\\ = q_{i,j}
$$

- Past, future are conditionally independent on present
- Transition matrix $Q$ can comprises $q_{i,j}$
  - each rows sums to 1

### types and properties of Markov chain

- properties
  - Accessible
    - accessible
      - j is accessible from i
    - communicate
      - j is accessible from i and i is accessible from j
  - Reducibility
    - `reducible`
    - `irreducible` (not reducible)
      - every state is communicate from each other
  - Periodicity
    - period k
      - state i has period k if $k = gcd{n > 0: P(X_n = i | X_0 = i) > 0}$
    - `periodic`
      - k >= 2
        - ex) 6, 12, 14, 16, ...
    - `aperiodic` (not periodic)
      - gcd of steps to return to the same state is 1
      - k = 1
        - ex) 1, 7, 8, 12, 13, 17, ...
      - Any state with a self-transition is aperiodic
  - Return Time
    - $RT_i = min\{n > 0: X_n = i | X_0 = i\}$
  - Transience
    - mean recurrence time
      - expected return time
    - `recurrent` state
      - positive recurrent state (non null persistent)
        - has finite mean recurrence time
      - null recurrent state (null persistent)
    - `transient` (not recurrent) state
      - state i is transient if it is possible never return to i again.
  - Ergodicity
    - ergodic state
      - positive recurrent state + aperiodic
    - Markov chain is ergodic if all states are ergodic
- `stationary distribution`
  - States vector $\pi$ is stationary for the chain if $\pi \cdot T = \pi$, where $T$ is the transition matrix of the Markov chain.
  - Questions
    - Does stationary distribution exist?
    - Is it unique?
    - Does chain converge to $S$?
    - How can we compute it?
  - $\pi_i = \lim_{n\to\infty} T_{i,j}^{(n)} = {1 \over E(RT_i)}$
  - can be many stationary distribution for a transition matrix of markov chain
  - $\pi \cdot T = \pi$
    - defined for each states
    - uniquely determined
  - positive stationary distribution
    - $\forall i,\pi_{i}>0$
- for irreducible Markov Chain with finite states
  - A stationary $\pi$ exists
  - $\pi$ is unique
  - $\pi_i = \lim_{n\to\infty} T_{i,j}^{(n)} = {1 \over E(RT_i)}$
  - If $T^m$ is strictly positive for some $m$, then $P(X_n = i)$ converges to $\pi_i$ as $n$ goes to $\infty$.
  - https://www.youtube.com/watch?v=aBGOyZv2pZE
- for irreducible Markov Chain
  - it has positive stationary distribution $\iff$ the markov chain is ergodic $\iff$ all of its states are positive recurrent
  - then $\pi$ is unique
- `detailed balance`
  - $\pi_{i}T_{i,j} = \pi_{j}T_{j,i}$
- `reversible` markov chain
  - if a Markov chain holds detailed balance then it's reversible
  - If a markov chain is reversible with respect to $\pi$, then $\pi$ is stationary.

- Ergodic Theorem for Markov Chains (Simple Version)
  - If $X_0, X_1, ..., X_n$ is an irreducible (time-homogeneous) discrete Markov Chain with stationary distribution $\pi$,
  - then ${1 \over n}\sum_{i=1}^{n}f(X_i) \to E(f(x))$ as $n \to \infty$ where $X \sim \pi$.

## case of random walk on undirected network

- $d_i$ is degree of $i$'th state where the degree is number of edges connected to the state
- then
  - $d_{i} T_{ij} = d_{j} T_{ji}$
  - $\pi_i = {d_i \over \sum_j d_j}$

## case of birth death chain

-

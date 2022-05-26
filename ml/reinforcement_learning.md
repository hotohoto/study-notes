# Reinforcement Learning

- Basically, this document extends https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html.
- For long content, it's encouraged to put it in a new separate file.

## TODO

- read and study evolution strategies again
- in the Pocliy Gradient theorem, what is the difference between $d_{\pi_\theta}(s)$ and $d(s)$

## What is RL
### Key Concepts

- Environment
- Agent
- State (S)
- Action (A)
- episode

(RL algorithms categories)
- model-based
  - Rely on the model of the environment; either the model is known or the algorithm learns it explicitly.
  - It needs to learn the model or the model is given
  - algorithms
    - Alpha Zero
    - MuZero
  - The model approximates the environment state transition and reward
- model-free
  - No dependency on the model during learning.
  - learns the dynamic of the environment (policy and/or state-value function)
  - algorithms
    - Q-learning
    - DQN
- on-policy
  - Use the deterministic outcomes or samples from the target policy to train the algorithm.
- off-policy
  - Training on a distribution of transitions or episodes produced by a different behavior policy rather than that produced by the target policy.

- Model
  - defines
    - Transition probability (P)
      - probability that describes which state will be by an action
    - Reward function (R)
      - an intermediate return
- Acumulated rewards (G)
  - $\gamma$ - discounting factor
- Policy ($\pi$)
  - probability that describes which action to take
  - either deterministic or stochastic
- Value function (V)
  - How good a state is.
  - expected long-term return with discount
  - types
    - Q-value or action value
      - long-term expected return of the current state s, taking action a under policy π
    - V-value or state value
      - long-term expected return of the current state s under policy π
    - A-value or advantage function
      - Q-value - V-value
- Optimal value function
### Markov Decision Processes

- MP
- MRP
- MDP

### Bellman Equations

- Bellman expectation equations
- Bellman optimality equations

## Common Approaches


### Dynamic programming
### Monte-Carlo methods
### Temporal-Difference Learning

- steps
    - starts with random value function
    - finds new value function until reaching optimal value function
  - related to `optimality Bellman operator`
- bootstrapping
  - can introduce deadly triads

(Algorithms)

- Sarsa
  - On-policy TD control - for the next action, uses the same policy including ε-greedy.
  - ε-greedy makes learning slow
- Q-learning
  - uses Q table
  - off-policy TD control - for the next action, uses max Q value
  - ε-greedy does not affect learning process
- DQN
  - uses deep net to represent Q table for huge state action space
  - tricks
    - experience replay
    - reward clipping
    - huber loss
      - ${1 \over 2} x^2$ for $|x| \le \delta$
      - $\delta(|x| - {1 \over 2} \delta)$ otherwise
- Double DQN
  - main network
    - network is trained here
    - the action is decided by this network
  - target network
    - copied and fixed after the last episode ends
    - get q value here
- Dueling DQN
  - divide the network results into $V(s)$ and $A(s, a) = Q(S,a) - V(s)$
  - and, reassemble the two values
  - known as better when the action space is large
- Noisy DQN
- DQN with prioritized experience replay
  - $|R_{t+1} + \gamma \max_a Q_t(S_{t+1}, a) - Q(S_t, a_t)|$ is the prioritization criterion

### Combinining TD and MC learning

- n-step TD learning (?)
### Policy Gradient

- policy iteration
  - steps
    - starts with an arbitrary policy
    - finds value function for the policy
    - updates policy
  - related to `Bellman operator`
  - algorithms
    - Policy gradient
      - REINFORCE
        - aproximated policy gradient
      - actor-critic

(Actor-critic)

- Actor
  - Outputs policy
- Crictic
  - Outputs values
- types
  - type#1
  - type#2
- algorithms
  - A3C(Asynchronous Advantage Actor-Critic)
    - runs multiple agents updating the shared network asynchronously and periodically
  - A2C(Advantage Actor-Critic)

### Evolution Strategies

## Known problems

### Exploration-exploitation dilemma

- ε-greedy
- ES - controlled by policy parameter perturbation

### Deadly triad issue

## Case studies

### AlphaGo Zero

### Alpha Zero
  - https://www.goratings.org/en/
  - https://en.wikipedia.org/wiki/Go_ranks_and_ratings#Elo_ratings_as_used_in_Go

### MuZero
- cnn
  - image input => hidden states
- rnn
  - previous hidden states + hypothetical next action => policy, value, immediate reward
## References

- https://medium.com/@parsa_h_m/deep-reinforcement-learning-dqn-double-dqn-dueling-dqn-noisy-dqn-and-dqn-with-prioritized-551f621a9823

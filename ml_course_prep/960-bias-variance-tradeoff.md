# bias variance tradeoff

Q1. What kind of complexity is needed for the given dataset?

Q2. How model complexity depends on its parameters?

Both questions have been studies a lot both theoretically and practically, but still there is no good theory for that.

For Q1 you can check things like:

- representation learning
- dimensionality reduction
- autoencoders
- deep belief networks (and Bayesian approach in general)
- information theory and data compression theory

For Q2:

- VC dim and structural risk minimization
- Occam learning and other similar things
- model pruning & optimization. And the opposite approach: building models from simple to complex, like RF or GBM.
- life-long / online / incremental learning

Both Q1 and Q2 are unsolved problems. Not sure if they are ever solvable. Many topics in Q1 and Q2 are quite math heavy and tend to go deep down: group theory, functional analysis etc.

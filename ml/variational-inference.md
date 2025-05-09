# Variational inference

https://www.jeremyjordan.me/variational-autoencoders

- we want to find the posterior distribution but it's intractable
- so we propose Q approximating it
- $\text{log marginal likelihood} = \text{ELBO(variational free energy)} + D_\text{KL}(q(w\mid\theta) \parallel p(w\mid D)))$
- Minimize KL-divergence
- maximize ELBO
- generalization of EM

## mean field approximation

- assumes all the hidden variables are independent to each other
  - http://www.edwith.org/aiml-adv/lecture/21315


## variational autoencoder

- we want to see the characteristics of z
  - can be used to
    - produce point or interval estimates
    - form predictive density of new data
    - to train a neural net decoder that outputs x from a sample drawn in latent space z
- so we propose Q approximating it

## Questions

- exponential family
  - what is it?
  - what is it for?

## EM algorithm

- Expectation
  - assign latent variable Z given parameters and observation
- Maximization
  - update parameters $\theta$ given observation and Z

## TODO

- https://en.wikipedia.org/wiki/Exponential_family
- https://en.wikipedia.org/wiki/Variational_Bayesian_methods
- http://www.tamarabroderick.com/tutorial_2018_icml.html
  - https://www.youtube.com/watch?v=Moo4-KR5qNg
- (Laplace Method) https://en.wikipedia.org/wiki/Laplace%27s_method
- https://github.com/yaringal/HeteroscedasticDropoutUncertainty

## References

Main Papers

- [G.E. Hinton, Keeping Neural Network simple by minimizing the description length of weights, 1993.](http://www.cs.toronto.edu/~fritz/absps/colt93.pdf)
- [Bishop, Ensemble Learning in Bayesian Neural Networks, 1998.](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ensemble-nato-98.pdf)
- [Alex Graves, Practical Variational Inference for Neural Networks, 2011.](https://www.cs.toronto.edu/~graves/nips_2011.pdf)
- [Charles Blundell et al., Bayes by Backprop, 2015.](https://arxiv.org/abs/1505.05424)
- [Yarin Gal et al., Dropout as a Bayesian Approximation, 2016](https://arxiv.org/pdf/1506.02142.pdf)

Additional Papers

- [Peter Grunwald, A tutorial introduction to the minimum description length principle, 2004](https://arxiv.org/abs/math/0406077)
- [Viet Hung Tran, Copula Variational Bayes inference via information geometry, 2018](https://arxiv.org/abs/1803.10998)
- [Vikram Mullachery et al., Bayesian Neural Network, 2018](https://arxiv.org/abs/1801.07710)

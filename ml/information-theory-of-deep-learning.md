# Stanford seminar - Information Theory of Deep Learning

## Questions

- functional
- Brownian motion


## Summary

https://youtu.be/XL07WEc2TRI

- Rethinking Statistical learning
  - PAC-like generalization bound
  - expressivity
  - hypothesis class
  - input compression bounds
- Information theory
  - large scale learning - typical input patterns
  - huge parameer space - exponentially many optimal solutions
  - Stochastic dynamics of the training process
    - convergence of SGD to locally-Gibbs (Max Entropy) weight distribution
    - the mechanism of representation compression in deep learning
    - convergence times - explains the benefit of the hidden layers

- mutual information
- each layer only loses information (each layer can be considered as a Markov chain.

NN memorizes
NN generalizes
NN over compress

- phase 1
  - NN exploits actual patterns in the data
- phase 2
  - compression
  - NN starts to memorize
  - hidden layers want to forget about the details of input

## References

- First paper on Memorization in DNNs:
  - https://arxiv.org/abs/1611.03530
- A closer look at memorization in Deep Networks:
  - https://arxiv.org/abs/1706.05394 
- Opening the Black Box of Deep Neural Networks via Information:
  - https://arxiv.org/abs/1703.00810 Other links:
  - https://youtu.be/gOn8Po_NPe4
  - https://en.wikipedia.org/wiki/Information_bottleneck_method
- Quanta Magazine blogpost on Tishby's work:
  - https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/
- Tishby's lecture at Stanford:
  - https://youtu.be/XL07WEc2TRI 
- Amazing lecture by Ilya Sutkever at MIT:
  -  https://youtu.be/9EN_HoEk3KY
  -  
  
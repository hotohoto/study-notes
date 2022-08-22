# The Frechet Inception Distance (FID) score

https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch

- The distance between feature vectors calculated for real and generated images
- The inception v3 model is used for image classification
- Developed to evaluate the performance of generative adversarial networks.
- The lower the better.
- Introduced in 2017 by Martin Heusel, et al. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- Suggested as a successor of the inception score

## How to calculate it

- load a pretrained inception v3 model
- remove the original output layer
- scale the real and generated images into 299 × 299 × 3
- The output is taken as the activations from the last pooling layer instead
  - the output layer has 2048 activations
- $\text{FID} = d^2 = ||\mu_1 – \mu_2||^2 + \operatorname{Tr}(C_1 + C_2 – 2\sqrt{C_1 C_2})$
  - $\mu_1$, $\mu_2$
    - the feature-wise mean of the real and generated images
      - e.g. 2048 element vectors where each element is the mean feature observed across the images
  - $C_1$, $C_2$
    - the covariance matrix for the real and generated feature vectors.

## Other similar indices

### Inception score

- The inception v3 model is used for image classification
- measures
  - the quality of generated images
  - the diversity of generated images
- ranges from 1 to the number of classes
- the higher the better

### Wasserstein-2 distance

- FID between two Gaussian distributions


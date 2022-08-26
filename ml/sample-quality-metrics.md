# Sample Quality Metrics

## Inception Score (IS)

$$
{\displaystyle IS(p_{gen},p_{dis}):=\exp \left(\mathbb {E} _{x\sim p_{gen}}\left[D_{KL}\left(p_{dis}(\cdot |x)\|\int p_{dis}(\cdot |x)p_{gen}(x)dx\right)\right]\right)}
$$

- measures how well a generator model captures the full "class" distribution
- an Inception-v3 network pretrained on ImageNet is used as its discriminator
- ranges from 1 to the number of classes
- the higher the better
- (drawback) not capturing diersity within a class
- https://en.wikipedia.org/wiki/Inception_score

## Fréchet Inception Distance (FID)

https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, NIPS, 2017](https://arxiv.org/abs/1706.08500)
- The 2-Wasserstein distance between 2 multivariate normal distributions fit to the sets of feature maps calculated for real and generated images
- The inception v3 model is used to get those feature maps
- calculation
  - load a pretrained Inception V3 model
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
    - NOTE: The formula looks a bit different from the wikipedia 🤔
      - https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance

## Improved Precision and Recall

- NeurIPS 2019
- https://arxiv.org/abs/1904.06991
- Improved Precision
  - measures fidelity
- Improved Recall
  - measures diversity
- TODO

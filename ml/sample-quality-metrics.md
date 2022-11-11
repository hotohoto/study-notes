# Sample Quality Metrics



## Inception Score (IS)

$$
{\displaystyle IS(p_{gen},p_{dis}):=\exp \left(\mathbb {E} _{x\sim p_{gen}}\left[D_{KL}\left(p_{dis}(\cdot |x)\|\int p_{dis}(\cdot |x)p_{gen}(x)dx\right)\right]\right)}
$$

- measures how well a generator model captures the full "class" distribution
- an Inception-v3 network pretrained on ImageNet is used as its discriminator
- ranges from 1 to the number of classes
- the higher the better
- (drawback) not capturing diversity within a class
- https://en.wikipedia.org/wiki/Inception_score



## Fréchet Inception Distance (FID)

https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, NIPS, 2017](https://arxiv.org/abs/1706.08500)
- The 2-Wasserstein distance between 2 multivariate normal distributions fit to the sets of feature maps calculated for real and generated images
- They calculated this between all the training images  and the 50k generated images
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
- definition
  - Precision
    - measures fidelity

  - Recall
    - measures coverage/diversity

- improvement
  - how?
    - embed images into a high-dimensional feature space (the same as before)
      - using a pretrained VGG-16 classifier
        - was better than InceptionV3 in appearance

      - extracting the corresponding activation vectors after the scond fully connected layer

    - take an 50,000 of samples
      - they should be the same number of samples

    - calculate pairwise l2 distances between all feature vectors
    - form a hypersphere for each feature vector
      - with radius equal to its kth nearest neighbor
        - k=3

    - estimate precision
      - by calculating how much percentage of the generated feature vectors belongs to the approximated manifold of the real feature vectors.

    - estimate recall
      - by calculating how much percentage of the real feature vectors belongs to the approximated manifold of the generated feature vectors.


$$
f(\phi, \Phi) =
\begin{cases}
1, & \text{if } \Vert \phi - \phi^\prime \Vert _2 \le \Vert \phi^\prime - \operatorname{NN}_k(\phi^\prime, \Phi)\Vert_2 \text{ for at least one } \phi^\prime \in \Phi \\
0, & \text{otherwise,}
\end{cases}
$$

$$
\operatorname{precision}(\Phi_r, \Phi_g) = {1 \over \vert \Phi_g\vert} \sum\limits_{\phi_g \in \Phi_g} f(\phi_g, \Phi_r)
$$

$$
\operatorname{recall}(\Phi_r, \Phi_g) = {1 \over \vert \Phi_r\vert} \sum\limits_{\phi_r \in \Phi_r} f(\phi_r, \Phi_g)
$$



- Resources

  - https://github.com/kynkaat/improved-precision-and-recall-metric

  - https://github.com/youngjung/improved-precision-and-recall-metric-pytorch



### Realism score R

- evaluates individual samples
- The max operator takes $\phi_r$ after discarding half of the hyperspheres with the largest radii. 
  - when training samples are sparse the hyperspheres may be large, and it may make the score unstable


$$
R(\phi_g, \Phi_r) =
\max\limits_{\phi_r}
\left\{
{
\Vert\phi_r - \operatorname{NN}_k(\phi_r, \Phi_r)\Vert_2
\over
\Vert\phi_g - \phi_r\Vert_2
}
\right\}
$$



## Codelength

- Or, bits per dimension (pixel)
- Motivation
  - "The entropy of a distribution is the expected codelength of an element sampled from that distribution."
  - "If we encode $X$ with the ideal code for $p$, what is our expected code length?"
- Where
  - $\mathbf{x}$
    - an image of $D$ pixels in logit space
  - $\mathbf{z}$
    - an image of $D$ pixels in [0, 256] image space
  - $p(\mathbf{x})$
    - the density in logit space returned by the model
  - $p(\mathbf{z})$
    - the density in [0, 256] image space
  - $\sigma(\cdot)$
    - the logistic sigmoid function
  - $\lambda$
    - $\lambda_\text{CIFAR10} = 10^{-6}$
    - $\lambda_\text{MNIST} = 0.05$
- transformation from $\mathbf{z}$ to $\mathbf{x}$
  - $\mathbf{x} = \operatorname{logit}(\lambda + (1 - 2 \lambda){\mathbf{z} \over 256})$
      - $\operatorname{logit}(p) = \ln {p \over 1 - p}$
- $p_z(\mathbf{z}) = p(\mathbf{x}) \left({1 - 2\lambda \over 256}\right)^D \left(\prod\limits_i \sigma(x_i)(1 - \sigma(x_i))\right)^{-1}$
- $b(\mathbf{x})$
  - the bits per pixel of image $\mathbf{x}$

$$
{\begin{aligned}
b(\mathbf{x}) &= - {\log_2p_z(\mathbf{z}) \over D} \\
&= - {\log p(\mathbf{x}) \over D \log 2} - \log_2(1 - 2 \lambda) + 8 + {1 \over D} \sum\limits_i(\log_2 \sigma(x_i) + \log_2(1 - \sigma(x_i)))
\end{aligned}}
$$

- References
  - [Masked Autoregressive Flow for Density Estimation, p12](https://arxiv.org/abs/1705.07057)
  - https://en.wikipedia.org/wiki/Entropy_(information_theory)
  - https://mlvu.github.io/lectures/31.ProbabilisticModels1.annotated.pdf




## Peak Signal-to-Noise Ratio (PSNR)

- may be used for super resolution task

$$
\text{MSE} = {1 \over MN} \sum\limits_{M, N} [I_1(m, n), I_2(m, n)]^2
$$

$$
\text{PSNR} = 10 \log_{10}{R^2 \over \text{MSE}}
$$

- $R$
  - the maximum color value 255 or 1
- https://m.blog.naver.com/mincheol9166/221771426327



## Structural Similarity Index Map (SSIM)

- e.g. may be used for super resolution task

$$
\text{SSIM}(x,y) = {(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2) \over (\mu_x^2 + \mu_y^2 + C1)(\sigma_x^2 + \sigma_y^2 + C2)}
$$

- $x$
  - an image
- $y$
  - an image
- $C_1 = {0.01 * L}^2$
- $C_2 = {0.03 * L}^2$
- $C_3 = {C_2 / 2}$
- $L$
  - 255 (0~255)
  - 1 (0~1)
- https://m.blog.naver.com/mincheol9166/221771426327



## Learned Perceptual Image Patch Similarity (LPIPS)

- https://arxiv.org/abs/1801.03924
- measures the perceptual similarity between two images




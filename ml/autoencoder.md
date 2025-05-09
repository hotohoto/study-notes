# AutoEncoder

## basic AutoEncoder

- Usually we use the encoder part, and the decoder part is for training
- The main goal is to represent the data in the space of lower dimension.
- Encoding and decoding for feature reduction
  - or higher dimension with sparsity regularization
- As pre-training hidden layers can be trained in the manner of unsupervised learning. Usually, after 2012 AlexNet, we don't do pre-training thanks to ReLU, dropout, max-out, data augmentation and deterministic batch normalization. When samples to train are too small unsupervised methods like autoencoder could be better than supervised methods.
- In some cases, it's related to PCA(Principal Component Analysis).

## de-noising AutoEncoder

- build plain AutoEncode first
- then use data with noise while training decoder part to generate clean data

## k-sparse AutoEncoder

- sparse coding + AutoEncoder
- pooling layer

## convolutional sparse encoder

## variational AutoEncoder

- to use decoder part as a generator
- reparameterization trick
  - z = mu + sigma * err
    - sample err from N(0,1) instead of sampling z from N(mu, sigma)
- usually blurry than GAN
- trying to represent all the training set rather than picking to one style of images to look like real.

compact coding

- complete
  - represent data with a minimal number of units
- overcomplete
  - represent data with a minimal number of active units
  - sparse learning

(references)
- https://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html

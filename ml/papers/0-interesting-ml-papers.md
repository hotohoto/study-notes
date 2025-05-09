# Interesting ML papers

## 2023


## 2022

- GALA: Toward Geometry-and-Lighting-Aware Object Search for Compositing
  - https://arxiv.org/abs/2204.00125
  
  - contrastive learning
    - of a foreground encoder and a background encoder
  
    - between original images and the images the foreground of which are transformed
  
  - can find the foreground objects that is best matching  in terms of geometry and lightning
  
  - alternate training
  
- Dynamic Relation Discovery and Utilization in Multi-Entity Time Series Forecasting
  - https://arxiv.org/abs/2202.10586
  - Microsoft Research Asia, Beijing
  - Attentional multi-graph neural network with automatic graph learning (A2GNN)
    - time-series encoder: LSTM
    - Auto Graph Learner (AGL)
      - Gumbel-softmax
        - to sample all feasible entity pairs in a differentiable way
        - the relation between any entity pair has the chance to be reserved in the learned graph
        - leverage the sparse matrix
      - for inference use the top C edges only
        - C was best around 10~30
    - Attentional Relation Learner
  - (take-aways)
    - Makes use of a predefined graph as inputs
  - (cons)
    - require entire entities as inputs (?)
    - cannot add unseen entities or remove existing entities when it comes to inference(?)
  - (related papers referred to)
    - 2016 Gumbel Softmax
      - Jang, Gu, Poole
    - 2019 Gumbel Graph Network (GGN)
    - 2020 NeuralSparse
      - Zheng et al.

- Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting
  - IJCAI
  - goals
    - Use Transformer for long term forecasting
    - Reduce computational complexity doing that
  - How
    - patch based attention
      - computational complexity
        - (vanila method)
          - $O(N^2)$
          - $N$: number of embeddings
        - (patch based attention)
          - $O(PS^2)$
          - $P$: the number of patches
          - $S$: the size of a patch
        - (patch based attention + downsampling)
          - $O(PS)$
          - introducing pseudo time points
    - triangular stacking
      - downsampling (as mentioned above)
      - fully connected layer
        - inputs: all the layers
    - variable specific modeling
      - weight factorization
        - d x d x N 👉 (d x a) (a x a x N) (a x d)
        - (similar to 1 by 1 convolution)
    - takeaways
      - suggest a decent way to the model complexity of Transformers
    - https://youtu.be/Z4CWwVxKoU0

- Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
  - Nvidia
  - https://arxiv.org/abs/2201.01266
  - Swin U-Net Transformer for segmentation tasks
  - BraTS 2021 segmentation challenge
  - 3D

## 2021

- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
  - ICCV 2021
  - Microsoft Research Asia
  - https://arxiv.org/abs/2103.14030
  - A hierarchical Transformer whose representation is computed with shifted windows
    - limits self-attention computation to non-overlapping local window
    - still allows cross-window connection
  - 2D

| title                                                        | url                              | target dimension | journal or conference |      |
| ------------------------------------------------------------ | -------------------------------- | ---------------- | --------------------- | ---- |
| Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | https://arxiv.org/abs/2103.14030 | 2D               | ICCV 2021             |      |
| UNETR: Transformers for 3D Medical Image Segmentation        | https://arxiv.org/abs/2103.10504 | 3D               | WACV 2022             |      |
| Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation | https://arxiv.org/abs/2105.05537 | 2D               | ECCV 2022 workshops   |      |
| Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images | https://arxiv.org/abs/2201.01266 | 3D               | MICCAI 2021 workshop  |      |



- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  - ICLR 2021
  - Google Research
  - Vision Transformer (ViT)
  - https://arxiv.org/abs/2010.11929
  - Applied transformers to sequence of image patches
  - Pretraining was helpful to beat SOTA convolutional networks

- Emerging Properties in Self-Supervised Vision Transformers
  - DINO, Facebook AI Research
  - https://arxiv.org/abs/2104.14294
  - self distillation with no labels
    - self supervised learning
    - contrastive representation learning
  - student
    - use global and local patches
  - teacher
    - use global patches
    - predictions are to be sharper than those of student
      - by setting a different temperature in the softmax
  - https://youtu.be/h3ij3F3cPIk
  - README

- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
  - [Internal Journal of Forecasting 2021](https://www.sciencedirect.com/science/article/pii/S0169207021000637)
  - https://arxiv.org/abs/1912.09363
  - Oxford, Google Cloud AI
  - quantile regression
    - more stable for targets distributed over a non Gaussian distribution with long tail
    - can forecast quantiles
  - multi-horizon forecasting
    - non iterative
  - model architecture
    - recurrent layers to learn local interactions
    - interpretable self-attention layers to learn long-term interactions
  - incorporates more types of inptus
    - static (time-invariant) covaraites
    - known future inputs
  - variable importance over fields/time
  - regime change analysis
    - distance using Bhattacharyya coefficient
  - interpretability achieved by using an attention layer
  - additional resources
    - https://github.com/google-research/google-research/tree/master/tft
    - https://github.com/jdb78/pytorch-forecasting
      - https://github.com/jdb78/pytorch-forecasting/tree/master/pytorch_forecasting/models/temporal_fusion_transformer
  - TFT, time series forecasting

- TS2Vec: Towards Universal Representation of Time Series
  - https://arxiv.org/abs/2106.10466
  - contrastive representation learning
  - problem settings
    - N: number of time series instances
    - T: sequence length
    - F: number of features for each instance
    - K: dimension of representation vectors
    - B: batch size
  - random cropping on x
      - to achieve position-agnostic representation learning
        - we don't want model to depend on the absolute position
  - encoder
    - FCN: x(i, t) -> z(i, t)
      - single linear layer
    - masking on z
      - a binary mask
      - size is (B, T)
      - the elements of the mask is sampled from a Bernouli distribution with p=0.5
      - 1 means to replace the corresponding value with zero
      - only for training
    - Dilated Convolution
      - 10 residual block with 1d CONV layer
      - z(i, t) -> r(i, t)
  - questions
    - scalability?
  - evaluation datasets
    - 125 datasets from UCR archive
    - 29 datasets from UEA archive
  - additional resources
    - https://github.com/yuezhihan/ts2vec
    - https://youtu.be/x5ApSiqr3EM

- Long-Range Transformers for Dynamic Spatiotemporal Forecasting
  - Spacetimeformer
  - https://arxiv.org/abs/2109.12218
  - sees multivariate time series in the perspective of spatiotemporal sequence
  - README, TSF(time series forecasting)

- Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
  - https://arxiv.org/abs/2012.07436
  - AAAI 2021
  - model architecture
    - sparse self attention
      - O(L log L) time/memory complexity
  - README, time series

- Learning to See by Looking at Noise
  - https://arxiv.org/abs/2106.05963

- Pay Attention to MLPs
  - https://arxiv.org/abs/2105.08050
  - gMLP
    - MLP + spatial gate
  - aMLP
    - gMLP + tiny attention

- Learning in High Dimension Always Amounts to Extrapolation
  - https://arxiv.org/abs/2110.09485

- Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
  - https://arxiv.org/abs/2104.13478
  - https://youtu.be/w6Pw4MOzMuo
  - differential geometry

- Reward is enough
  - https://www.sciencedirect.com/science/article/pii/S0004370221000862
  - RL, reinforcement learning
  - README

- Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation
  - https://arxiv.org/abs/2106.04399

- Intriguing Properties of Contrastive Losses
  - https://arxiv.org/abs/2011.02803
  - Ting Chen et al. (Google Research)
  - NeurIPS 2021
  - more analysis on contrastive learning like SimCLR
    - generalized contrastive losses
      - $L_\text{generalized contrastive} = L_\text{alignment} + L_\text{distribution}$
      - SWD(Sliced Wasserstein Distance) for supporting diverse prior distributions.
    - Instance-based objective can learn on images with multiple objects and learn good local features
      - SimCLR can learn on images with multiple objects
      - SimCLR learns local features that exhibit hierarchical properties
    - features supression limits the potential of contrastive learning
      - Easy-to-learn features (MNIST digits) suppress the learning of other features (Image Net object class)
      - The presence of dominant object suppresses the learning of features of smaller objects
      - Extra channels with a few bits of easy-to-learn mutual information suppress the learning of all features in RGB channels

- Why AI is Harder Than We Think
  - https://arxiv.org/abs/2104.12871
  - Summary
    - The process of human intelligence is more complex than we think.
      - There are unexpected obstacles between narrow and general intelligence.
      - Human intelligence is integrated with body, emotions, desires, a strong sense of selfhood and autonomy, and a commonsense understanding of the world.
        - Not clear if these can be separated
    - Wishful mnemonics misleads us about our understanding AI
      - "Neural" networks
      - Machine "learning" or deep "learning"
      - AlphaGo "thought" it would win
      - "General" Language Understanding Evaluation
  - Questions
    - Can we formulate a simulation environment mimic the way human intercact the world, the body and many other human attributes
      - simulator
        - pain
        - vision
        - smell
        - temperature
        - taste
        - sounds
          - spoken language
        - sense of direction
        - emotions and desires
          - hunger
          - attachment
          - curiosity
      - aids
        - pretrained CNN
        - mfcc

## 2020

- Training Generative Adversarial Networks with Limited Data
  - Karras, NVIDIA
  - NIPS2020
  - https://arxiv.org/abs/2006.06676
  - Overfitting in GAN
    - discriminator outputs for the real training images and fake images diverge 
    - discriminator accuracy on the validation images decreases

  - The correct way to calculate FID
    - calculate it between the full training set and 50k generated images

  - Augmentation that do not leak
    - any augmentation is non-leaking as long as the corruption process is represented by an invertible transformation of probability distributions over the data space.

  - we can make almost any augmentation non-leaking by only applying it at a probability $p < 1$.
  - Adaptive discriminator augmentation (ADA)
    - overfitting heuristics for GAN
      - $r_v = {\mathbb{E}[D_\text{train}] - \mathbb{E}[D_\text{validation}] \over \mathbb{E}[D_\text{train}] - \mathbb{E}[D_\text{generated}]}$
      - $r_t = \mathbb{E}[\operatorname{sign}(D_\text{train})]$
        - they found this was better than $r_v$

    - starts from $p = 0$
    - increase $p$ a little if it looks overfitting and vice versa

  - augmentation methods
    - useful 
      - pixel blitting
      - geometric transformation

    - moderate
    - not useful

  - number of images
    - 2k
      - augmentations was useful

    - 10k
      - less helpful

    - 140k
      - harmful

- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
  - https://arxiv.org/abs/2010.05646
  - 카카오엔터프라이즈 AI Lab
  - GAN
    - one generator
      - mel-spectrogram to Waveform
      - transposed convlution
        - deconvolution + stride + padding
        - https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
    - two discriminators

- Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs
  - https://arxiv.org/abs/2003.00152
  - (tutorial) https://e2eml.school/batch_normalization.html
  - Batch normalization
  - README

- Implicit Neural Representations with Periodic Activation Functions
  - https://arxiv.org/abs/2006.09661
  - https://vsitzmann.github.io/siren/
  - used sine functions which is periodic as for activation functions
    - NN weights will act as angular velocity
    - NN bias will shift phase
    - Since the cosine function is the derivative of sine function and they are different only by the phase, we can easily derive the derivative function of whole NN.
    - NN is now much better at representation especially for images or audio signals.
    - NN can be trained much faster than when using other activation function.
    - Weights should be carefuly initialized.
  - (addtionally)
    - We can train a NN to represent an image by training it to take inputs of coordinates and returns a color
    - We can use NN to solve mathematical or physical problems such as PDE or Obtimizaiton problems just by setting the loss function properly.

- A Simple Framework for Contrastive Learning of Visual Representations
  - https://arxiv.org/abs/2002.05709
  - Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
  - ICML 2020
  - SimCLR
    - image transformation: $x \to \tilde{x}_i$
    - f
      - $h_i = f(\tilde{x}_i)$
      - base encoder
    - g
      - $z_i = g(h_i) = W^{(2)} \sigma(W^{(1)h_i})$
      - A small neural network projection head
      - maps representations to the space where constrastive loss is applied
      - this is not used when generating representations during inference but it's important for better representations during training
      - may drop information such as colors which is not useful
    - major components for better representations learning
      - important to choose composition of data augmentations to form positive pairs
        - recommendation
          - random crop and resize, color distortion, blur
        - single augmentation method alone doesn't work that much
      - separate projection for contrastive loss function
      - unsupervised learning benefits from scaling up more than supervised learning
      - normalized cross entropy loss with adjustable temperature was better than other losses
    - objective function
      - $\ell_{i, j}=-\log \frac{\exp \left(\operatorname{sim}\left(\boldsymbol{z}_{i}, \boldsymbol{z}_{j}\right) / \tau\right)}{\sum_{k=1}^{2 N} \mathbb{1}_{[k \neq i]} \exp \left(\operatorname{sim}\left(\boldsymbol{z}_{i}, \boldsymbol{z}_{k}\right) / \tau\right)}$
      - NT-Xent
        - the normalized temperature-scaled cross entropy loss
      - make them contrastive
    - larger batch size and longer training are good to get the better performance

- Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
  - https://arxiv.org/abs/2006.10739
  - http://people.eecs.berkeley.edu/~bmild/fourfeat/
  - README

- Implicit Geometric Regularization for Learning Shapes
  - https://arxiv.org/pdf/2002.10099.pdf
  - https://www.youtube.com/watch?v=rUd6qiSNwHs&feature=youtu.be
  - README

- Reformer: The Efficient Transformer
  - https://arxiv.org/abs/2001.04451
  - https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0
  - https://www.youtube.com/watch?v=i4H0kjxrias
  - https://en.wikipedia.org/wiki/Locality-sensitive_hashing
  - https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html
  - $O(L \log L)$ computation and memory complexity
    - $L$ length of sequence
  - Locality sensitive hashing
  - Value

- A Simple Framework for Contrastive Learning of Visual Representations
  - https://arxiv.org/abs/2002.05709
  - https://www.youtube.com/watch?v=FWhM3juUM6s
  - apply 2 different data augmentation
  - make the CNN results from those 2 data to be similar
  - can be used for semi-supervised learning, unsupervised learning
  - self-supervised learning

- AutoML-Zero: Evolving Machine Learning Algorithms From Scratch
  - https://arxiv.org/abs/2003.03384
  - https://hoya012.github.io/blog/automl-zero-review/
  - README
  - prerequisite
    - The Evolved Transformer
    - Neural Architecture Search with Reinforcement Learning

- Towards a Human-like Open-Domain Chatbot
  - https://arxiv.org/abs/2001.09977
  - NLP, chatbot, seq2seq
  - SSA(Sensibleness and Specificity Average)
  - perplexity
  - Evolved Transformer
  - Adafactor optimizer
    - to save memory
  - training details
    - TPUv3 pod (2048 TPU cores)
      - TPUv3 has 16GB of high-bandwidth memory
    - 30 days
    - 2.6B parameters
    - Meena dataset containing 40B words
    - dropout = 0.1 (to stem overfitting)
    - T(Temperature) = 0.8
    - N(Number of samples) = 20
  - pros
    - Looks like enough as a counterpart for language practice
  - cons
    - not that smart
  - possible improvements
    - How can it build complex knowledge-base and use it

- Lecture notes on ridge regression (v5)
  - https://arxiv.org/abs/1509.09169
  - high dimensional data would be collinear
    - then the moment matrix is singular
      - moment matrixes are symmetric
      - symmetric matrixes have orthogonal eigenvectors
      - moment matrixes can be decomposed as the sum of products of all eigenvalues, the corresponding eigenvector, and its transposed vector.
      - the inverse of moment matrixes can be decomposed as the sum of products of all reciprocal values of eigenvalue, the corresponding eigenvector, and its transposed vector.
    - then OLS has no solution
  - as ad-hoc, we can add diagonal values to the moment matrix to make it nonsingular
  - as post-hoc, minimizing squared error with ridge penalty can be considered as equivalent
  - Bayesian linear regression can be considered as equivalent as well
  - ridge regressor is a biased estimator

- The Optimal Ridge Penalty for Real-world High-dimensional Data Can Be Zero or Negative due to the Implicit Ridge Regularization
  - https://arxiv.org/abs/1805.10939
  - when n << p, the λ → 0 limit, corresponding to the minimum-norm OLS solution, can have good generalization performance
    - minimum-norm OLS can be found by pseudo inverse just like plain OLS solutions
  - explicit ridge regularization with λ > 0 can fail to provide any further improvement;
  - moreover, the optimal value of λ in this regime can be negative;
  - this happens when the response variable is predicted by the high-variance directions while the low-variance directions together with the minimum-norm requirement effectively perform shrinkage and provide implicit ridge regularization
  - OLS, minimum-norm OLS

- Training Recurrent Neural Networks Online by Learning Explicit State Variables
  - https://openreview.net/pdf?id=SJgmR0NKPr
  - ICLR2020
  - Fixed point propagation
  - explicitly learn state vectors breaking the dependencies across time
  - README, RNN, online learning, time series

- N-BEATS: Neural basis expansion analysis for interpretable time series forecasting
  - https://arxiv.org/abs/1905.10437
  - ElementAI founded by Yoshua Bengio
  - ICLR2020
  - univariate timeseries point forecasting
  - model
    - deep neural net
      - backward and forward residual links
      - fully connected layers
      - the deeper the better
        - they tried 30 stack of depth 5
  - datasets for evaluation
    - M3
      - https://forecasters.org/resources/time-series-data/m3-competition/
    - M4
      - https://www.kaggle.com/yogesh94/m4-forecasting-competition-dataset
    - Tourism
      - https://www.kaggle.com/c/tourism1
      - https://www.kaggle.com/c/tourism2
  - time series
- InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs
  - https://arxiv.org/abs/2005.09635
  - propose a way to edit a latent vector to modify the resulting image
  - process
    - given a training set of image-and-label pairs

    - train a StyleGAN or any other GAN using the images in the training set

    - find a latent vector $w$ for each training set pair

    - train a SVM predicting the label given $w$

    - modify $w$ with respect to the normal vector direction of the separating hyperplane found by SVM 

    - with the modified $w^\prime$, you can generate a modified image

## 2019

- Large Scale GAN Training for High Fidelity Natural Image Synthesis
  - ICLR 2019
  - BigGAN/BigGAN-deep
    - class conditioned GAN
  - tried to scale up a few conditional GAN models
  - baseline: SA-GAN
  - how?
    - shared embeddings for the generator
    - skip-z connections from the latent variable
    - truncation trick
      - trading off variety and fidelity explicitly
      - train a model with $z \sim \mathcal{N}(\mathbf{0}, I)$
      - when sampling resample values if the value went beyond the threshold
        - threshold = 2
          - more variety less fidelity
        - threshold = 0.04
          - less variety more fidelity
      - orthogonal regularization
        - $R_\beta(W) = \beta||W^T W - I||^2_F$
        - $R_\beta(W) = \beta||W^T W \odot (\mathbf{1} -I)||^2_F$
          - (empirically this was better)
    - etc.
      - smaller learning rate
      - train D twice and train G once
      - compute and apply stats for Batch Normalization across all devices
      - increase batch size
      - make it wider (use 50% more channels)
  - observation
    - The symptoms of mode collapse are sharp and sudden
    - Mode collapse happens when the singular values in G explode
    - The performance of D is more important than the performance of G
- MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis
  - https://arxiv.org/abs/1910.06711
  - generator
    - no global noise vector
    - residual connection
    - kernel size as a multiple of stride
    - dilation grows a power of the kernel-size
    - weight normalization
  - discriminator
    - 3 discriminators with identical architecture
      - D1
        - raw audio
      - D2
        - raw audio downsampled by a factor of 2
      - D3
        - raw audio downsampled by a factor of 4
    - weight normalization
  - training objective
    - the hinge loss version of the GAN objective
    - use feature matching objective to train the generator
  - spectrogram to wave
- Loss Landscape Sightseeing with Multi-Point Optimization
  - https://arxiv.org/abs/1910.03867
  - Loss surface is surprisingly diverse and intricate in terms of landscape patterns it contains.
  - Adding batch normalization makes it more smooth.
  - README
- YOLACT++: Better Real-time Instance Segmentation
  - https://arxiv.org/abs/1912.06218v1
    - this paper includes [the original YOLACT paper](https://arxiv.org/abs/1904.02689v2)
  - official implementation
    - https://github.com/dbolya/yolact
- HarDNet: A Low Memory Traffic Network
  - https://arxiv.org/abs/1909.00948v1
  - state of the art realtime semantic segmentation
  - MAC(number of multiply-and-accumulates operations)
    - the lower the better
  - CIO(Convolutional Input/Output)
  - MoC
    - MAC over CIO
  - IoU
    - Intersection over Union
    - https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  - c: number of channels
  - h: height of feature maps for a convolution layer l
  - w: width of feature maps for a convolution layer l
  - k: index of layer
  - memory traffic
    - the less the better
  - related works
    - stochastic depth regularization (2016)
    - DenseNet (2017)
      - concatenating all proceeding layers as a shortcut
    - SparseNet
      - sparsified dense nets
    - LogDenseNet
      - globally sparsified dense nets
    - FractalNet
      - averages shortcuts
  - HarDNet
    - Harmonic Densely Connected Network
    - shortcut from $k$ to $k - 2^n$ where n is a natural number
    - $c = km^n$
    - $m = [1.6, 1.9]$
    - HDB(Harmonic Dense Block)
    - optional bottleneck
      - $c_o = \sqrt{c_\text{in} / c_\text{out}} \times c_\text{out}$
    - implementations
      - https://github.com/PingoLH/Pytorch-HarDNet
      - https://github.com/PingoLH/FCHarDNet
      - https://github.com/PingoLH/PytorchSSD-HarDNet
- MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
  - https://arxiv.org/abs/1911.08265
  - RL, README
- U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation
  - https://arxiv.org/abs/1905.01164
  - GAN, README
- SinGAN: Learning a Generative Model from a Single Natural Image
  - https://arxiv.org/abs/1905.01164
  - GAN
- Single Headed Attention RNN: Stop Thinking With Your Head
  - https://arxiv.org/abs/1911.11423
  - https://smerity.com/articles/2017/baselines_need_love.html (shared by Yury)
  - README
- Stand-Alone Self-Attention in Vision Models
  - https://arxiv.org/abs/1906.05909
  - README
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
  - https://arxiv.org/abs/1905.11946
  - README
- Rethinking the value of network pruning
  - https://arxiv.org/abs/1810.05270
  - README
- Lottery ticket hypothesis
  - https://arxiv.org/abs/1803.03635
  - one-shot pruning
    - steps
      - Randomly initialize a neural network
        - It's like buying a lot of lottery tickets.
        - Weights with good initial values for the architecture are winning ticket.
          - They will be trained well.
          - Without other tickets, they can be trained to have good accuracy
        - Big networks have bought tickets.
      - Train the network
      - Set p% of weights with the lowest magnitude from each layer to 0 (this is the pruning)
      - Reset the pruned network weights to their original random initializations to retrain and see if it can achieve the same accuracy
  - Iterative pruning
    - It does one-shot pruning iteratively
    - It generates smaller network than one-shot pruning.
  - Pruning itself can be considered as a learning process.
- High-Fidelity Image Generation With Fewer Labels
  - https://arxiv.org/abs/1903.02271
  - README
- GauGANs-Semantic Image Synthesis with Spatially-Adaptive Normalization
  - https://arxiv.org/abs/1903.07291
  - README
- Deep Equilibrium Models
  - https://arxiv.org/abs/1909.01377
  - README
- IMAGENET-Trained CNNs are Biased Towards Texture
  - https://arxiv.org/abs/1811.12231
  - README
- ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations
  - https://arxiv.org/abs/1909.11942
  - README
- Zero-Shot Word Sense Disambiguation Using Sense Definition Embeddings via IISc Bangalore & CMU
  - https://www.aclweb.org/anthology/P19-1568/
  - README
- A Geometric Perspective on Optimal Representations for Reinforcement Learning
  - https://arxiv.org/abs/1901.11530
  - README
- Weight Agnostic Neural Networks
  - https://arxiv.org/abs/1906.04358
  - README
- Deep Double Descent: Where Bigger Models and More Data Hurt
  - https://arxiv.org/abs/1912.02292
  - README
- On the Measure of Intelligence
  - https://arxiv.org/abs/1911.01547
  - README
- The Evolved Transformer
  - https://arxiv.org/abs/1901.11117
  - README, NAS
  - prerequisite
    - Neural Architecture Search with Reinforcement Learning
- Augmented Neural ODEs
  - https://arxiv.org/abs/1904.01681
  - Address the limitations of the original Neural ODE method by
    - adding an dimension
      - make ODE function simpler with less evaluations
      - may end up with overfitting
- When Gaussian Process Meets Big Data: A Review of Scalable GPs
  - https://arxiv.org/abs/1807.01065
  - https://exoplanet.dfm.io/en/stable/tutorials/gp/
  - README
- Encoding high-cardinality string categorical variables
  - https://arxiv.org/abs/1907.01860
  - Gamma Poisson Factorization on substring counts
    - high interpretability
  - Min-hash encoder
    - LSH based
    - fast approximation of string similarities
    - highly scalable
- Deep Factors for Forecasting
  - https://arxiv.org/abs/1905.12417
  - README, time series
- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
  - https://arxiv.org/abs/1912.09363
- Generalized Sliced Wasserstein Distances
  - https://arxiv.org/abs/1902.00434
  - Soheil Kolouri et al.
  - SW, SWD, GSW
- An Entity Embeddings Deep Learning Approach for Demand Forecast of Highly Differentiated Products
  - https://www.sciencedirect.com/science/article/pii/S2351978920303243
  - they tried to solve these questions
    - how to forecast the quantity of sale of a new product?
      - learn entity embedding for each product
      -   for a new product, take the timeseries of similar products into account.
            - try to find similar products and get a good initial value from them.
            - generate pseudo time series using it.
    - how to deal with unstable data corresponding to fashion trends and customers diversity.
      - cluster data and remove outliers from the perspective of each cluster
- Learning Loss for Active Learning
  - https://arxiv.org/abs/1905.03677
  - https://youtu.be/YU_NO7pYObM
  - CVPR
  - pick unlabled examples that are most probable to improve the model performance by labeling them
  - types
    - entropy based
    - least confident examples
    - loss based
      - marginal loss
        - we want the loss prediction module to be able to tell which loss would be greater among each pair of examples. and the difference to be larger than xi
        - can be applied any form of problem settings

## 2018

- Self-Attention with Relative Position Representations
  - https://arxiv.org/abs/1803.02155
  - extend self-attention by taking account of relative positional embeddings
    - add a learnt relative positional embedding into K/V
    - relative position is clipped as (-k, k) so there can be 2k + 1 position values
    - weights are shared across multi heads
    - so 2(2k+1) positional embeddings needed
  
- YOLOv3: an Incremental Improvement
  - https://arxiv.org/abs/1804.02767
  - README
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  - https://arxiv.org/abs/1810.04805
  - [Review in Korean](https://reniew.github.io/47/)
  - pre-train transformer model with bidirectional setting
  - MNLI(Multi-Genre Natural Language Inference) dataset
    - https://cims.nyu.edu/~sbowman/multinli/
  - NER(Named Entity Recognition)
  - NSP(Next Sentence Prediction)
  - SQuAD(Stanford Question Answering Dataset)
    - https://rajpurkar.github.io/SQuAD-explorer/
  - NLP
- Universal Language Model Fine-tuning for Text Classification
  - https://arxiv.org/abs/1801.06146
  - ULMFiT
  - NLP
- Understanding Batch Normalization
  - https://arxiv.org/abs/1806.02375
  - BN, batch normalization
- How Does Batch Normalization Help Optimization?
  - https://arxiv.org/abs/1805.11604
  - BN, batch normalization
- A Tutorial on Deep Learning for Music Information Retrieval
  - https://arxiv.org/abs/1709.04396
  - spectrograms
    - STFT (Short-Time Fourier Transform)
      - a type of FFT
        - O(N log(N))
    - Mel-spectrogram
      - compressed STFT optimized for human auditory perception
      - not invertible to audio signal
    - CQT (Constant-Q Transform)
      - frequency distribution of pitch for musical notes
    - Chromagram
      - pitch class profile
      - energy distribution
      - CQT representation folding in the frequency axis
  - practical advices
    - data preprocessing
      - use normalization
      - use logarithmic mapping
      - downsampling to 8-16kHz is fine
    - aggregating information
      - pooling
      - strided convolutions
      - recurrent layers
    - depth of networks
      - 5 or more convolution network layers
- Music Transformer
  - https://arxiv.org/abs/1809.04281
  - https://magenta.tensorflow.org/music-transformer
  - README
- Similarity encoding for learning with dirty categorical variables
  - https://arxiv.org/abs/1806.00979
- BRITS: Bidirectional Recurrent Imputation for Time Series
  - https://arxiv.org/abs/1805.10572v1
  - README, time series
- Temporal Pattern Attention for Multivariate Time Series Forecasting
  - https://arxiv.org/abs/1809.04206
  - README, time series
- Neural Ordinary Differential Equations
  - https://arxiv.org/abs/1806.07366
  - Best paper award at NeurIPS 2018
  - Review
    - ODE
      - there is only one dependent variable
    - ResNet looks like Euler method for ODE methods.
    - Replace residual network with ODE
    - going
      - from t=0 (where the target function value is an input value)
      - to t=1 (where the taret function value is the corresponding output value)
- Glow: Generative Flow with Invertible 1x1 Convolutions
  - https://arxiv.org/abs/1807.03039
  - Review
    - flow based model
      - bijective
        - can find latent variable directly
      - find jacobian determnants easily by using the functions that the jacobian matrix of each is triangular.
- WaveGlow: A Flow-based Generative Network for Speech Synthesis
  - https://arxiv.org/abs/1811.00002
  - Glow + WaveNet
  - flow based
- FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models
  - https://arxiv.org/abs/1810.01367
  - published after Glow 2018
  - ICLR 2019
  - flow based
  - prerequisites
    - adjoint sensitivity
    - Hutchinson's trace estimator
    - https://blog.shakirm.com/2015/09/machine-learning-trick-of-the-day-3-hutchinsons-trick/
  - TODO
    - https://www.youtube.com/watch?v=JPIy50saEoA
- Wasserstein Auto-Encoders
  - https://arxiv.org/abs/1711.01558
  - ICLR 2018
  - WAE-GAN, WAE-MMD
  - README
- On the Information Bottleneck Theory of Deep Learning
  - https://openreview.net/forum?id=ry_WPG-A-
  - demonstrates what's happening in the layers of neural networks during training in terms of mutual information
    - I(X;T)
    - I(T;Y)
  - initial fitting phase
    - network learns with respect to both inputs and outputs
  - compression phase
    - forgets information which is not related to the output labels
  - https://youtu.be/bLqJHjXihK8

## 2017

- A general reinforcement learning algorithm that masters chess, shogi and Go through self-play
  - https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go
  - RL, reinforcement learning, AlphaZero
- Attention is all you need
  - https://arxiv.org/abs/1706.03762
  - https://nlp.seas.harvard.edu/2018/04/03/attention.html
  - http://jalammar.github.io/illustrated-transformer/
  - NLP, self-attention, transformer
  - encoder
    - bidirectional
  - decoder
    - unidirectional with masking
  - attention with Q, K, V
  - positional encoding
    - https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
  - label smoothing
    - the idea of using labels like `0.7, 0.1, 0.1, 0.1` rather than `1.0, 0.0, 0.0, 0.0`
    - BLEU was better but PPL was worse
- Wide Residual Networks
  - https://arxiv.org/abs/1605.07146
    - when it comes to wide networks, less 50 residual blocks are enough for usual image classification problem
    - CNN, ResNet
- Neural Architecture Search with Reinforcement Learning
  - https://arxiv.org/abs/1611.01578
  - README, NAS
- Mask R-CNN
  - https://arxiv.org/abs/1703.06870
  - instance segmentation
- The Reversible Residual Network: Backpropagation Without Storing Activations
  - https://arxiv.org/abs/1707.04585
  - saving memory by calculating input rather than keep it in memory during backpropagation
  - RevNet

- Cat2Vec Learning Distributed Representation of Multi-field Categorical Data.
  - (Cat2Vec, ICLR 2017) https://openreview.net/pdf?id=HyNxRZ9xg
  - (PMLN, DLP 2019) https://dl.acm.org/doi/abs/10.1145/3326937.3341251
  - Word order is not assumed.
  - PMLN(Pairwise Multi Layer Nets) is used to find the interactions between fields.
    - addition
    - element-wise multiplication
  - Trained while discerning the true data from the modified fake data.

- DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
  - https://arxiv.org/abs/1704.04110
  - Amazon
  - ICML 2017
  - RNN for probabilistic forecasting
    - incorporating negative binomial likelihood for count data
    - special treatment for the case when the magnitudes of the time series vary widely
    - learn by maximizing log likelihood
  - additional advantages
    - less feature engineering
    - makes probabilistic forecasts in the form of Monte Carlo samples
      - that can be used to compute a consistent quantile estimates
    - able to provide forecasts for items with little or no history at all
      - a case where traditional single-item forecasting methods fail
      - by learning from similar items
    - does not assume Gaussian noise
      - can incorporate a wide range of likelihood functions
      - allowing the user to choose one that is appropriate for the statistical properties of the data
  - README, time series

- Unbiased Online Recurrent Optimization
  - https://arxiv.org/abs/1702.05043
  - (ICLR 2018) https://openreview.net/pdf?id=rJQDjk-0b
  - README, time series, online learning, RNN

- Improved Training of Wasserstein GANs
  - https://arxiv.org/abs/1704.00028
  - WGAN-GP
  - README

- Wasserstein GAN
  - https://arxiv.org/abs/1701.07875
  - Wasserstein distance
    - Earth-Mover distance
    - https://youtu.be/CDiol4LG2Ao
  - Addressed the vanilla GAN problems
    - vanishing gradients
    - mode collapse
  - limitation
    - weight clipping to ensure Lipschitz continuity was too harsh
  - But practically, the improvement is not that great
  - WGAN

- Least Squares Generative Adversarial Networks
  - https://arxiv.org/abs/1611.04076
  - Additional resources
    - https://jaejunyoo.blogspot.com/2017/03/lsgan-1.html
    - https://en.wikipedia.org/wiki/F-divergence
  - LSGAN, Pearson χ2 divergence

## 2016

- Entity Embedding of Categorical Variables
  - https://arxiv.org/abs/1604.06737
  - Cheng Guo, Felix Berkhahn
  - Each one-hot encoded field is mapped into an Euclidean space
    - all those outputs are concatenated with of other varaibles
    - and they all are trained in a network at the same time.
  - The entitiy embedding matrix for each category field can be used for visualizing categorical data
  - The concatenated outputs can be used for data clustering
  - one-hot encoded values are mapped
  - questions
    - how to choose the embedding dimension??
      - the more complex the more dimensions
      - guess how many features might be needed to describe the entities
      - start with the number of categories minus one
    - Linear Relation Embeddings vs Structured Embeddings
    - Spearman rank correlation coefficient
      - measures monotonic correlation
      - different from the Pearson correlation that measures linear correlation
      - https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

- Robust Online Time Series Prediction with Recurrent Neural Networks
  - https://ieeexplore.ieee.org/document/7796970
  - robust against anamalies by giving weight/penalty
  - time series, LSTM, RNN, online learning

- Online ARIMA Algorithms for Time Series Prediction
  - https://ojs.aaai.org/index.php/AAAI/article/view/10257
    - or http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.3100&rep=rep1&type=pdf
  - Uses ARIMA(k + m, d, 0) instead of ARIMA(k, d, q)
  - ARIMA, time series

- f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization
  - https://arxiv.org/abs/1606.00709
  - additional references
    - [UNIST 유재준 교수님 한글블로그](https://jaejunyoo.blogspot.com/2017/06/f-gan.html)
    - [conjugate function](https://convex-optimization-for-all.github.io/contents/chapter03/2021/02/12/03_03_the_conjugate_function/)
    - https://en.wikipedia.org/wiki/F-divergence
  - f-GAN

- Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
  - https://arxiv.org/abs/1511.06434
  - DCGAN

## 2015

- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  - https://arxiv.org/abs/1502.01852
  - weight initialization
  - Kaiming He initialization



## 2014

- (Blog) Neural Networks, Manifolds, and Topology
  - https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/

- The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo
  - https://arxiv.org/abs/1111.4246
  - sampling, README

- Neural Machine Translation by Jointly Learning to Align and Translate
  - https://arxiv.org/abs/1409.0473
  - README, attention

- Neural Turing Machine
  - https://arxiv.org/abs/1410.5401
  - README, attention

- Conditional Generative Adversarial Nets
  - https://arxiv.org/abs/1411.1784
  - Conditional GAN

- Generative Adversarial Networks
  - https://arxiv.org/abs/1406.2661
  - Ian J. Goodfellow et al.
  - GAN
  - limitation
    - mode collapse
    - vanishing gradients due to JS-divergence

- Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
  - LSTM and GRU have advantages over vanilla RNN
    - can remember input features
    - can bypass timesteps so that the error could be backward-propagated losing less of the gradient

## 2013

- Auto-Encoding Variational Bayes
  - https://arxiv.org/abs/1312.6114
  - Diederik P Kingma et al.
  - object
    - minimizes a reconstruction error
      - in the sense of the KL divergence between the parametric posterior and the true posterior
    - with regularization
      - making KL divergence of q_φ(z|x)||p_θ(z)
  - pros
    - stable training
    - encoder-decoder architecture
    - nice latent manifold structure
  - cons
    - generated samples are blurry especially when it comes to natural images
  - VAE

## 2010

- Understanding the difficulty of training deep feedforward neural networks
  - http://proceedings.mlr.press/v9/glorot10a.html
  - weight initialization
  - Xavier Glorot initialization

## 2009

- Feature-Weighted Linear Stacking
  - https://arxiv.org/abs/0911.0460
  - the second place team on the Netflix Prize competition used this
  - the weight for each ensemble base learner is also estimated by a linear meta leaner

## 1987

- A Conceptual Introduction to Hamiltonian Monte Carlo
  - https://arxiv.org/abs/1701.02434
  - HMC, MCMC, sampling, README

## 1978

- Regression Quantiles
  - Roger Koenker; Gilbert Bassett, Jr.
  - Gaussian noise assumption doesn't best fit into heavy tail noise distributions. So use order statistics instead.
  - Quantile Loss
    - $\sum q(y - \hat{y})_+ + (1-q)(\hat{y} - y)_+$
      - where
        - $(\cdot)_+ = max(0, \cdot)$
        - $q$: quantile e.g. 0.1 for 10th percentile, 0.5 for median, 0.9 for 90th percentile, etc.
        - $\hat(y)$: a prediction calculated by the regression model
  - robustness
    - resiliance of statistical procedure to deviations from the assumptions of hypothetical models
  - robustic statistics
    - good even under non-normal distributions
    - not senstiively affected by outliers
  - Cramér-Rao bound
    - $\operatorname{var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$
      - $I(\theta)$: fisher information
  - Lindeberg's condition
    - Sufficent condition for the central limit theorem (CLT) to hold for a sequence of independent random variables.
  - Gauss–Markov theorem
  - estimators
    - least squares
      - minimizing the sum of the squares of the residuals
      - types
        - linear least sqaures
          - ordinary least sqaures
          - weighted linear least sqaures
          - generalized linear least squares
        - non-linear least sqaures
    - sample mean
      - mean of a sample
      - sensitive to outliers
        - not robustic statistics
      - Gaussian noise was used only to use sample
    - L-estimators
      - α-trimmed mean
        - the mean of the sample after the proportion α of largest and smallest observation have been removed
      - Huber's minimax estimator(?)
    - sample median
      - least absolute error (LAE) estimator
      - an alternative to the least sqaures estimator
    - M-estimators

## references

- https://analyticsindiamag.com/best-machine-learning-papers-2019-nips-icml-ai/

# ML

## Questions

- What is PSI, AUPRC, KS
- How do Cluster Sets work
- What is Basis expansion?
  - http://www.csc.kth.se/utbildning/kth/kurser/DD3364/Lectures/Lecture4.pdf
  - http://madrury.github.io/jekyll/update/statistics/2017/08/04/basis-expansions.html
- How to calculate field importance values and why it is associated with the model details
- How does the model update works
- what is min bin size
- how to stabilize deep net

## General machine learning

### Machine learning types

Semi-supervised learning

- pre-condition
  - there are a lot of unlabeled data
  - there are a few labeled data
- For example, Facebook let users label recognized faces after grouping them first without labeling them.

Active learning

- the learner picks samples and asks user the labels interactively while learning
- a form of semi-supervised learning


Transfer learning

- put some more weights for the model’s difference.
- when we cannot access both data A and B together.
- when A is not related to B that much

Joint learning

- concatenate two datasets then we have more data.
- leave missing data as missing data
- use when we have small A but we have many B

Pre-training

- to set initialization
- sometimes we reuse just the bottom layers of NN from A for B

Online learning

- real-time model update

Batch learning

- sporadic model update with all data including newly gathered data

Instance-Based Learning

- remember every instance and predict with the average of most similar k instances
- k-NN (K-Nearest Neighbors)
- if there are tied distance we use mean of the values
- we could use the weighted average
- in this case, k = n makes sense
- or we could do locally weighted average
- which could be called locally weighted regression
- lazy
  - learn fast
  - inference slowly
- distance metric
  - euclidean
  - manhattan
- bias
  - locality
    - the closer the more similar
  - smoothness
    - we average
  - all features matter equally
      in terms of the definition of distance
### Gradient descent based Optimization

(Batch) Gradient descent

- see all data and move
- cost function 의 지형에서 가장 낮은 지점을 찾기
- 바닥만 보고 지형의 가장 낮은 지점 찾는 상황에 비유할 수 있음.
- global minimum을 찾는 것은 사실 상 보장할 수 없음.

SGD(Stochastic Gradient Descent)

- 2 cases
  - SGD
    - see only single data and move
  - Mini-batch gradient descent
    - see batch data and move
    - when the batch size is small there is tendency to train the model better in terms of performance

Newton's method

- Use 2nd order polynomial
- Uses Hessian,, calculations are required as much as square of number of parameters
  - its cost is too high for a number of parameters
- Go to the supposedly minimum point directly with 2nd order function approximation
  - but still it can use learning rate due to the limit of approxmation

Momentum

- Adjust direction
- tries to fix oscillating problems
- Move following current gradient descent and go further following the momentum

NAG(Nesterov Accelerated Gradient)

- Adjust direction
- better at oscillating problem and escaping local minimum than momentum method
- Move following the momentum first and calculate the gradient descent there
- Known as good for RNN

AdaGrad

- Accumulate weight changes for each weight separately and penalty that weight separately with a proportional amount to the accumulated change size
- number of learning rates is the same as the number of parameters
- monotonically decreasing learning rate

Adadelta

- Adjust step size
- Move with the penalty of recent changes
- uses exponential moving average
- Prevent monotonic decreasing learning rates
- there is no learning rate as a hyper parameter

RMSprop (Root Mean Square Propagation)

- Adjust step size
- Move with the penalty of recent changes
- uses exponential moving average in the denominator
- Almost same as Adadelta but has learning rate as a hyper parameter
- Move with the penalty of recent changes
- introduced by Geoff Hinton in a Coursera Lecture

Adam (Adaptive Moment Estimation)

- Adjust both step and direction
- epsilon
- prevent the changes from being too large
- for regression, the default value `1e-8` might be too small so it could affect initial weight changes too much.
  - Try the values other values like. `1e-4`, `1e-2`, `1e-1`, `1.0`, or something another bigger number like this.
- beta1
  - direction parameter
  - the greater it is the more it uses the direction of momentum
- beta2
  - step size parameter
  - the greater it is the more it uses the size of previous steps
- m_t
  - adjust momentum
- v_t
  - adjust step size
- there are the bias correction terms constant to be an unbiased estimator

NAdam

- Adam + NAG

AMSGrad

- non increasing learning rate

AdaBound or AmsBound

- https://medium.com/syncedreview/iclr-2019-fast-as-adam-good-as-sgd-new-optimizer-has-both-78e37e8f9a34
- bound to a range.
- the range starts from (0, ∞) and converges in the end

### Independent and identically distributed random variables

- defined for 2 random variables.
- can be used for 2 field variable (2 or more columns)
  - e.g. Naive Bayes classifier assumes that input features are i.i.d.
- can be used for 2 observations (2 or more rows)
  - e.g. in usual ML settings, we assume each single observation (row) was realized from a single multivariate random variable.
  - In TS problem, we would want to make the observations i.i.d. by applying the window transform.

https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables

### Cross validaiton

- trained model needs to be tested with a data set unseen during training
- would not be needed when we have a huge enough test set
  - but it's usually not true
  - we want to make the training set as big as possible
- we want to estimate the test error by using the train set
  - it's a type of resampling
- types
  - exhaustive cross validation
    - leave-p-out cross validation
      - low bias on the validation error
  - non-exhausitve cross validation
    - k-fold cross validation
    - hold-out method or validation set approach
      - high bias on the validation error when the validation set size is smaller

### Imputation

- How to handle missing values
- they might be replaced with mean/mode/most frequent value/k-NN/...
- https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779


### Gradient value clipping

- Gradient value clipping involves clipping the derivatives of the loss function to have a given value if a gradient value is less than a negative threshold or more than the positive threshold.


### SMOTE

- SMOTE(Synthetic Minority Oversampling Technique)
  - generates synthetic instances using the linear interpolation within the minority class
  - steps
    - Pick an instance from the minority class randomly.
    - Find the k-nearest neighbors of the current instance.
    - Take the difference between the current instance and one of the neighbors, then multiply the difference by a random number gap between 0 and 1.
    - Add the result from the previous step to the current instance and create a new synthetic instance.

- Borderline SMOTE
  - learn borders first to generate samples around
- SVM SMOTE
  - train SVM to find the decision boundary
- Kmeans SMOTE
  - clustering
  - filtering - select clusters to be oversampled
  - oversampling - SMOTE within a cluster


- https://escholarship.org/uc/item/99x0w9w0

## discrimitive models

Linear regression

- MLE assuming y is normally distributed
- MAP assuming y is normally distributed and weights are distributed with infinity as the variance

Ridge regression

- MAP assuming y is normally distributed and weights are normally distributed
- The regularization term controls the weight variance ratio to y's variance
- If n > p
  - $\lambda_\text{opt} \gt 0$
- It n << p
  - $\lambda_\text{opt}$ can be negative in some cases
    -
- Tikonov regularization is a generalized Ridge riegression method
  - https://en.wikipedia.org/wiki/Tikhonov_regularization

Lasso regression

- MAP assuming y is normally distributed and weights are distributed as laplace distribution
- The regularization term controls the standard deviation ratio of weights to variance of y
- with the L1 norm regularization term
- it will select features inherently

Ensemble learning

- Begging
  - Bootstrap aggregating
  - https://en.wikipedia.org/wiki/Ensemble_learning
- Boosting
  - https://data-matzip.tistory.com/4
- Stacking
  - train a meta learner with the out-of-fold predictions
  - https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/


## Generative models

![Taxonomy of deep generative models](https://d3i71xaburhd42.cloudfront.net/652d89b80279db596d2447d3f0ea7e1bfb6a9e02/3-Figure1-1.png)

### [Autoencoder](./autoencoder.md)
### Generative adversarial network

### Flow-based generative model

(basic method)

- reversible map
  - from simple random latent varaiable
    - we can sample values from this
    - we know the likelihood function
  - to the training data
    - we can calculate the likelihood by using change of variable law of probabilities

(variants)

- continuous normalizaing flow (FFJORD)
  - formulates the flow as a continuous-time dynamic instead of constructing flow by function composition
  - allows unrestricted neural network architecture

(References)

- [Probabilistic Graphical Models - Stanford cs228 lecture notes](https://ermongroup.github.io/cs228-notes/)
- https://en.wikipedia.org/wiki/Flow-based_generative_model
- https://stanniszhou.github.io/discussion-group/post/ffjord/

## General AI

### Some ideas

- MuZero
- choose best action with MCTS
- use context information
  - eyes
  - nose
  - ears
  - people's emotions
  - people's talk
  - knowledge
- give proper reward and train with it
- every thing to context embedding
- context embedding 2 action
- explainable context embedding
- make a bigger model which is sparsely activated.

## Explainable AI

- latent variables as a graph
- https://christophm.github.io/interpretable-ml-book/

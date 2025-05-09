# Optimizers



## Terms

### Weight decay

- regularize weights to be smaller
- For SGD it's equivalent to L2 regularization



## Algorithms

### Adam

- hyperparameters
  - learning rate
    - the base learning rate to be multiplied by at the end for updating weights(parameters)
  - $\beta_1$
    - how much the first momentum is taken into account
    - PyTorch default: 0.9
  - $\beta_2$
    - how much the second momentum is taken into account
    - PyTorch default: 0.999
- for each parameter its own learning rate is applied
- weight decay depends on the learning rate
- requires warming up

### AdamW

- weight decay doesn't depend on the learning rate
- https://arxiv.org/abs/1711.05101
- requires warming up

### AdamWR

- perform warmup restart of learning rate to escape from local minima
- https://arxiv.org/abs/1711.05101



## Schedulers

- the final number of iterations/epochs might be fixed, and it can be a downside



## References

- https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html
- https://hiddenbeginner.github.io/deeplearning/paperreview/2020/01/04/paper_review_AdamWR.html
# AutoML Framework

## Quetions

- Can we modularize hyperparameter tunning for each transform?
  - Why fit_autotune should be in the model?
  - Can we use external hyperparameter tuning library
  - can we consider the pipepline building as a type of hyperparameter tunning
    - tree like space definition is required?
    - type of algorithm is just one of hyperparameters in most cases
  - We cannot search all the hyperparameters, so user may want to stop/continue searching depending on the resource/

## ideas

- Get just minimum settings from users.
- Can we unify/automate the way how we use pipeline components?
  - adding/removing algorithms/transforms
    - algorithm selection
    - transform selection
  - let's tune hyperparameters always
    - receive hyper parameter space always
    - allow receiving a single hyperparameter point as a kind of hyper parameter space
    - Let's make hyperparameters something that can be customizable
  - fitting
  - timeseries specifics
  - export
- distributed system
  - send a pipeline with a hyperparameter to a node
  - manager sends the task of fitting a pipeline for each hyperparameter
  - manager gathers the results and chooses hyperparameters one by one

## controller

## entities

- models
  - the implementation of auto-tuning needs to be out of the model class
    - the input shape may depends on the parameters.
      - e.g. ARIMA
    - we'd like to initialize the model parameters with the constructor.
    - refer to [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) in the Hugging Face library
- transforms
  - in pytorch they're callable
  - can we fit the chain of transforms by calling the fit method of the outer chain tranform?


## values

- `from_client()` looks bad since it adds dependency on the clients which is the opposite direction of dependencies

## use cases

## blocks

- Controller
  - receives and translate a task request
  - build manager instance and run it
  - run evaluate usecase
  - load/unload required objects
- Usecase
  - manager for main training operations
    - pipeline owner like an orchestra conductor
    - examples
      - single mannual model manager
      - single auto model manager
      - cascade factory manager
      - auto ensemble model manager
      - auto model manager
      - flexible auto model manager
        - select pipeline/algorithms flexibly using optuna or ray.tune
        - keep history in case user might want to see that
  - sub simple operations
    - evaluate
- Transform
  - methods
    - fit
    - apply
  - examples
    - imputation
  - mixins
    - reversible
    - flexible
      - can change connections within the internal transforms
      - affects following transforms and makes it difficult to apply user-defined settings
    - fixed
      - It cannot change connections within the internal transforms.
      - Instead, it should be modified by an external module.
- Source
  - abstraction layer to access data
- Repository
  - abstraction layers to save objects
    - pipeline objects
    - other objects

## Implmenetation idea details
### Window transform

- make "n_lags" to be always the number of lags as used in the literature (like you mentioned)
  - 0 `n_lags` for `X` to be `X_T-0`
  - 1 `n_lags` for `X` to be `X_T-0`, `X_T-1`
- `WindowTransform(field, window)`
  - A transform class that is responsible for pure windowing not directly related to time series models.
  - arguments
    - `field`
    - `window`

(5,) or 5: window_size = 5, n_lags = 4, gap = 0, n_targets = 0
(2, 3) :   window_size = 5, n_lags = 1, gap = 0, n_targets = 3
(2, 1, 2): window_size = 5, n_lags = 1, gap = 1, n_targets = 2
(0, 1, 2): window_size = 3, n_lags = 0, gap = 1, n_targets = 2
(0, 0, 2): window_size = 2, n_lags = 0, gap = 0, n_targets = 2

From the perspective of pipeline, input window or window can mean n_lags + 1? (Not targets?)
  - Is there another term that can refer to the corresponding time points?
From the perspective of dataset, input window or window can mean n_lags + 1 + gap + n_targets?

For example we could create a window transform for training like this.

```py
WindowTransform(
    field='X'
    window=n_lags + 1 + gap + n_targets,
    idx_start= - n_lags
    idx_drop_range=(n_lags + 1, n_lags + 1 + gap),
)
```

For the backward compatibility, we could keep the old arguments and deprecate them. And raise error when the old and new argument names are used at the same time.

## General python boilerplate

- https://www.cosmicpython.com/
  - how to apply orm with depedency injection
    - https://github.com/cosmicpython/code/blob/chapter_02_repository/orm.py

## References

- https://github.com/mindsdb/lightwood
- https://github.com/tensorflow/adanet
- https://medium.com/datadriveninvestor/20-automl-libraries-for-the-data-scientists-e591068dbc6b

## extra keywords

- platform
- design-pattern

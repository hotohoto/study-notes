# MLFLow

- experiment
  - can have runs
  - by default the `Default` experiment is used
- run
  - can log things including model
  - logs and artifacts can be saved in any place depending on `MLFLOW_TRACKING_URI`
  - can log a model including all the execution environment via `mlflow.*.log_model()`
  - can register a logged model via `mlflow.register_model()`
- (remote server)
  - experiments
    - includes all the experiments, runs, logs, and artifacts including model themselves
  - models
    - shows only the registered model
    - they are just links to their own log page in the experiments tab.

# BentoML

## Entities

- model
  - can be a model in any of available type 
    - PyTorch, Diffusers, sklearn, PyTorch Lightning, ...
    - refer to https://docs.bentoml.org/en/latest/frameworks/index.html

- `bentoml.Model`
  - is a kind of packaged model for BentoML
  - can be created from your model of any types
    - e.g. for your `sklearn` model
      - run `bentoml.sklearn.save_model("iris_clf", clf)`
      - and it might be saved at `~/bentoml/iris_clf/kuo67urq6cgfuaav/`
  - includes
    - `model.yaml` 
    - `saved_model.pkl`
  - can be queried by
    - `bentoml models list`
    - `bentoml models get iris_clf:latest`
  - can be loaded by
    - `loaded_model = bentoml.sklearn.load_model("iris_clf:latest")`
- runnable
  - may contains one or more models
  - there are pre-built runnable classes that can be used out-of-box for some typical model types
  - or you can use your custom runnable class
    - refer to https://docs.bentoml.org/en/latest/concepts/runner.html#custom-model-runner
- runner
  - can be created by
    - `iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()`
  - can be initialized by
    - `iris_clf_runner.init_local()`
  - can be used by
    - `iris_clf_runner.predict.run([[5.9, 3.0, 5.1, 1.8]])`
- Bento
  - the distribution format
  - specified by `bentofile.yaml`
    - `service`
    - `labels`
    - `include`
      - specifies (python) source files
    - `python`
      - specifies dependencies
    - `models`
  - built by
    - `bentoml build`
  - required for a service to run
- service
  - it's defined by `service.py`
  - it can be run by `bentoml serve service.py:svc --reload`
  - A running service provides REST API
  - the configuration can be specified via an environment variable
    -  `BENTOML_CONFIG=/path/to/bentoml_configuration.yaml`
- containerized service
  - can be made by `bentoml build --containerize`
  - can be run by `docker run --rm -p 3000:3000 iris_classifier:latest`



## References

- https://docs.bentoml.com/en/latest/tutorial.html


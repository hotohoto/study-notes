# ONNX

## model

- can have a graph

## graph

- comutation dataflow graph
- can have nodes

## node (and built-in operators)

- a node is a call to an operator
- can have values or graph

### built-in operators

- https://github.com/onnx/onnx/blob/master/docs/Operators.md

## values

- value
  - `OrtValue` in Python
  - `OnnxValue` in Java
  - types
    - tensor
      - numpy array in Python
      - `OnnxTensor` in Java
      - can hold many values since it a "tensor"
      - can vectorize chunks even in Java as we do in python
    - sequence
      - non-fixed-length list of tensors
      - list of numpy arrays in Python
      - `OnnxSequence` in Java
        - At the moment, in Java, sequence can be used only as outputs
    - map
      - dictionary in Python
      - `OnnxMap` in Java
        - At the moment, in Java, map can be used only as outputs

## execution provider

- CPUExecutionProvider
- CUDA

## Snippets

```py
Dumper(..., default_flow_style=False)
```

```py
self.add_representer(bytes, Dumper.binary_representer)
...
def add_representer(self, data_type, representer):
    self.yaml_representers[data_type] = representer
...
def binary_representer(self, data):
    data = base64.encodebytes(data).decode('ascii')
    return self.represent_scalar('tag:yaml.org,2002:binary', data, style='|')
```

## References

- https://www.onnxruntime.ai/python/index.html
- https://github.com/onnx/onnx-tensorflow/blob/master/test/backend/test_model.py
- https://github.com/onnx/tensorflow-onnx


# DeepSpeed

- https://www.deepspeed.ai/
- https://github.com/microsoft/DeepSpeed



## Limitations

- it can train only a single main module
  - may not be applicable to some problems
- only the forward method of the main module should be called
  - e.g. the ema method within the model class may not be appropriate to be called



## Setup

Prepare CUDA

- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda
```



List and select CUDA

```shell
sudo update-alternatives --display cuda
sudo update-alternatives --config cuda
```



Install dependencies

```shell
pip install deepspeed
```



check further prerequisites required

```shell
ds_report
```



install further dependencies following the directions in the report e.g.

```shell
sudo apt install libaio-dev
```





## Training



configurations

- ZeRO stage 1
  - optimizer states
- ZeRO stage 2
  - optimizer states + gradients
- ZeRO stage 3
  - optimizer states + gradients + model parameters
- optimizer offload
  - ZeRO stage 2
  - with optimizer states + gradients offloaded
- param offload
  - ZeRO stage 3
  - with the model parameters offloaded



## Inference

- ZeRO stage 3 with ZeRO-Infinity 🤔



## Accelerator integration

- https://huggingface.co/docs/accelerate/usage_guides/deepspeed



Possible options:

- via the plugin without code change
  - no code change required

- via a config file with a bit of code change



## Analysis

### FusedAdam

- only for GPUs
- adamw mode is on by default
- doesn't change parameters??

### How DeepSpeed works

- It modifies `torch.Parameter` class
  - add some attributes

### Debug - "no attribute 'partition_numel'"

- parameters in model.parameters() are updated to be deepspeed parameters
- but parameters in optmizer.param_groups are not updated
- they must be the same objects but it seems not the case
  - since deepspeed assume only one model to be optimized
    - https://github.com/huggingface/accelerate/issues/253




```py
(engine.py)
DeepSpeedEngine.__init__()
- _configure_optimizer(client_optimizer=None)
  - if client_optimizer is None
    - basic_optimizer = self._configure_basic_optimizer(model_parameters)
      - optimizer = FusedAdam(adam_w_mode=True)
  - _configure_zero_optimizer(basic_optimizer)
    (stage3.py)
    - DeepSpeedZeroOptimizer_Stage3.__init__(init_optimizer=optimizer)
      - self.parameter_offload = DeepSpeedZeRoOffload(module=module,...)
        - self._convert_to_zero_parameters(ds_config, module, mpu)
          - Init(module=module,...)
            - self._convert_to_zero_parameters(module.parameters(recurse=True))
              - self._convert_to_deepspeed_param(param)
                - param.partition_numel = partition_numel
              - param.partition()
                - self._partition_param(param, has_been_updated=has_been_updated)
                  - param.ds_tensor = partitioned_tensor
      - self.trainable_param_groups = self._get_trainable_parameter_groups()
      - _create_fp16_partitions_with_defragmentation
        - _create_fp16_sub_groups(group)
          - params_group_numel = sum([param.partition_numel() for param in params_group])
        AttributeError: 'Parameter' object has no attribute 'partition_numel'
```



## References

- related papers

  - [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
    - TODO
  - [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
    - TODO
  - [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
    - NVMe
    - TODO

- etc

  - https://huggingface.co/docs/transformers/main_classes/deepspeed

    


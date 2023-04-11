# DeepSpeed

- https://www.deepspeed.ai/
- https://github.com/microsoft/DeepSpeed



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



Install dependencies

```shell
pip install deepspeed
```



Possible options:

- via a config file with a bit of code change
- via the plugin without code change



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

    




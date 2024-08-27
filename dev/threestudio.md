# threestudio

launch.py

- based on pytorch lightning
- train/validate/test/export
- load_config

## data

- `random-multiview-camera-datamodule`
  - light_sample_strategy
    - dreamfusion
      - random

## renderer

NerfVolumeRenderer

- estimator
- geometry
- material
- background



## geometry

ImplicitVolume

```
ImplicitVolume(
  (encoding): CompositeEncoding(
    (encoding): TCNNEncoding(
      (encoding): Encoding(n_input_dims=3, n_output_dims=32, seed=1337, dtype=torch.float32, hyperparams={'base_resolution': 16, 'hash': 'CoherentPrime', 'interpolation': 'Linear', 'log2_hashmap_size': 19, 'n_features_per_level': 2, 'n_levels': 16, 'otype': 'Grid', 'per_level_scale': 1.4472692012786865, 'type': 'Hash'})
    )
  )
  (density_network): VanillaMLP(
    (layers): Sequential(
      (0): Linear(in_features=32, out_features=64, bias=False)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=64, out_features=1, bias=False)
    )
  )
  (feature_network): VanillaMLP(
    (layers): Sequential(
      (0): Linear(in_features=32, out_features=64, bias=False)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=64, out_features=3, bias=False)
    )
  )
)
```



## materials

- `no-material`
  - rgb
- `diffuse-with-point-light-material`
  - 50%: random light assumed
  - 50% manual light assumed
- `hybrid-rgb-latent-material`
- `neural-radiance-material`
- `pbr-material`
  - a point light predicted

- `sd-latent-adapter-material`



## classes

### (system)

- BaseSystem(pl.LightningModule, Updateable, SaverMixin)
  - load_weights()
  - set_resume_status()
  - configure()
  - post_configure()
  - configure_optimizers()
- BaseLift3DSystem(BaseSystem)
  - geometry
  - material
  - background
  - renderer
  - on_fit_start()
  - on_test_end()
  - on_predict_start()
  - predict_step()
  - on_predict_epoch_end()
  - on_predict_end()
  - guidance_evaluation_save()
- Zero123(BaseLift3DSystem)

### (modules)

- BaseModule(nn.Module, Updateable)

- BaseGeometry(BaseModule)
- BaseExplicitGeometry(BaseGeometry)
  - configure()
- TetrahedraSDFGrid(BaseExplicitGeometry)
  - configure()
  - initialize_shape()
  - create_from()
  - export(0)
- BaseMaterial(BaseModule)
- PBRMaterial(BaseMaterial)
- BaseBackground(BaseModule)
- Renderer(BaseModule)
- StableDiffusionUnifiedGuidance(BaseModule)
- StableDiffusionVSDGuidance(BaseModule)
- Zero123UnifiedGuidance(BaseModule)

### (etc)

- IsosurfaceHelper
- MarchingTetrahedraHelper(IsosurfaceHelper)



### Configs

- https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md
- trial directory:
  - `[exp_root_dir]/[name]/[tag]@[timestamp]`
- https://github.com/threestudio-project/threestudio/blob/main/DOCUMENTATION.md
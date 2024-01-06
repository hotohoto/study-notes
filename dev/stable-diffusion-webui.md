# Stable diffusion web UI



## stable-diffusion-webui

https://github.com/AUTOMATIC1111/stable-diffusion-webui

- features
  - https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features
  - styles can be applied as weighted conditions
  - high-resolution fix
    - txt2img generates an LR image
    - img2img  generates an HR image from the LR image 
- ui
  - modules.ui
  - based on gradio
- api
  - modules.api
  - based on FastAPI



```sh
CUDA_VISIBLE_DEVICES=9 python launch.py --no-half --no-half-vae
```



### Call graph

(Generate clicked)

- modules.txt2img
  - p = modules.processing.StableDiffusionProcessingTxt2Img
  - modules.processing.process_images(p)
    - modules.processing.process_images_inner(p)
      - samples_ddim = p.sample()
        - p.sampler = sd_samplers.create_sampler(...)
          - module.sd_samplers.KDiffusionSampler(...)
            - (model_wrap_cfg): CFGDenoiser
              - (inner_model): CompVisVDenoiser
                - (inner_model): LatentDiffusion
        - p.sampler.sample(...)
      - ...
      - p.sd_model.decode_first_stage(x)
        - ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage()
          - self.first_stage_model.decode(z)





## AbdBarho/stable-diffusion-webui-docker

```shell
git clone git@github.com:AbdBarho/stable-diffusion-webui-docker.git
cd stable-diffusion-webui-docker

docker compose --profile download up --build
# wait until its done, then:
docker compose --profile [ui] up --build
# where [ui] is one of: invoke | auto | auto-cpu | comfy | comfy-cpu
#e.g.
docker compose --profile auto up --build
```



- https://github.com/AbdBarho/stable-diffusion-webui-docker
- https://github.com/AbdBarho/stable-diffusion-webui-docker/wiki/Setup



### Install ControlNet

```shell
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

git clone https://huggingface.co/lllyasviel/ControlNet-v1-1
sudo mv ControlNet-v1-1/* data/config/auto/extensions/sd-webui-controlnet/models/
sudo chown root:root data/config/auto/extensions/sd-webui-controlnet/models/*
```





## References

- https://github.com/civitai/civitai/wiki/How-to-use-models

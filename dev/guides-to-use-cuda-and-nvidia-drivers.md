# Guides to use CUDA and Nvidia drivers



## How Docker works with CUDA

<img src="https://docscontent.nvidia.com/dims4/default/5236c0f/2147483647/strip/true/crop/1020x969+0+0/resize/1020x969!/quality/90/?url=https%3A%2F%2Fk3-prod-nvidia-docs.s3.us-west-2.amazonaws.com%2Fbrightspot%2Fdita%2F0000018b-a729-dab5-abab-e77976e10000%2Fdeeplearning%2Fframeworks%2Fuser-guide%2Fgraphics%2Fsoftware_stack_zoom.png" alt="software_stack_zoom.png" style="zoom:67%;" />

- https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html



##### Files mounted on docker containers when `--gpus` is used:

- `/dev/nvidia0`, ...
- `/usr/lib/x86_64-linux-gnu/libcuda.so`, ...
- `nvidia-smi`
- (maybe there are more... 🤔)



## CUDA Components



- PyTorch
  - it has its own cuda runtime as its python dependencies
    - https://discuss.pytorch.org/t/install-pytorch-with-cuda-12-1/174294/2
    - check out
      - `pip freeze |grep nvidia`
      - `find venv |grep libcudart`
    - so it doesn't require `cuda-toolkit` to be installed manually for usual cases ⭐⭐⭐



- NVIDIA CUDA
  - API to use GPU for general purpose processing
  - it's contrast to OpenGL or Direct3D
  - installing CUDA seems to include installing cuda-toolkit, nvidia-driver, and so on



- cuda-toolkit
  - sub components
    - runtime
      - which includes `cudart.so`
    - the other libraries
    - compiler
    - documentations
  - for docker containers
    - it seems not mandatory to install cuda-toolkit manually unless you're going to compile codes
    - because required libraries are available via python dependencies
- nvidia-driver (cuda-driver)
  - most(?) recent drivers seem to support CUDA
  - components
    - cuda library
      - user mode library
        - which may mean `/usr/lib/x86_64-linux-gnu/libcuda.so.465.19.01`
      - backward compatible with respect to cuda-toolkit version
        - https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        - to get better compatibility ⭐⭐⭐
          - use higher NVIDIA driver version in the host machine
          - use lower cuda-toolkit version in the container
      - used by cuda runtime api
      - mounted when a docker container has been run with the `--gpus` option
      - https://stackoverflow.com/a/45431322/1874690
    - ...

- Hardware GPUs

  - `/dev/nvidia0`
    - mounted when a docker container has been run with the `--gpus` option

  - ...



- NVML
  - A C-based API for monitoring and managing various states of the NVIDIA GPU devices.
  - A python wrapper is also available
    - https://pypi.org/project/nvidia-ml-py/
  - https://developer.nvidia.com/nvidia-management-library-nvml
  - used by `nvidia-smi`



## Check GPU status

```bash
nvidia-smi -L
nvcc --version

# check the cuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# when nvidia-smi don't know which process is using GPU0
sudo fuser -v /dev/nvidia0
```



## Install a proper CUDA toolkit version

```bash
sudo apt install cuda-toolkit-11-1
```

- https://developer.nvidia.com/cuda-toolkit-archive
- currently, according to https://github.com/open-mmlab/mmcv#installation,
  - CUDA 11.3 looks promising for newer torch versions (torch 1.10 ~ 1.11)
  - CUDA 11.1 looks good for old torch versions  (torch 1.8 ~ 1.10)
  - CUDA 10.2 looks most popular but it's supported up to ubuntu 18.04 only



## Change CUDA version

https://stackoverflow.com/questions/45477133/how-to-change-cuda-version

```bash
sudo update-alternatives --query cuda
sudo ln -sfT /etc/alternatives/cuda /usr/local/cuda
sudo update-alternatives --config cuda
vi ~/.bashrc  # export PATH=$PATH:/usr/local/cuda/bin
```



## Change gcc/g++ version

https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11

sudo update-alternatives --config gcc
sudo update-alternatives --config g++
```



## CUDA hello world

```c
 #include "stdio.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>();
    return 0;
}
```

```bash
nvcc hello.cu -o hello
./hello
```

take a look at [this](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/) for more examples



## Install cuDNN

```bash
# https://installati.one/ubuntu/22.04/nvidia-cudnn/
sudo apt-get -y install nvidia-cudnn
```

- All versions: https://developer.nvidia.com/rdp/cudnn-archive



## Use only specific GPUs

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
```

(using a container - tentative, not tested)

```bash
nvidia-docker run -it --rm -v ~/workspace:/workspace -v ~/data/:/data --ipc=host --network=host --name=container_name -gpus '"device=6,7"' pytorch-1.11-11.3-8
```




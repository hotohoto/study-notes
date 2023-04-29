# GPU

## Check GPU status

```bash
nvidia-smi -L
nvcc --version

# check the cuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# when nvidia-smi don't know which process is using GPU0
sudo fuser -v /dev/nvidia0
```

## install a proper CUDA version

```bash
sudo apt install cuda-toolkit-11-1
```

- https://developer.nvidia.com/cuda-toolkit-archive
- currently, according to https://github.com/open-mmlab/mmcv#installation,
  - CUDA 11.3 looks promising for newer torch versions (torch 1.10 ~ 1.11)
  - CUDA 11.1 looks good for old torch versions  (torch 1.8 ~ 1.10)
  - CUDA 10.2 looks most popular but it's supported up to ubuntu 18.04 only

## change CUDA version

https://stackoverflow.com/questions/45477133/how-to-change-cuda-version

```bash
sudo update-alternatives --query cuda
sudo ln -sfT /etc/alternatives/cuda /usr/local/cuda
sudo update-alternatives --config cuda
vi ~/.bashrc  # export PATH=$PATH:/usr/local/cuda/bin
```

## change gcc/g++ version

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

## install cuDNN

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

[TOC]

# Carla-simulator



- 0.9.15
  - requires vulkan
    - wsl not supported
  - off-screen seems working with the official docker image, but it's hard to learn Carla without full visualization
    - TODO: try https://carla.readthedocs.io/en/latest/tuto_G_pygame/
- modes
  - (default with screen)
  - off-screen
    - https://carla.readthedocs.io/en/latest/adv_rendering_options/#off-screen-mode
  - no-rendering👎



## Run backend

- https://carla.readthedocs.io/en/latest/build_docker/

```bash
docker run -it --name=carla --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh  -RenderOffScreen

# run as a daemon
docker run -d --name=carla --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh  -RenderOffScreen

# run as a daemon with restart policy
docker run -d --name=carla --restart=unless-stopped --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh  -RenderOffScreen

# run as a daemon with a specific port and a specific GPU
docker run -d --name=carla --privileged --gpus device=0 --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw --shm-size=32gb \
carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh  -RenderOffScreen -carla-port=4000
```



## Setup for development

- Refer to
  - https://carla.readthedocs.io/en/latest/build_docker_unreal/
  - https://www.unrealengine.com/en-US/ue-on-github
- Create an account for Epic Games
- Connect the Epic Games account to the GitHub account
- Join the Epic Games organization on GitHub
  - You need to click the "Join" button in the invitation email
- Create a classic personal access token (PAT) at your GitHub developer settings
  - Requires `repo` and `read:org` scopes



### Custom build (based on 0.9.15)

(context/assumption)

- It requires 600G+ to build Carla
- The primary storage doesn't have enough free space
- The secondary storage have enough free space
- So I decide to run a builder container instead of building images directly
  - to mount carla repository in the secondary storage as the working directory
- Later, I guess, I can make an image from the files built



(Remarks)

- it requires uid=1000 for fbx sdk to be installed
  - so we don't use the uid/gid of the current host machine user
- Map folder has been moved to BitBucket



Prepare Build.Dockerfile (Mostly from Prerequisite.Dockerfile)

```dockerfile
FROM ubuntu:18.04

USER root

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update ; \
apt-get install -y wget software-properties-common && \
add-apt-repository ppa:ubuntu-toolchain-r/test && \
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - && \
apt-add-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" && \
apt-get update ; \
apt-get install -y build-essential \
clang-8 \
lld-8 \
g++-7 \
cmake \
ninja-build \
libvulkan1 \
python \
python-pip \
python-dev \
python3-dev \
python3-pip \
libpng-dev \
libtiff5-dev \
libjpeg-dev \
tzdata \
sed \
curl \
unzip \
autoconf \
libtool \
rsync \
libxml2-dev \
git \
sudo \
git-lfs \
aria2 && \
pip3 install -Iv setuptools==47.3.1 && \
pip3 install distro && \
update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180 && \
update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180 && \
git lfs install

RUN useradd -m carla && echo "carla:1234" | chpasswd
RUN usermod -aG sudo carla
USER carla

WORKDIR /workspaces/carla
ENV UE4_ROOT /workspaces/carla/UE4.26
```



```bash
git clone -b 0.9.15 https://github.com/carla-simulator/carla
cd carla
git clone --depth 1 -b carla git@github.com:CarlaUnreal/UnrealEngine.git
wget https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz -P Build/

docker build \
-t carla-build \
-f Build.Dockerfile \
.

docker run \
--rm \
-d \
--name=carla-build \
--gpus=all \
--privileged \
--net=host \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--shm-size=32gb \
-v $(pwd):/workspaces/carla \
carla-build \
sleep infinity

docker exec -it carla-build bash
```



(In the container)

```bash
sudo chown carla:carla -R /workspaces/carla
ln -s UnrealEngine $UE4_ROOT

cd $UE4_ROOT
nohup \
./Setup.sh && \
./GenerateProjectFiles.sh && \
make \
>> build.log 2>&1 &

cd /workspaces/carla

#./Update.sh
mkdir Unreal/CarlaUE4/Content

git -C Unreal/CarlaUE4/Content clone -b 0.9.15 https://bitbucket.org/carla-simulator/carla-content.git Carla 

nohup make CarlaUE4Editor >> carla_ue4_editor.log 2>&1 &
nohup make PythonAPI >> python_api.log 2>&1 &
nohup make build.utils >> build_utils.log 2>&1 &
nohup make package >> package.log 2>&1 &

# rm -r /workspaces/carla/carla/Dist
```



(out of container)

```bash
cd ..
tar xzf carla/Dist/CARLA_0.9.15.tar.gz| docker build -t carla-custom:0.9.15 -

# run from a custom image as a daemon with a specific port and a specific GPU
docker run -d --name=carla-custom-2000 --restart=unless-stopped --privileged --gpus device=0 \
--net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carla:0.9.15-custom \
/bin/bash ./CarlaUE4.sh -RenderOffScreen -carla-port=2000 -ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=0
```



## Ports

(Backend)

- carla-port
  - default: 2000
  - (aliases)
    - world-port
    - carla-rpc-port
    - carla-world-port

- carla-streaming-port
  - default: carla-port + 1

- carla-secondary-port
  - default: carla-port + 2


(Client)

- traffic manager rpc port: 8000



## Directories

- UE4.26 or UnrealEngine
  - custom unreal engine
- Unreal/CarlaUE4
  - carla unreal project
- Unreal/CarlaUE4/Content
  - map files are here??
- PythonAPI
  - contains server/client implementation
  - uses libcarla

- LibCarla
  - contains server/client implementation




- Build
  - carla makes files while building its own targets
- carla/Dist
- Util/DockerUtils/fbx/dependencies
  - ??



```
CARLA_DOCKER_UTILS_FOLDER=/workspaces/carla/Util/DockerUtils
FBX2OBJ_FOLDER=/workspaces/carla/Util/DockerUtils/fbx
FBX2OBJ_DIST=/workspaces/carla/Util/DockerUtils/dist
FBX2OBJ_DEP_FOLDER=/workspaces/carla/Util/DockerUtils/fbx/dependencies
FBX2OBJ_BUILD_FOLDER=/workspaces/carla/Util/DockerUtils/fbx/build
```

  



## Prepare new maps

```
docker exec -it -u root carla_hyheo apt-get install python3-pip libjpeg8 libtiff5 x11-xserver-utils x11-apps
docker exec -it -u root carla_hyheo xhost local:root

docker exec -it carla_hyheo python3 -m pip install carla
docker exec -it carla_hyheo python3 PythonAPI/util/config.py --map Town05
```



Alternatively, you may use the `carla` account.

```
docker exec -it -u root loving_lalande passwd carla
docker exec -it -u root loving_lalande apt-get install sudo
docker exec -it -u root loving_lalande adduser carla sudo
docker exec -it loving_lalande bash
python3 -m venv venv
. venv/bin/activate
pip install carla
./PythonAPI/util/config.py --map Town05
```





## Quickstart

- https://carla.readthedocs.io/en/latest/start_quickstart/

```sh
# run with display
docker run --restart=unless-stopped -d --privileged --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh

# run with the off-screen mode
docker run -d --restart=unless-stopped --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```



```
pip install carla
```



```py
import carla
import random

# Connect to the server and retrieve the world object
client = carla.Client('128.0.0.1', 2000)
client.load_world('Town03')
world = client.get_world()
print(world)

# Retrieve the spectator object
spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation

# Set the spectator with an empty transform
spectator.set_transform(carla.Transform())
# This will set the spectator at the origin of the map, with 0 degrees
# pitch, yaw and roll - a good way to orient yourself in the map

...
```



## Working on the source code

```bash
git clone git@github.com:carla-simulator/carla.git
cd carla
./Update.sh  # requires 20GB to download things inlcuding assets
```



## Example client code

```
TBD
```



## Trouble shooting

```
RuntimeError: trying to create rpc server for traffic manager; but the system failed to create because of bind error.
```

- local port 8000 or the custom traffic manager rpc port seems to be in use
- use another port when calling related function
  - `client.get_trafficmanager(backend_tm_port)`
  - `vehicle.set_autopilot(True, port)`



## Notes

- walkers require `WalkerAIController` to be animated
  - calling `walker_ai_controller.go_to_location()` might end up with a segmentation fault
    - especially if the target walker is not at a valid position
      - e.g. while the walker is in the air

  - Also, it looks unstable to restart/respawn walker ai controller for a walker instance

- vehicles require `set_autopilot(True, port)` to be animated

- when you run multiple clients and servers, each traffic manager port should be different from the others ⭐
  - they are going to be open in the client side host




## Sensors

### Distortion

- (analysis)
  - 0.9.15 기준(?) plumb bob model 을 쓰고 있다고 알려져있음.
    - radial + tangential distortion
- (references)
  - https://github.com/carla-simulator/carla/issues/3130
  - https://calib.io/blogs/knowledge-base/camera-models

### Color converters

#### carla.ColorConverter.LogarithmicDepth

https://github.com/carla-simulator/carla/blob/1ef3f55c9555de401681cb26ce87f81718943624/LibCarla/source/carla/image/ColorConverter.h#L21-L24





## Synchronous mode vs asynchronous mode

real time vs simulation time

simulation time-step:

- variable time-step
- fixed time-step

sub-stepping

- only for physics simulation
- turned on by default



mode:

- synchronous mode + variable time steps
  - not recommended
- synchronous mode + fixed time steps
  - pros
    - required for traffic manager ⭐
  - cons
    - too expensive to run animation
- asynchronous mode + variable time steps
  - Carla default
  - pros
    - good at animation
  - cons
    - bad at heavy sensor imaging
- asynchronous mode + fixed time steps
  - pros
    - good at animation
    - make it fastest up to the server performance
  - cons
    - bad at heavy sensor imaging



## Build on windows

- install cmake x64
  - https://cmake.org/download/



## Carla implementation details

- `FCarlaServer` -> `UCarlaEpisode` -> `UActorDispatcher` -> `ATagger`, `FActorRegistry`


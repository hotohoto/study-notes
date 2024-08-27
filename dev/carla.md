# Carla-simulator



- 0.9.15

  - requires vulkan
    - wsl not supported
  - off-screen seems working with the official docker image, but it's hard to learn Carla without full visualization
  - 👉
    - try it on ubuntu
    - try it with pygame
    - https://carla.readthedocs.io/en/latest/tuto_G_pygame/
- modes

  - (default with screen)

  - off-screen
    - https://carla.readthedocs.io/en/latest/adv_rendering_options/#off-screen-mode

  - no-rendering👎



## Setup

- https://carla.readthedocs.io/en/latest/build_docker/

```bash
docker pull carlasim/carla:0.9.15

# original commands as references
sudo docker run --privileged --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla:0.9.12 /bin/bash ./CarlaUE4.sh

sudo docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen

# modified commands
sudo docker run -itd --privileged --gpus '"device=0"' --net=host \
--name carla_hyheo \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
carlasim/carla:0.9.15 sleep infinity
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

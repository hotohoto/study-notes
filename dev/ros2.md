# ROS2

- (versions)
    - https://docs.ros.org/en/kilted/Releases.html

## TODO

- https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

## Installation

```bash
locale # check for UTF-8
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade

# check versions available (e.g. jazzy, kilted, ...)
sudo apt list |grep ros-|grep ros-base

# install 
sudo apt install ros-jazzy-desktop
sudo apt install ros-jazzy-ros-base
```

(test)

```bash
# talker
source /opt/ros/kilted/setup.bash
ros2 run demo_nodes_cpp talker
```

```bash
# listener
source /opt/ros/kilted/setup.bash
ros2 run demo_nodes_py listener
```

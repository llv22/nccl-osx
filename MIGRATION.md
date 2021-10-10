# Migration changes from NCCL 2.9,6-1 to NCCL-macOS 2.9.6 version

Compared via [2.9.6 to 2.5.8](https://github.com/llv22/nccl-osx/compare/dev-2.9.6-for-jax...dev-2.5.8-for-jax)

## 1. project settings

* .vscode/: add launch.json, tasks.json, settings.json and c_cpp_properties.json  
* .gitignore  
* README.md
* VERSION.txt  
* SNVML-LICENSE.txt  
* README_TO_RELEASE.md
* Makefie

## 2. independent test

copy from [Test folder of nccl2.5.8](https://github.com/llv22/nccl-osx/tree/dev-2.5.8-for-jax/test).

## 3. src code change

### 3.1 ext-net

* ext-net/dummy/Makefile

### 3.2 makefiles

* makefiles/common.mk
* makefiles/version.mk 

### 3.3 src

* src/Makefile
* src/include/nvml.h
* src/include/nvmlwrap.h
* src/nvml/nvidia-ml.cc
* src/symbol.exports

### 3.4 src(essential)

* src/bootstrap.cc
* src/debug.cc
* src/enqueue.cc
* src/graph/paths.cc
* src/graph/topo.cc
* src/group.cc
* src/include/cpuset.h
* src/include/debug.h
* src/include/shm.h
* src/include/socket.h
* src/misc/utils.cc
* src/transport/net_socket.cc
* src/transport/shm.cc

* src/init.cc
* src/misc/ibvwrap.cc
* src/misc/nvmlwrap.cc

## 4. mac device status

* reference from UESTC

(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/class  
0x060400  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/vendor  
0x8086  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/device  
0x6f08  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/subsystem_device  
0x0000  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/subsystem_vendor  
0x8086  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/max_link_speed  
8 GT/s  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/max_link_width  
16  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/../max_link_speed  
cat: '/sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/../max_link_speed': No such file or directory  
(base) administrator@UESTC:~/dl-frameworks/nccl-tests$ cat /sys/devices/pci0000:00/0000:00:03.0/pci_bus/0000:03/device/numa_node  
0  

(base) administrator@UESTC:~$ cat /sys/devices/system/node/node0/cpumap  
ffffff  

* reference from ubuntu-gpu

* macOS local simulation
/sys/class/0000:c4:00.0/  
sudo sh -c "echo '0x060400' >> class"  
sudo sh -c "echo '0x8086' >> vendor"  
sudo sh -c "echo '0x6f09' >> device" # different  
sudo sh -c "echo '0x0000' >> subsystem_device"  # different  
sudo sh -c "echo '0x8086' >> subsystem_vendor"  # different  
sudo sh -c "echo '8 GT/s' >> max_link_speed"  
sudo sh -c "echo '16' >> max_link_width"  
sudo sh -c "echo '0' >> numa_node"  

/sys/class/0000:45:00.0/  

sudo sh -c "echo '0x060400' >> class"  
sudo sh -c "echo '0x8086' >> vendor"  
sudo sh -c "echo '0x6f08' >> device" # different  
sudo sh -c "echo '0x0000' >> subsystem_device"  # different  
sudo sh -c "echo '0x8086' >> subsystem_vendor"  # different  
sudo sh -c "echo '8 GT/s' >> max_link_speed"  
sudo sh -c "echo '16' >> max_link_width"  
sudo sh -c "echo '0' >> numa_node"  

class  
vendor  
device  
subsystem_device  
subsystem_vendor  
max_link_speed  
max_link_width  
numa_node  

/sys/devices/system/node/node  

cpumap  

sudo mkdir -p /sys/devices/system/node/node0  
cd /sys/devices/system/node/node0  
sudo sh -c "echo 'ffffff' >> cpumap"  

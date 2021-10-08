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
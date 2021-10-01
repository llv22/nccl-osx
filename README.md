# NCCL on macOS with stub libraries

Optimized primitives for collective multi-GPU communication migrated to Mac OS X (10.13 - 10.13.6).
In order to make library and nccl-test compatible, each nccl library version will be unique mapped to nccl-test on macOS version. When you start your work from my version, please also keep in mind.

* [nccl-2.5.7-1-release](https://github.com/llv22/nccl-osx/tree/dev-2.5.7-for-jax) should be tested by [nccl-tests-v2.0.0 on mac](https://github.com/llv22/nccl-tests-macOS-cuda/tree/v2.0.0-built).

## Motivation

Why do we need NCCL on Mac OS X?  

* Because when using [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build), [Tom](https://github.com/TomHeaven) found some objection detection frameworks use distributed GPU training, which requires at least one distributed GPU backend functional. GPU backends of Pytorch consists of NCCL and GLOO. GLOO is dependent of NCCL. Thus, we need NCCL.
With the NCCL migration, GLOO can be compiled on Mac OS X and works fine as a distributed GPU backend of Pytorch. However, using of NCCL backend of Pytorch will fail at "unhandled system error" and I figured out the cause is many nccl codes heavily coupled with ubuntu system.
That is why I was motivated to migrate nccl library fully support for macOS, because it's a very good opportunity to learn how topo system of GPU resources are managed by NCCL framework and also understand dependencies between CUDA and NCCL.

* Another drive behind this project is supporting XLA in [JAX on macOS cuda](https://github.com/llv22/jax-macOS-cuda). One of the essential functionalities of JAX is to leverage XLA to bridge with NCCL under GPU. Currently Tom already finished one ported Tensorflow version in [tensorflow-osx-build](https://github.com/TomHeaven/tensorflow-osx-build) and the provided XLA in this module could greatly reduce my effort to enable XLA for JAX on macOS cuda. But I have to let XLA bridge on macOS cuda working through, which is another reason why I forked this repository for continuing with Tom's work.

## Status

This migration is fully functional after Orlando's effort for adding functionalities required on macOS. The relevant works are mainly listed as follow:

* construct libnvidia-smi.dylib and drafts basic functionalities. Fortunately, as my devices is GeoForce 1080 which can't support NVLink and Cuda exposes peer2peer API, I can manage to construct a simple wrapper library without including events monitor logic in NVML. One more clarification also raised up to Nvidia team, refer to <https://forums.developer.nvidia.com/t/how-to-use-cuda-api-to-implement-nvlink-query/189191>.
* migrate thread synchronization and shared memory allocation on macOS
* disable system check for /sys/dev check on macOS [in NCCL version > 2.5.7, the team has already started work to eliminate such dependency. I will follow-up later]

With built nccl library that passed all nccl test in [nccl-test on mac](https://github.com/llv22/nccl-tests-macOS-cuda), we can utilize it on macOS 10.13.6.ß

* It helps enable distributed GPU training for [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build) through GLOO backend.  

* Orlando could enable JAX working with XLA fully

Another important information is that Orlando changed several implementation of <https://github.com/TomHeaven/nccl-osx> and my next step is to migrate the latest release version [v2.11.4.-1](https://github.com/NVIDIA/nccl/releases/tag/v2.11.4-1). Anyone who tries to refine my work please checkout my master, which is ahead of tom's original version. About next release for v2.11.4-1, I may create a new branch develop checked-out from nccl main repository. As originated from different source code, the compatibility between 2.5.7 and 2.11.4 may not be guaranteed.  

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## What's inside

### 1, Available functionalities

At present, the library implements the following collectives operations, which have all passed nccl-test migrated from linux:

* all-reduce 【passed】
* all-gather 【passed】
* reduce-scatter 【passed】
* reduce     【passed】
* broadcast  【passed】

You can easily reproduce my result from <https://github.com/llv22/nccl-tests-macOS-cuda>. The latest test results with my two GeForce 1080, macOS 10.13.6, cuda10.1 + cudnn7.3.6 as follow:

```bash
===begin all_reduce_perf test===
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid  12978 on    Orlando device  0 [0x45] GeForce GTX 1080
#   Rank  1 Pid  12978 on    Orlando device  1 [0xc4] GeForce GTX 1080
#
#                                                     out-of-place                       in-place          
#       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2   float     sum    25.34    0.00    0.00  0e+00    25.23    0.00    0.00  0e+00
          16             4   float     sum    27.29    0.00    0.00  0e+00    25.11    0.00    0.00  0e+00
          32             8   float     sum    28.05    0.00    0.00  0e+00    25.50    0.00    0.00  0e+00
.......
    33554432       8388608   float     sum    39235    0.86    0.86  0e+00    39141    0.86    0.86  0e+00
    67108864      16777216   float     sum    77875    0.86    0.86  0e+00    77864    0.86    0.86  0e+00
   134217728      33554432   float     sum   154639    0.87    0.87  0e+00   157295    0.85    0.85  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.414483 
#

===begin all_gather_perf test===
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid  12979 on    Orlando device  0 [0x45] GeForce GTX 1080
#   Rank  1 Pid  12979 on    Orlando device  1 [0xc4] GeForce GTX 1080
#
#                                             out-of-place                       in-place          
#       size         count    type     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)             (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             1   float    21.03    0.00    0.00  0e+00    20.68    0.00    0.00  0e+00
          16             2   float    22.54    0.00    0.00  0e+00    20.58    0.00    0.00  0e+00
          32             4   float    23.51    0.00    0.00  0e+00    21.01    0.00    0.00  0e+00
.......
    33554432       4194304   float    22517    0.75    0.75  0e+00    22296    0.75    0.75  0e+00
    67108864       8388608   float    44479    0.75    0.75  0e+00    44221    0.76    0.76  0e+00
   134217728      16777216   float    87825    0.76    0.76  0e+00    85704    0.78    0.78  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.347488 
#

===begin broadcast_perf test===
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid  12991 on    Orlando device  0 [0x45] GeForce GTX 1080
#   Rank  1 Pid  12991 on    Orlando device  1 [0xc4] GeForce GTX 1080
#
#                                                     out-of-place                       in-place          
#       size         count    type    root     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2   float       0    21.42    0.00    0.00  0e+00    21.28    0.00    0.00  0e+00
          16             4   float       0    25.86    0.00    0.00  0e+00    20.27    0.00    0.00  0e+00
          32             8   float       0    23.29    0.00    0.00  0e+00    27.63    0.00    0.00  0e+00
.......
    33554432       8388608   float       0    22752    1.47    1.47  0e+00    22774    1.47    1.47  0e+00
    67108864      16777216   float       0    45533    1.47    1.47  0e+00    45543    1.47    1.47  0e+00
   134217728      33554432   float       0    91250    1.47    1.47  0e+00    91248    1.47    1.47  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.698235 
#

===begin reduce_perf test===
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid  13003 on    Orlando device  0 [0x45] GeForce GTX 1080
#   Rank  1 Pid  13003 on    Orlando device  1 [0xc4] GeForce GTX 1080
#
#                                                     out-of-place                       in-place          
#       size         count    type   redop    root     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                             (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2   float     sum       0    24.91    0.00    0.00  0e+00    36.37    0.00    0.00  0e+00
          16             4   float     sum       0    44.78    0.00    0.00  0e+00    26.80    0.00    0.00  0e+00
          32             8   float     sum       0    28.95    0.00    0.00  0e+00    24.89    0.00    0.00  0e+00
.......
    33554432       8388608   float     sum       0    22029    1.52    1.52  0e+00    21946    1.53    1.53  0e+00
    67108864      16777216   float     sum       0    43930    1.53    1.53  0e+00    43756    1.53    1.53  0e+00
   134217728      33554432   float     sum       0    87680    1.53    1.53  0e+00    90443    1.48    1.48  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.716373 
#

===begin reduce_scatter_perf test===
# nThread 1 nGpus 2 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 validation: 1 
#
# Using devices
#   Rank  0 Pid  13004 on    Orlando device  0 [0x45] GeForce GTX 1080
#   Rank  1 Pid  13004 on    Orlando device  1 [0xc4] GeForce GTX 1080
#
#                                                     out-of-place                       in-place          
#       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             1   float     sum    20.91    0.00    0.00  0e+00    20.95    0.00    0.00  0e+00
          16             2   float     sum    22.71    0.00    0.00  0e+00    20.94    0.00    0.00  0e+00
          32             4   float     sum    23.70    0.00    0.00  0e+00    21.06    0.00    0.00  0e+00
.......
    33554432       4194304   float     sum    22749    0.74    0.74  0e+00    22813    0.74    0.74  0e+00
    67108864       8388608   float     sum    45201    0.74    0.74  0e+00    47278    0.71    0.71  0e+00
   134217728      16777216   float     sum   168931    0.40    0.40  0e+00   184732    0.36    0.36  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.328477 
#
```

These operations are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

### 2, Milestone and next step

### Finished milestones

* [NCCL-macOS 2.5.7.1 release full](https://github.com/llv22/nccl-osx/releases/tag/nccl-macOS-2.5.7-1-release-full) is the first milestone which finishes the migration from linux to ubuntu system.

* [NCCL-macOS 2.5.7.2 release full](https://github.com/llv22/nccl-osx/releases/tag/nccl-macOS-2.5.7-2-release-full) fixes the issue of initializing nccl library on macOS during setup of the socket server.

### Next step

* Investigation with why IPv6 socket server isn't working on macOS via <https://github.com/llv22/nccl-osx/blob/dev-2.5.7-for-jax/test/ipv6_socket.cc>  
* check if nccl library needs another upgrade, as now Jax 0.2.14 is needed by AlphaFold2, which can provide the key representation of the 3D protein structures 

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.

## Build

To install NCCL on Mac OS X 10.13, first ensure Homebrew, XCode 9(.4.1) and CUDA-SDK (10.0 or 10.1) are properly installed.  

**Currently I used XCode 10.1, CUDA-SDK(10.1), makefile installed by Homebrew for compiling.**
**Base line version: <https://github.com/NVIDIA/nccl/blob/v2.5.7-1>**

Note: the official and tested builds of NCCL can be downloaded from: <https://developer.nvidia.com/nccl>. You can skip the following build steps if you choose to use the official builds.

1, For generating libraries based on gcc

To build the library :

```shell
cd nccl
make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :

```shell
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

2, For generating libraries based on clang++

To build the fat library compatible with *-std=c++14*, as this version is consistent with JAX:

```shell
make src.build CXX=clang++ -j8
make src.build CXX=clang++ PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7 -j12
make src.build CXX=clang++ PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7 TRACE=1 -j12 #enable trace for compiling, about 16 minutes
make src.build CXX=clang++ PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7 TRACE=1 NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61" -j12 #enable trace for compiling only for GTX1080, refer to https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/, about 5 minutes
```

## Install

1, install on default local /usr/local

```shell
make install
```

2, install on customized location $PREFIX

```shell
make install PREFIX=/usr/local/nccl-2.5.7
```

Install on local folder

```shell
make install PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7
```

## Todo

1, why additional dylib for macOS is needed?  

* refer to <https://developer.apple.com/forums/thread/26542>

2, NCCL 2.5.6 development guideline (Currently Orlando used 2.5.7-1 as based line, but documentation should keep consistent with 2.5.6)  

* As the first milestone aims at migration of source code of 2.5.7 and fully support nccl-tests, going through the development documentation of 2.5.7 becomes a must-have
* General development guide, refer to <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_256/nccl-developer-guide/docs/index.html>  
* About enabling tracing and logging for debugging purposes, refer to <https://github.com/NVIDIA/nccl/issues/197> and environment variable for controlling log of nccl 2.5.6 in <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_256/nccl-developer-guide/docs/env.htm>

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.

Migration to Mac OS X is done by [Orlando](https://github.com/llv22).

I sincerely appreciate the effort of [TomHeaven](https://github.com/TomHeaven/nccl-osx), but the implementations between Tom and I already diverged after 10-15 commits, especially when deciding to implement a simplified version of libnvidia-ml.dylib from scratch.

# NCCL-OSX

Optimized primitives for collective multi-GPU communication migrated to Mac OS X (10.13 - 10.13.6).

## Motivation

Why do we need NCCL on Mac OS X?  

* Because when using [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build), [Tom](https://github.com/TomHeaven) found some objection detection frameworks use distributed GPU training, which requires at least one distributed GPU backend functional. GPU backends of Pytorch consists of NCCL and GLOO. GLOO is dependent of NCCL. Thus, we need NCCL.
With the NCCL migration, GLOO can be compiled on Mac OS X and works fine as a distributed GPU backend of Pytorch. However, using of NCCL backend of Pytorch will fail at "unhandled system error" and I cannot figure out the cause.

* Another drive behind this project is supporting XLA in [JAX on macOS cuda](https://github.com/llv22/jax-macOS-cuda). One of the essential functionalities of JAX is to leverage XLA to bridge with NCCL under GPU. Currently Tom already finished one ported Tensorflow version in [tensorflow-osx-build](https://github.com/TomHeaven/tensorflow-osx-build) and the provided XLA in this module could greatly reduce my effort to enable XLA for JAX on macOS cuda. But I have to let XLA bridge on macOS cuda working through, which is another reason why I forked this repository for continuing with Tom's work.

## Status

Long story short, this migration is NOT fully functional, but  

* It helps enable distributed GPU training for [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build) through GLOO backend.  

* Orlando is working on how to enable it for JAX with XLA

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## What's inside

At present, the library implements the following collectives operations:

* all-reduce 【Not working】
* all-gather 【Not tested】
* reduce-scatter 【Working】
* reduce     【Not tested】
* broadcast  【Not tested】

These operations are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

## Requirements

NCCL requires at least CUDA 7.0 and Kepler or newer GPUs. For PCIe based platforms, best performance is achieved when all GPUs are located on a common PCIe root complex, but multi-socket configurations are also supported.

## Build

To install NCCL on Mac OS X 10.13, first ensure Homebrew, XCode 9(.4.1) and CUDA-SDK (10.0 or 10.1) are properly installed.  

**Currently I used XCode 10.1, CUDA-SDK(10.1), makefile installed by Homebrew for compiling.**

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
```

## Install

1, Simply run

```shell
make install
```

2, Simply run

```shell
make install PREFIX=/usr/local/nccl-2.5.7
```

## Tests

There are problems of compiling [nccl-tests](https://github.com/nvidia/nccl-tests) on Mac OS X.

In fact, not all functions of NCCL works on Mac OS X.

* This project is to help Pytorch-osx-build
* This project also aims at making JAX XLA bridge working
* Not all functionalities have been successfully passed

## Todo

1, whether dylib is needed?  
refer to <https://developer.apple.com/forums/thread/26542>

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.

Migration to Mac OS X is done by [TomHeaven](https://github.com/TomHeaven/nccl-osx) and [Orlando](https://github.com/llv22).

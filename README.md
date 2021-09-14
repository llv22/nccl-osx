# NCCL on macOS with stub libraries

Optimized primitives for collective multi-GPU communication migrated to Mac OS X (10.13 - 10.13.6).

## Motivation

Why do we need NCCL on Mac OS X?  

* Because when using [pytorch-osx-build](http://github.com/TomHeaven/pytorch-osx-build), [Tom](https://github.com/TomHeaven) found some objection detection frameworks use distributed GPU training, which requires at least one distributed GPU backend functional. GPU backends of Pytorch consists of NCCL and GLOO. GLOO is dependent of NCCL. Thus, we need NCCL.
With the NCCL migration, GLOO can be compiled on Mac OS X and works fine as a distributed GPU backend of Pytorch. However, using of NCCL backend of Pytorch will fail at "unhandled system error" and I figured out the cause is many nccl codes heavily coupled with ubuntu system.
That is why I was motivated to migrate nccl library fully support for macOS, because it's a very good opportunity to learn how topo system of GPU resources are managed by NCCL framework and also understand dependencies between CUDA and NCCL.

* Another drive behind this project is supporting XLA in [JAX on macOS cuda](https://github.com/llv22/jax-macOS-cuda). One of the essential functionalities of JAX is to leverage XLA to bridge with NCCL under GPU. Currently Tom already finished one ported Tensorflow version in [tensorflow-osx-build](https://github.com/TomHeaven/tensorflow-osx-build) and the provided XLA in this module could greatly reduce my effort to enable XLA for JAX on macOS cuda. But I have to let XLA bridge on macOS cuda working through, which is another reason why I forked this repository for continuing with Tom's work.

## Status

This migration is fully functional after Orlando's effort for adding functionalities required on macOS. The relevant works are mainly listed as follow:

* construct libnvidia-smi.dylib and drafts basic functionalities. Fortunately, as my devices is GeoForce 1080 which can't support NVLink and Cuda exposes peer2peer API, I can manage to construct a simple wrapper library without including events monitor logic in NVML. One more clarification also raised up to Nvidia team, refer to <>.
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
          64            16   float     sum    29.32    0.00    0.00  0e+00    26.27    0.00    0.00  0e+00
         128            32   float     sum    28.89    0.00    0.00  0e+00    26.74    0.00    0.00  0e+00
         256            64   float     sum    29.74    0.01    0.01  0e+00    27.76    0.01    0.01  0e+00
         512           128   float     sum    30.67    0.02    0.02  0e+00    28.16    0.02    0.02  0e+00
        1024           256   float     sum    32.41    0.03    0.03  0e+00    28.57    0.04    0.04  0e+00
        2048           512   float     sum    32.58    0.06    0.06  0e+00    31.10    0.07    0.07  0e+00
        4096          1024   float     sum    35.50    0.12    0.12  0e+00    34.06    0.12    0.12  0e+00
        8192          2048   float     sum    43.15    0.19    0.19  0e+00    42.45    0.19    0.19  0e+00
       16384          4096   float     sum    62.68    0.26    0.26  0e+00    63.39    0.26    0.26  0e+00
       32768          8192   float     sum    97.87    0.33    0.33  0e+00    96.21    0.34    0.34  0e+00
       65536         16384   float     sum    127.6    0.51    0.51  0e+00    127.8    0.51    0.51  0e+00
      131072         32768   float     sum    210.4    0.62    0.62  0e+00    209.0    0.63    0.63  0e+00
      262144         65536   float     sum    366.2    0.72    0.72  0e+00    415.4    0.63    0.63  0e+00
      524288        131072   float     sum    668.3    0.78    0.78  0e+00    719.0    0.73    0.73  0e+00
     1048576        262144   float     sum   1283.6    0.82    0.82  0e+00   1322.9    0.79    0.79  0e+00
     2097152        524288   float     sum   2559.9    0.82    0.82  0e+00   2499.4    0.84    0.84  0e+00
     4194304       1048576   float     sum   5010.3    0.84    0.84  0e+00   4964.4    0.84    0.84  0e+00
     8388608       2097152   float     sum   9936.4    0.84    0.84  0e+00   9858.0    0.85    0.85  0e+00
    16777216       4194304   float     sum    19765    0.85    0.85  0e+00    19624    0.85    0.85  0e+00
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
          64             8   float    24.27    0.00    0.00  0e+00    22.02    0.00    0.00  0e+00
         128            16   float    23.36    0.00    0.00  0e+00    20.71    0.00    0.00  0e+00
         256            32   float    23.51    0.01    0.01  0e+00    21.53    0.01    0.01  0e+00
         512            64   float    24.12    0.01    0.01  0e+00    21.37    0.01    0.01  0e+00
        1024           128   float    24.13    0.02    0.02  0e+00    22.27    0.02    0.02  0e+00
        2048           256   float    25.25    0.04    0.04  0e+00    23.55    0.04    0.04  0e+00
        4096           512   float    27.44    0.07    0.07  0e+00    25.20    0.08    0.08  0e+00
        8192          1024   float    33.42    0.12    0.12  0e+00    31.07    0.13    0.13  0e+00
       16384          2048   float    48.05    0.17    0.17  0e+00    45.42    0.18    0.18  0e+00
       32768          4096   float    65.57    0.25    0.25  0e+00    61.45    0.27    0.27  0e+00
       65536          8192   float    85.14    0.38    0.38  0e+00    84.49    0.39    0.39  0e+00
      131072         16384   float    131.6    0.50    0.50  0e+00    133.5    0.49    0.49  0e+00
      262144         32768   float    217.9    0.60    0.60  0e+00    217.4    0.60    0.60  0e+00
      524288         65536   float    401.9    0.65    0.65  0e+00    436.1    0.60    0.60  0e+00
     1048576        131072   float    738.2    0.71    0.71  0e+00    784.1    0.67    0.67  0e+00
     2097152        262144   float   1445.1    0.73    0.73  0e+00   1571.4    0.67    0.67  0e+00
     4194304        524288   float   2929.9    0.72    0.72  0e+00   2851.2    0.74    0.74  0e+00
     8388608       1048576   float   5874.5    0.71    0.71  0e+00   5672.3    0.74    0.74  0e+00
    16777216       2097152   float    11495    0.73    0.73  0e+00    11334    0.74    0.74  0e+00
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
          64            16   float       0    21.87    0.00    0.00  0e+00    20.11    0.00    0.00  0e+00
         128            32   float       0    22.79    0.01    0.01  0e+00    20.35    0.01    0.01  0e+00
         256            64   float       0    22.70    0.01    0.01  0e+00    20.51    0.01    0.01  0e+00
         512           128   float       0    23.16    0.02    0.02  0e+00    20.20    0.03    0.03  0e+00
        1024           256   float       0    24.34    0.04    0.04  0e+00    23.20    0.04    0.04  0e+00
        2048           512   float       0    24.99    0.08    0.08  0e+00    23.25    0.09    0.09  0e+00
        4096          1024   float       0    26.39    0.16    0.16  0e+00    24.49    0.17    0.17  0e+00
        8192          2048   float       0    28.81    0.28    0.28  0e+00    27.70    0.30    0.30  0e+00
       16384          4096   float       0    42.07    0.39    0.39  0e+00    41.29    0.40    0.40  0e+00
       32768          8192   float       0    57.42    0.57    0.57  0e+00    58.73    0.56    0.56  0e+00
       65536         16384   float       0    79.90    0.82    0.82  0e+00    80.69    0.81    0.81  0e+00
      131072         32768   float       0    129.2    1.01    1.01  0e+00    125.8    1.04    1.04  0e+00
      262144         65536   float       0    216.0    1.21    1.21  0e+00    221.1    1.19    1.19  0e+00
      524288        131072   float       0    399.5    1.31    1.31  0e+00    415.4    1.26    1.26  0e+00
     1048576        262144   float       0    792.7    1.32    1.32  0e+00    772.5    1.36    1.36  0e+00
     2097152        524288   float       0   1519.2    1.38    1.38  0e+00   1492.6    1.41    1.41  0e+00
     4194304       1048576   float       0   2913.4    1.44    1.44  0e+00   2896.9    1.45    1.45  0e+00
     8388608       2097152   float       0   5711.1    1.47    1.47  0e+00   5700.7    1.47    1.47  0e+00
    16777216       4194304   float       0    11343    1.48    1.48  0e+00    11402    1.47    1.47  0e+00
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
          64            16   float     sum       0    52.07    0.00    0.00  0e+00    46.06    0.00    0.00  0e+00
         128            32   float     sum       0    63.60    0.00    0.00  0e+00    27.16    0.00    0.00  0e+00
         256            64   float     sum       0    40.89    0.01    0.01  0e+00    29.73    0.01    0.01  0e+00
         512           128   float     sum       0    24.93    0.02    0.02  0e+00    25.44    0.02    0.02  0e+00
        1024           256   float     sum       0    59.34    0.02    0.02  0e+00    26.78    0.04    0.04  0e+00
        2048           512   float     sum       0    29.86    0.07    0.07  0e+00    34.96    0.06    0.06  0e+00
        4096          1024   float     sum       0    31.08    0.13    0.13  0e+00    28.93    0.14    0.14  0e+00
        8192          2048   float     sum       0    34.44    0.24    0.24  0e+00    36.83    0.22    0.22  0e+00
       16384          4096   float     sum       0    72.95    0.22    0.22  0e+00    50.74    0.32    0.32  0e+00
       32768          8192   float     sum       0    55.64    0.59    0.59  0e+00    54.91    0.60    0.60  0e+00
       65536         16384   float     sum       0    72.52    0.90    0.90  0e+00    74.77    0.88    0.88  0e+00
      131072         32768   float     sum       0    115.3    1.14    1.14  0e+00    117.7    1.11    1.11  0e+00
      262144         65536   float     sum       0    216.7    1.21    1.21  0e+00    203.0    1.29    1.29  0e+00
      524288        131072   float     sum       0    378.7    1.38    1.38  0e+00    382.1    1.37    1.37  0e+00
     1048576        262144   float     sum       0    775.7    1.35    1.35  0e+00    730.8    1.43    1.43  0e+00
     2097152        524288   float     sum       0   1444.6    1.45    1.45  0e+00   1448.5    1.45    1.45  0e+00
     4194304       1048576   float     sum       0   2921.8    1.44    1.44  0e+00   2835.5    1.48    1.48  0e+00
     8388608       2097152   float     sum       0   5521.7    1.52    1.52  0e+00   5539.3    1.51    1.51  0e+00
    16777216       4194304   float     sum       0    11017    1.52    1.52  0e+00    10986    1.53    1.53  0e+00
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
          64             8   float     sum    23.41    0.00    0.00  0e+00    21.03    0.00    0.00  0e+00
         128            16   float     sum    23.66    0.00    0.00  0e+00    21.03    0.00    0.00  0e+00
         256            32   float     sum    28.14    0.00    0.00  0e+00    21.60    0.01    0.01  0e+00
         512            64   float     sum    24.02    0.01    0.01  0e+00    21.51    0.01    0.01  0e+00
        1024           128   float     sum    24.94    0.02    0.02  0e+00    21.89    0.02    0.02  0e+00
        2048           256   float     sum    25.78    0.04    0.04  0e+00    23.86    0.04    0.04  0e+00
        4096           512   float     sum    27.87    0.07    0.07  0e+00    25.33    0.08    0.08  0e+00
        8192          1024   float     sum    32.26    0.13    0.13  0e+00    30.98    0.13    0.13  0e+00
       16384          2048   float     sum    46.74    0.18    0.18  0e+00    44.36    0.18    0.18  0e+00
       32768          4096   float     sum    67.25    0.24    0.24  0e+00    65.56    0.25    0.25  0e+00
       65536          8192   float     sum    84.43    0.39    0.39  0e+00    85.86    0.38    0.38  0e+00
      131072         16384   float     sum    126.6    0.52    0.52  0e+00    129.4    0.51    0.51  0e+00
      262144         32768   float     sum    216.3    0.61    0.61  0e+00    224.7    0.58    0.58  0e+00
      524288         65536   float     sum    390.9    0.67    0.67  0e+00    392.1    0.67    0.67  0e+00
     1048576        131072   float     sum    849.5    0.62    0.62  0e+00    770.5    0.68    0.68  0e+00
     2097152        262144   float     sum   1583.2    0.66    0.66  0e+00   1476.5    0.71    0.71  0e+00
     4194304        524288   float     sum   2996.6    0.70    0.70  0e+00   2923.8    0.72    0.72  0e+00
     8388608       1048576   float     sum   5745.5    0.73    0.73  0e+00   5868.6    0.71    0.71  0e+00
    16777216       2097152   float     sum    11665    0.72    0.72  0e+00    11539    0.73    0.73  0e+00
    33554432       4194304   float     sum    22749    0.74    0.74  0e+00    22813    0.74    0.74  0e+00
    67108864       8388608   float     sum    45201    0.74    0.74  0e+00    47278    0.71    0.71  0e+00
   134217728      16777216   float     sum   168931    0.40    0.40  0e+00   184732    0.36    0.36  0e+00
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0.328477 
#
```

These operations are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

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

1, why addtional dylib for macOS is needed?  

* refer to <https://developer.apple.com/forums/thread/26542>

2, NCCL 2.5.6 development guideline (Currently Orlando used 2.5.7-1 as based line, but documentation should keep consistent with 2.5.6)  

* As the first milestone aims at migration of source code of 2.5.7 and fully support nccl-tests, going through the development documentation of 2.5.7 becomes a must-have
* General development guide, refer to <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_256/nccl-developer-guide/docs/index.html>  
* About enabling tracing and logging for debugging purposes, refer to <https://github.com/NVIDIA/nccl/issues/197> and environment variable for controlling log of nccl 2.5.6 in <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_256/nccl-developer-guide/docs/env.htm>

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.

Migration to Mac OS X is done by [Orlando](https://github.com/llv22).

I sincerely appreciate the effort of [TomHeaven](https://github.com/TomHeaven/nccl-osx), but the implementations between Tom and I already diverged after 10-15 commits, especially when deciding to implement a simplified version of libnvidia-ml.dylib from scratch.

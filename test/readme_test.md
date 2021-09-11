# Local simple test to avoid loading from scratch

## 1, test case 1: compile nvlik

### 1.1 Documentation

* <https://codingbyexample.com/2020/09/14/p2p-memcpy-with-nvlink>
* <https://forums.developer.nvidia.com/t/the-simplest-way-to-compile-a-cu-file-looking-for-hello-world-example-for-compilation/21151>

### 1.2 Command line

```bash
nvcc -o nvlik nvlik.cu -I/usr/local/cuda/include -lcudart -L/usr/local/cuda/lib
```

## 2, test case 2: load symbol

refer to loadsym.cc, src/symbol.exports and src/nvml/nvidia-ml.cc

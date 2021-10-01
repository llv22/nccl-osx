# Release process of NCCL on macOS with stub libraries

## 1, building complete version

This version is ready for all architecture under Nvidia GPU.

```bash
version=2.5.7-2
make src.build CXX=clang++ PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7 TRACE=1 -j12
make lic
cp VERSION.txt build/
cd build
tar -czvf nccl-${version}-release-full.tar.gz *.txt lib/ include/
```

## 2, building arch61 version

This version is only for GeForce 1080 architecture under Nvidia GPU - arch61.

```bash
version=2.5.7-2
make src.build CXX=clang++ PREFIX=/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/drivers_mac/nccl-osx/nccl-2.5.7 TRACE=1 NVCC_GENCODE="-gencode=arch=compute_61,code=sm_61" -j12
make lic
cp VERSION.txt build/
cd build
tar -czvf nccl-${version}-release-arch61.tar.gz *.txt lib/ include/
```

## additional building

```bash
cd ext-net/dummy
make
```

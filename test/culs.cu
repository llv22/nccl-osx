#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <nvml.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <sstream>
#include "cuda.h"
#include "cuda_runtime_api.h"

using namespace std;

//https://dept-info.labri.fr/~thibault/tmp/culs.cu
static int coresPerMP(int major, int minor)
{
  switch (major) {
    case 1: return 8;
    case 2:
      switch (minor) {
        case 0: return 32;
        case 1: return 48;
      }
    case 3:
      return 192;
    case 5:
      return 128;
    case 6:
      switch (minor) {
        case 0: return 64;
        case 1:
        case 2: return 128;
      }
      break;
    case 7:
      return 64;
  }
  return -1;
}

int
main(int argc, char** argv)
{
  struct cudaDeviceProp prop;
  cudaError_t cures;
  int cnt;
  int dev;
  int version;
  nvmlReturn_t nvret;

// #if CUDART_VERSION >= 9000
//   nvmlInit();
// #endif
  cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);

  cures = cudaGetDeviceCount(&cnt);
  if (cures)
    exit(1);

  cudaDriverGetVersion(&version);
  printf("driver version %d\n", version);
  cudaRuntimeGetVersion(&version);
  printf("runtime version %d\n", version);

  for (dev = 0; dev < cnt; dev++) {
    printf("GPU%d\n", dev);
    cures = cudaGetDeviceProperties(&prop, dev);
    if (cures)
      exit(1);
    printf("%s\n", prop.name);
    printf("global: %0.3f GiB\n", (float) prop.totalGlobalMem / (1 << 30));
    printf("shared: %u KiB\n", prop.sharedMemPerBlock >> 10);
    printf("const:  %u KiB\n", prop.totalConstMem >> 10);
    printf("Clock %0.3fGHz\n", (float)(float)  prop.clockRate / (1 << 20));
    printf("%u MP\n", prop.multiProcessorCount);
    printf("capability %u.%u\n", prop.major, prop.minor);
    char busid[16];
    snprintf(busid, sizeof(busid), "%04x:%02x:%02x.%x", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, 0);
    printf("busid: %s\n", busid);
    printf("prop.pciBusID-%02x : prop.pciDeviceID-%02x.0\n", prop.pciBusID, prop.pciDeviceID);
    printf("%u cores\n", prop.multiProcessorCount*coresPerMP(prop.major, prop.minor));
    printf("async engine %d\n", prop.asyncEngineCount);
    printf("concurrentKernels %d\n", prop.concurrentKernels);
#if CUDART_VERSION >= 5050
    printf("streamPriorities %d\n", prop.streamPrioritiesSupported);
    printf("ECC %s\n", prop.ECCEnabled?"on":"off");
#endif
    stringstream ss;
    ss<<prop.pciDomainID<<":"<<prop.pciBusID<<":"<<prop.pciDeviceID;
    string pciBusId = ss.str();
    printf("composite pciBusId - %s\n", pciBusId.c_str());
/*
#if CUDART_VERSION >= 9000
    nvmlDevice_t device;
    nvret = nvmlDeviceGetHandleByPciBusId(busid, &device);
    if (nvret == NVML_SUCCESS) {
      unsigned long long value;
      unsigned int power;
      unsigned i, res;
      printf("got NVML device %p\n", device);
      nvret = nvmlDeviceGetTotalEnergyConsumption(device, &value);
      if (nvret == NVML_SUCCESS)
        printf("Energy since boot: %llumJ\n", value);
      else if (nvret != NVML_ERROR_NOT_SUPPORTED)
        printf("nvmlDeviceGetTotalEnergyConsumption err %d\n", nvret);
      nvret = nvmlDeviceGetPowerUsage(device, &power);
      if (nvret == NVML_SUCCESS)
        printf("Power: %umW\n", power);
      else if (nvret != NVML_ERROR_NOT_SUPPORTED)
        printf("nvmlDeviceGetPowerUsage err %d\n", nvret);
      for (i = 0; i < NVML_NVLINK_MAX_LINKS; i++) {
	printf("link %d\n", i);
	nvmlEnableState_t active;
	nvmlDeviceGetNvLinkState(device, i, &active);
	printf(" active: %d\n", active);
	if (active == NVML_FEATURE_ENABLED) {
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_P2P_SUPPORTED, &res);
	  printf(" p2p: %u\n", res);
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_SYSMEM_ACCESS, &res);
	  printf(" sysmem: %u\n", res);
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_P2P_ATOMICS, &res);
	  printf(" p2patom: %u\n", res);
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_SYSMEM_ATOMICS, &res);
	  printf(" sysmematom: %u\n", res);
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_SLI_BRIDGE, &res);
	  printf(" SLI: %u\n", res);
	  nvmlDeviceGetNvLinkCapability(device, i, NVML_NVLINK_CAP_VALID, &res);
	  printf(" valid: %u\n", res);
	  nvmlPciInfo_t pci;
	  nvmlDeviceGetNvLinkRemotePciInfo(device, i, &pci);
	  printf(" target: %04x:%02x:%02x (%s) %08x:%08x\n", pci.domain, pci.bus, pci.device, pci.busId, pci.pciDeviceId, pci.pciSubSystemId);
	}
      }
    }
    else
    {
      printf("nvml: %d\n", nvret);
    }
#endif
    printf("\n");
  */
  }

  float *A, *B;

  cures = cudaHostAlloc((void**) &A, 10*sizeof(*A), cudaHostAllocPortable | cudaHostAllocMapped);
  printf("%d\n", cures);

  *A = 42;
  B = (float*) malloc(10*sizeof(*B));
  *B = 43;

  cures = cudaHostRegister(A, 10*sizeof(*A), cudaHostRegisterPortable | cudaHostRegisterMapped);
  printf("register hostalloc: %d\n", cures);
  void *p;
  cures = cudaHostUnregister(A);
  printf("unregister hostalloc: %d\n", cures);
  cures = cudaHostGetDevicePointer(&p, A, 0);
  printf("%d %p\n", cures, p);

  cures = cudaHostRegister(B, 10*sizeof(*B), cudaHostRegisterPortable | cudaHostRegisterMapped);
  printf("register: %d\n", cures);
  cures = cudaHostGetDevicePointer(&p, B, 0);
  printf("%d %p\n", cures, p);
  cures = cudaHostUnregister(B);
  printf("unregister: %d\n", cures);

  return 0;
}
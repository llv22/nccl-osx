/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nvmlwrap.h"

#ifndef NVML_DIRECT
#include <dlfcn.h>
#include "core.h"

static enum { nvmlUninitialized, nvmlInitializing, nvmlInitialized, nvmlError } nvmlState = nvmlUninitialized;

/**
 * This section must be implemented in order to make nccl working.
 */
//0, mapping to nvmlErrorString in NVML API
static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);
//1, mapping to nvmlInit in NVML API
static nvmlReturn_t (*nvmlInternalInit)(void);
//2, mapping to nvmlShutdown in NVML API
static nvmlReturn_t (*nvmlInternalShutdown)(void);
//3, mapping to nvmlDeviceGetHandleByPciBusId in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device);
//4, mapping to nvmlDeviceGetNvLinkState in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkState)(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
//5, mapping to nvmlDeviceGetNvLinkRemotePciInfo in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
//6, mapping to nvmlDeviceGetNvLinkCapability in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkCapability)(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult);
//7, mapping to nvmlDeviceGetCudaComputeCapability in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetCudaComputeCapability)(nvmlDevice_t device, int* major, int* minor);

/**
 * This section is optional in order to make nccl working.
 */
//8, mapping to nvmlDeviceGetMinorNumber in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetMinorNumber)(nvmlDevice_t device, unsigned int* minorNumber);
//9, mapping to nvmlDeviceGetIndex in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetIndex)(nvmlDevice_t device, unsigned* index);
//10, mapping to nvmlDeviceGetHandleByIndex in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t* device);
//11, mapping to nvmlDeviceGetPciInfo in NVML API
static nvmlReturn_t (*nvmlInternalDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t* pci);

// Used to make the NVML library calls thread safe
pthread_mutex_t nvmlLock = PTHREAD_MUTEX_INITIALIZER;

//see: used in init.cc Line795, only have to load libnvidia-ml.so.1, not blocker
ncclResult_t wrapNvmlSymbols(void) {
  if (nvmlState == nvmlInitialized)
    return ncclSuccess;
  if (nvmlState == nvmlError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&nvmlState, nvmlUninitialized, nvmlInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (nvmlState == nvmlInitializing) pthread_yield_np();
    return (nvmlState == nvmlInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* nvmlhandle = NULL;
  void* tmp;
  void** cast;

  INFO(NCCL_ALL, "prepare to dlopen(\"libnvidia-ml.so.1\", RTLD_NOW)");
  nvmlhandle=dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvmlhandle) {
    WARN("Failed to open libnvidia-ml.so.1");
    goto teardown;
  }
  INFO(NCCL_ALL, "finish to dlopen(\"libnvidia-ml.so.1\", RTLD_NOW)");
#define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

#define LOAD_SYM_OPTIONAL(handle, symbol, funcptr) do {\
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      INFO(NCCL_INIT,"dlsym failed on %s, ignoring", symbol); \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByIndex", nvmlInternalDeviceGetHandleByIndex);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetPciInfo", nvmlInternalDeviceGetPciInfo);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetMinorNumber", nvmlInternalDeviceGetMinorNumber);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkState", nvmlInternalDeviceGetNvLinkState);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkRemotePciInfo", nvmlInternalDeviceGetNvLinkRemotePciInfo);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkCapability", nvmlInternalDeviceGetNvLinkCapability);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetCudaComputeCapability", nvmlInternalDeviceGetCudaComputeCapability);

  nvmlState = nvmlInitialized;
  return ncclSuccess;

teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceGetHandleByIndex = NULL;
  nvmlInternalDeviceGetPciInfo = NULL;
  nvmlInternalDeviceGetMinorNumber = NULL;
  nvmlInternalDeviceGetNvLinkState = NULL;
  nvmlInternalDeviceGetNvLinkRemotePciInfo = NULL;
  nvmlInternalDeviceGetNvLinkCapability = NULL;

  if (nvmlhandle != NULL) dlclose(nvmlhandle);
  nvmlState = nvmlError;
  INFO(NCCL_ALL, "failed to dlopen(\"libnvidia-ml.so.1\", RTLD_NOW)");
  return ncclSystemError;
}


//see: used in init.cc Line796
ncclResult_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalInit();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlInit() failed: %s",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in init.cc Line808
ncclResult_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalShutdown();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlShutdown() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in topo.cc Line574
ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByPciBusId() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in topo.cc Line250
ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  if (nvmlInternalDeviceGetNvLinkState == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkState(device, link, isActive), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkState() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in topo.cc Line254
ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  if (nvmlInternalDeviceGetNvLinkRemotePciInfo == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkRemotePciInfo(device, link, pci), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkRemotePciInfo() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in topo.cc Line246
ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkCapability(device, link, capability, capResult), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkCapability() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: used in topo.cc Line229
ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetCudaComputeCapability(device, major, minor), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetCudaComputeCapability() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: not referenced
ncclResult_t wrapNvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
  if (nvmlInternalDeviceGetMinorNumber == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetMinorNumber(device, minorNumber), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetMinorNumber() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: not referenced
ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetIndex(device, index), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: not referenced
ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByIndex(index, device), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

//see: not referenced
ncclResult_t wrapNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  if (nvmlInternalDeviceGetPciInfo == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetPciInfo(device, pci), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetPciInfo() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

#endif

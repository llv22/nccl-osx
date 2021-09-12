/*****************************************************************************************
 * Copyright (c) 2020-2021, Orlando's CUDA-based Simplification NVML. All rights reserved.
 *
 * See SNVML-LICENSE.txt for license information. 
 * Take care: nvidia-ml.cc is a separate part of source code with Apache License, while other
 * parts of nccl on macOS still adheres to Nvidia's declared license information illustrated
 * in LICENSE.txt
 *****************************************************************************************/
#ifndef NVIDIA_ML_H_
#define NVIDIA_ML_H_

#include "nvml.h"
#include "nccl.h"
#include "core.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <string>

using namespace std;

struct nvmlDevice_st
{
    cudaDeviceProp* prop; //!< cudaDevice properties
    int gpuIndex;         //!< The gpu index of CUDA
    int driverVer;        //!< The driver version of CUDA
    int runtimeVer;       //!< The runtime version of CUDA
};

/** Number of devices detected at component_init time */
static int device_count = 0;
/** NVML devices detected at component_init time, refer to https://medium.com/devoops-and-universe/monitoring-nvidia-gpus-cd174bf89311 */
static vector<nvmlDevice_t> devices;

// Symbolic name for visibility("default") attribute.
#define EXPORT __attribute__((visibility("default")))

/**
 * Mandatory to load && Used in implementation init.cc Line795, only have to load libnvidia-ml.so.1, not blocker
 * Exported function 0: nvmlErrorString.
 */ 
EXPORT                        // Symbol to export
const char* nvmlErrorString(nvmlReturn_t result) {
    if (NVML_SUCCESS == result)
    {
        return "The operation was successful";
    }
    else if (NVML_ERROR_UNINITIALIZED == result)
    {
        return "NVML was not first initialized with nvmlInit()";
    }
    else if (NVML_ERROR_INVALID_ARGUMENT == result)
    {
        return "A supplied argument is invalid";
    }
    else if (NVML_ERROR_NOT_SUPPORTED == result)
    {
        return "The requested operation is not available on target device";
    }
    else if (NVML_ERROR_NO_PERMISSION == result)
    {
        return "The current user does not have permission for operation";
    }
    else if (NVML_ERROR_ALREADY_INITIALIZED == result)
    {
        return "Deprecated: Multiple initializations are now allowed through ref counting";
    }
    else if (NVML_ERROR_NOT_FOUND == result)
    {
        return "A query to find an object was unsuccessful";
    }
    else if (NVML_ERROR_INSUFFICIENT_SIZE == result)
    {
        return "An input argument is not large enough";
    }
    else if (NVML_ERROR_INSUFFICIENT_POWER == result) {
        return "A device's external power cables are not properly attached";
    }
    else if (NVML_ERROR_DRIVER_NOT_LOADED == result) {
        return "NVIDIA driver is not loaded";
    }
    else if (NVML_ERROR_TIMEOUT == result) {
        return "User provided timeout passed";
    }
    else if (NVML_ERROR_IRQ_ISSUE== result) {
        return "NVIDIA Kernel detected an interrupt issue with a GPU";
    }
    else if (NVML_ERROR_LIBRARY_NOT_FOUND == result) {
        return "NVML Shared Library couldn't be found or loaded";
    }
    else if (NVML_ERROR_FUNCTION_NOT_FOUND == result) {
        return "Local version of NVML doesn't implement this function";
    }
    else if (NVML_ERROR_CORRUPTED_INFOROM == result) {
        return "infoROM is corrupted";
    }
    else if (NVML_ERROR_GPU_IS_LOST == result) {
        return "The GPU has fallen off the bus or has otherwise become inaccessible";
    }
    else if (NVML_ERROR_RESET_REQUIRED == result) {
        return "The GPU requires a reset before it can be used again";
    }
    else if (NVML_ERROR_OPERATING_SYSTEM == result) {
        return "The GPU control device has been blocked by the operating system/cgroups";
    }
    else if (NVML_ERROR_LIB_RM_VERSION_MISMATCH == result) {
        return "RM detects a driver/library version mismatch";
    }
    else if (NVML_ERROR_IN_USE == result) {
        return "An operation cannot be performed because the GPU is currently in use";
    }
    else if (NVML_ERROR_MEMORY == result) {
        return "Insufficient memory";
    }
    else if (NVML_ERROR_NO_DATA == result) {
        return "No data";
    }
    else if (NVML_ERROR_VGPU_ECC_NOT_SUPPORTED == result) {
        return "The requested vgpu operation is not available on target device, because ECC is enabled";
    }
    else if (NVML_ERROR_INSUFFICIENT_RESOURCES == result) {
        return "Ran out of critical resources, other than memory";
    }
    else if (NVML_ERROR_UNKNOWN == result) {
        return "An internal driver error occurred";
    }
    else {
        return "Unexpected error in NVML system domain";
    }
}

/**
 * Mandatory to load && Used in implementation init.cc Line796
 * Exported function 1: nvmlInit_v2, nccl internal symbol: wrapNvmlInit
 * This API will be remapped to nvmlInit in nvml.h.
 * Implementation refers to https://searchcode.com/codesearch/view/43324646/, which is implementation of papi.c devices management by uci/utk.edu
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlInit(void) {
    int cnt;
    int driverVer;
    int runtimeVer;

    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);

    cudaError_t cures = cudaGetDeviceCount(&cnt);
    if (cures){
        WARN("can't retrieve gpu device number, failed to successfully load driver");
        return NVML_ERROR_UNINITIALIZED;
    }
    cudaDriverGetVersion(&driverVer);
    cudaRuntimeGetVersion(&runtimeVer);
    for(int i = 0; i < cnt; i++) {
        struct cudaDeviceProp* _prop = new cudaDeviceProp;
        cudaError_t _cures = cudaGetDeviceProperties(_prop, i);
        if (cures) {
            INFO(NCCL_INIT, "gpu %d device can't be loaded successfully, skip", i);
        }
        else {
            //see: initialization of device and put into NVML list
            nvmlDevice_t devicePtr = new nvmlDevice_st;
            devicePtr->driverVer = driverVer;
            devicePtr->runtimeVer = runtimeVer;
            devicePtr->prop = _prop;
            devicePtr->gpuIndex = i;
            devices.push_back(devicePtr);
            device_count++;
        }
    }
    return NVML_SUCCESS;
}

void destory(nvmlDevice_t &device)
{
    delete(device);
}

/**
 * Mandatory to load && Used in implementation init.cc Line808
 * Exported function 2: nvmlShutdown, nccl internal symbol: wrapNvmlShutdown
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlShutdown(void) {
    if (devices.size() > 0)
    {
        for_each(devices.begin(), devices.end(), destory);
    }
    
    device_count = 0;
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in implementation topo.cc Line574
 * Exported function 3: nvmlDeviceGetHandleByPciBusId_v2, nccl internal symbol: wrapNvmlDeviceGetHandleByPciBusId
 * This API will be remapped to nvmlDeviceGetHandleByPciBusId in nvml.h.
 * pciBusId: domain:bus:device PCI identifier, please refer to https://developer.download.nvidia.com/compute/DevZone/NVML/doxygen/structnvml_pci_info__t.html for how to construct busId
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    nvmlReturn_t status = NVML_ERROR_NOT_FOUND;
    for(int i = 0; i < devices.size(); i++) {
        cudaDeviceProp* _cudaProp = devices[i]->prop;
        char _localBusId[16];
        snprintf(_localBusId, sizeof(_localBusId), "%04x:%02x:%02x.%x", _cudaProp->pciDomainID, _cudaProp->pciBusID, _cudaProp->pciDeviceID, 0);
        if(strcmp(pciBusId, _localBusId)) {
            device = &(devices[i]);
            status = NVML_SUCCESS;
        }
    }
    return status;
}

/**
 * Optional to load && Used in implementation topo.cc Line250
 * Exported function 4: nvmlDeviceGetNvLinkState, nccl internal symbol: wrapNvmlDeviceGetNvLinkState
 */
EXPORT                        // Symbol to export 
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {

}

/**
 * Optional to load && Used in implementation topo.cc Line254
 * Exported function 5: nvmlDeviceGetNvLinkRemotePciInfo_v2, nccl internal symbol: wrapNvmlDeviceGetNvLinkRemotePciInfo
 * This API will be remapped to nvmlDeviceGetNvLinkRemotePciInfo in nvml.h.
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {

}

/**
 * Optional to load && Used in implementation topo.cc Line246
 * Exported function 6: nvmlDeviceGetNvLinkCapability, nccl internal symbol: wrapNvmlDeviceGetNvLinkCapability
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                                   nvmlNvLinkCapability_t capability, unsigned int *capResult) {

}

/**
 * Mandatory to load && Used in implementation topo.cc Line229
 * Exported function 7: nvmlDeviceGetCudaComputeCapability, nccl internal symbol: wrapNvmlDeviceGetCudaComputeCapability
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {

}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 8: nvmlDeviceGetMinorNumber, nccl internal symbol: wrapNvmlDeviceGetMinorNumber
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber) {
    return NVML_ERROR_NOT_SUPPORTED;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 9: nvmlDeviceGetIndex, nccl internal symbol: wrapNvmlDeviceGetIndex
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    return NVML_ERROR_NOT_SUPPORTED;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 10: nvmlDeviceGetHandleByIndex_v2, nccl internal symbol: wrapNvmlDeviceGetHandleByIndex
 * This API will be remapped to nvmlDeviceGetHandleByIndex in nvml.h.
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    return NVML_ERROR_NOT_SUPPORTED;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 11: nvmlDeviceGetPciInfo_v3, nccl internal symbol: wrapNvmlDeviceGetPciInfo
 * This API will be remapped to nvmlDeviceGetPciInfo in nvml.h.
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    return NVML_ERROR_NOT_SUPPORTED;
}

#endif
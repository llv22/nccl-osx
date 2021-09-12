/*****************************************************************************************
 * Copyright (c) 2020-2021, Orlando's CUDA-based Simplification NVML. All rights reserved.
 *
 * See SNVML-LICENSE.txt for license information. 
 * Take care: nvidia-ml.cc is a separate part of source code with Apache License, while other
 * parts of nccl on macOS still adheres to Nvidia's declared license information illustrated
 * in LICENSE.txt
 *****************************************************************************************/

#include "nvml.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>

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
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlInit(void) {

}

/**
 * Mandatory to load && Used in implementation init.cc Line808
 * Exported function 2: nvmlShutdown, nccl internal symbol: wrapNvmlShutdown
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlShutdown(void) {

}

/**
 * Mandatory to load && Used in implementation topo.cc Line574
 * Exported function 3: nvmlDeviceGetHandleByPciBusId_v2, nccl internal symbol: wrapNvmlDeviceGetHandleByPciBusId
 * This API will be remapped to nvmlDeviceGetHandleByPciBusId in nvml.h.
 */
EXPORT                        // Symbol to export
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {

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
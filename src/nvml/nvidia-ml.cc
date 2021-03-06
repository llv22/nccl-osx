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

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline cudaError_t __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, cudaGetErrorString(err), file, line);
    }
    return err;
}

using namespace std;

struct nvmlDevice_st
{
    cudaDeviceProp *prop; //!< cudaDevice properties
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
EXPORT // Symbol to export
    const char *
    nvmlErrorString(nvmlReturn_t result)
{
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
    else if (NVML_ERROR_INSUFFICIENT_POWER == result)
    {
        return "A device's external power cables are not properly attached";
    }
    else if (NVML_ERROR_DRIVER_NOT_LOADED == result)
    {
        return "NVIDIA driver is not loaded";
    }
    else if (NVML_ERROR_TIMEOUT == result)
    {
        return "User provided timeout passed";
    }
    else if (NVML_ERROR_IRQ_ISSUE == result)
    {
        return "NVIDIA Kernel detected an interrupt issue with a GPU";
    }
    else if (NVML_ERROR_LIBRARY_NOT_FOUND == result)
    {
        return "NVML Shared Library couldn't be found or loaded";
    }
    else if (NVML_ERROR_FUNCTION_NOT_FOUND == result)
    {
        return "Local version of NVML doesn't implement this function";
    }
    else if (NVML_ERROR_CORRUPTED_INFOROM == result)
    {
        return "infoROM is corrupted";
    }
    else if (NVML_ERROR_GPU_IS_LOST == result)
    {
        return "The GPU has fallen off the bus or has otherwise become inaccessible";
    }
    else if (NVML_ERROR_RESET_REQUIRED == result)
    {
        return "The GPU requires a reset before it can be used again";
    }
    else if (NVML_ERROR_OPERATING_SYSTEM == result)
    {
        return "The GPU control device has been blocked by the operating system/cgroups";
    }
    else if (NVML_ERROR_LIB_RM_VERSION_MISMATCH == result)
    {
        return "RM detects a driver/library version mismatch";
    }
    else if (NVML_ERROR_IN_USE == result)
    {
        return "An operation cannot be performed because the GPU is currently in use";
    }
    else if (NVML_ERROR_MEMORY == result)
    {
        return "Insufficient memory";
    }
    else if (NVML_ERROR_NO_DATA == result)
    {
        return "No data";
    }
    else if (NVML_ERROR_VGPU_ECC_NOT_SUPPORTED == result)
    {
        return "The requested vgpu operation is not available on target device, because ECC is enabled";
    }
    else if (NVML_ERROR_INSUFFICIENT_RESOURCES == result)
    {
        return "Ran out of critical resources, other than memory";
    }
    else if (NVML_ERROR_UNKNOWN == result)
    {
        return "An internal driver error occurred";
    }
    else
    {
        return "Unexpected error in NVML system domain";
    }
}

/**
 * Mandatory to load && Used in implementation init.cc Line796
 * Exported function 1: nvmlInit_v2, nccl internal symbol: wrapNvmlInit
 * This API will be remapped to nvmlInit in nvml.h.
 * Implementation refers to https://searchcode.com/codesearch/view/43324646/, which is implementation of papi.c devices management by uci/utk.edu
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlInit(void)
{
    int cnt;
    int driverVer;
    int runtimeVer;

    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);

    cudaError_t cures = cudaGetDeviceCount(&cnt);
    if (cures)
    {
        WARN("can't retrieve gpu device number, failed to successfully load driver");
        return NVML_ERROR_UNINITIALIZED;
    }
    cudaDriverGetVersion(&driverVer);
    cudaRuntimeGetVersion(&runtimeVer);
    for (int i = 0; i < cnt; i++)
    {
        struct cudaDeviceProp *_prop = new cudaDeviceProp;
        cudaError_t _cures = cudaGetDeviceProperties(_prop, i);
        if (_cures)
        {
            INFO(NCCL_INIT, "gpu %d device can't be loaded successfully, skip", i);
        }
        else
        {
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

__inline__ void destory(nvmlDevice_t &device)
{
    INFO(NCCL_ALL, "destory device = %p", device);
    delete (device);
}

/**
 * Mandatory to load && Used in implementation init.cc Line808
 * Exported function 2: nvmlShutdown, nccl internal symbol: wrapNvmlShutdown
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlShutdown(void)
{
    // if (devices.size() > 0)
    // {
    //     for_each(devices.begin(), devices.end(), destory);
    // }
    // device_count = 0;
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in implementation topo.cc Line574
 * Exported function 3: nvmlDeviceGetHandleByPciBusId_v2, nccl internal symbol: wrapNvmlDeviceGetHandleByPciBusId
 * This API will be remapped to nvmlDeviceGetHandleByPciBusId in nvml.h.
 * pciBusId: domain:bus:device PCI identifier, please refer to https://developer.download.nvidia.com/compute/DevZone/NVML/doxygen/structnvml_pci_info__t.html for how to construct busId
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device)
{
    nvmlReturn_t status = NVML_ERROR_NOT_FOUND;
    for (int i = 0; i < devices.size(); i++)
    {
        cudaDeviceProp *_cudaProp = devices[i]->prop;
        char* _localBusId = new char[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
        snprintf(_localBusId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, "%04x:%02x:%02x.%x", _cudaProp->pciDomainID, _cudaProp->pciBusID, _cudaProp->pciDeviceID, 0);
        if (strcmp(pciBusId, _localBusId))
        {
            *device = devices[i];
            // INFO(NCCL_ALL, "return device %p with busId %s for gpu[%d]", *device, _localBusId, i);
            status = NVML_SUCCESS;
        }
    }
    return status;
}

__inline__ nvmlReturn_t queryOnlineDevice(nvmlDevice_t _d, nvmlDevice_t& realDevice) {
    if (devices.size() == 0) {
       return NVML_ERROR_NOT_FOUND;
    }
    vector<nvmlDevice_t>::iterator it = find(devices.begin(), devices.end(), _d);
    if (it == devices.end()) {
        WARN("target device %p can't be found", _d);
        return NVML_ERROR_GPU_IS_LOST;
    }
    realDevice = *it;
    return NVML_SUCCESS;
}

__inline__ nvmlReturn_t queryOnlineDeviceIndex(nvmlDevice_t _d, nvmlDevice_t& realDevice, int& index) {
    if (devices.size() == 0) {
       return NVML_ERROR_NOT_FOUND;
    }
    vector<nvmlDevice_t>::iterator it = find(devices.begin(), devices.end(), _d);
    index = it - devices.begin();
    if (it == devices.end()) {
        WARN("target device %p can't be found", _d);
        return NVML_ERROR_GPU_IS_LOST;
    }
    realDevice = *it;
    return NVML_SUCCESS;
}

__inline__ nvmlReturn_t queryOnlineDevice(nvmlDevice_t _d, unsigned int& index) {
    if (devices.size() == 0) {
       return NVML_ERROR_NOT_FOUND;
    }
    vector<nvmlDevice_t>::iterator it = find(devices.begin(), devices.end(), _d);
    if (it == devices.end()) {
        WARN("target device %p can't be found", _d);
        return NVML_ERROR_GPU_IS_LOST;
    }
    index = it - devices.begin();
    return NVML_SUCCESS;
}

/**
 * Optional to load && Used in implementation topo.cc Line250
 * Exported function 4: nvmlDeviceGetNvLinkState, nccl internal symbol: wrapNvmlDeviceGetNvLinkState
 * refer to https://on-demand.gputechconf.com/gtc/2017/presentation/s7128-davide-rossetti-how-to-enable.pdf
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive)
{    
    nvmlDevice_t realDevice = NULL;
    nvmlReturn_t status = queryOnlineDevice(device, realDevice);
    if (status != NVML_SUCCESS) {
        return status;
    }
    //see: https://on-demand.gputechconf.com/gtc/2017/presentation/s7128-davide-rossetti-how-to-enable.pdf
    if (link > NVML_NVLINK_MAX_LINKS) {
        WARN("nvlink only supports maximum value %d", NVML_NVLINK_MAX_LINKS);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    int can_access_peer_0_1;
    int gpuid_0 = realDevice->gpuIndex;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, link);
    if (can_access_peer_0_1) {
        *isActive = NVML_FEATURE_ENABLED;
    }
    else {
        *isActive = NVML_FEATURE_DISABLED;
    }
    return NVML_SUCCESS;
}

__inline__ void copy(cudaDeviceProp* &from, nvmlPciInfo_t* &to) {
    to->bus = from->pciBusID;
    to->pciDeviceId = from->pciDeviceID;
    to->domain = from->pciDomainID;
    to->pciDeviceId = from->pciDeviceID;
    snprintf(to->busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, "%04x:%02x:%02x.%x", from->pciDomainID, from->pciBusID, from->pciDeviceID, 0);
}

/**
 * Optional to load && Used in implementation topo.cc Line254
 * Exported function 5: nvmlDeviceGetNvLinkRemotePciInfo_v2, nccl internal symbol: wrapNvmlDeviceGetNvLinkRemotePciInfo
 * This API will be remapped to nvmlDeviceGetNvLinkRemotePciInfo in nvml.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci)
{
    nvmlDevice_t realDevice = NULL;
    nvmlReturn_t status = queryOnlineDevice(device, realDevice);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    if (link > NVML_NVLINK_MAX_LINKS) {
        WARN("nvlink only supports maximum value %d", NVML_NVLINK_MAX_LINKS);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (link > devices.size() || link == realDevice->gpuIndex) {
        WARN("device doesn't exists or don't make sense to copy by itself");
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    nvmlDevice_t target = devices[link];
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, realDevice->gpuIndex, target->gpuIndex);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, target->gpuIndex, realDevice->gpuIndex);
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        copy(target->prop, pci);
        return NVML_SUCCESS;
    }
    else {
        return NVML_ERROR_NOT_SUPPORTED;
    }
}

/**
 * Optional to load && Used in implementation topo.cc Line246
 * Exported function 6: nvmlDeviceGetNvLinkCapability, nccl internal symbol: wrapNvmlDeviceGetNvLinkCapability
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                  nvmlNvLinkCapability_t capability, unsigned int *capResult)
{
    nvmlDevice_t realDevice = NULL;
    if (link > NVML_NVLINK_MAX_LINKS) {
        WARN("nvlink only supports maximum value %d", NVML_NVLINK_MAX_LINKS);
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    nvmlReturn_t status = queryOnlineDevice(device, realDevice);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    *capResult = 0;
    cudaDeviceP2PAttr queryAttribute;
    if (NVML_NVLINK_CAP_P2P_SUPPORTED == capability) {
        queryAttribute = cudaDevP2PAttrAccessSupported;
    }
    else if (NVML_NVLINK_CAP_P2P_ATOMICS == capability) {
        queryAttribute = cudaDevP2PAttrNativeAtomicSupported;
    }
    else if (NVML_NVLINK_CAP_SYSMEM_ACCESS == capability) {
        queryAttribute = cudaDevP2PAttrCudaArrayAccessSupported;
    }
    else {
        INFO(NCCL_ALL, "can't support nvmlNvLinkCapability_t capability = %d, as limited capacity of NVLINK now", capability);
        return NVML_ERROR_NOT_SUPPORTED;
    }
    int value = 0;
    //see: for simplication purpose
    // cudaError_t result = checkCudaErrors(cudaDeviceGetP2PAttribute(&value, queryAttribute, realDevice->gpuIndex, link));
    cudaError_t result = cudaDeviceGetP2PAttribute(&value, queryAttribute, realDevice->gpuIndex, link);
    *capResult = value;
    if (result != cudaSuccess) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in implementation topo.cc Line229
 * Exported function 7: nvmlDeviceGetCudaComputeCapability, nccl internal symbol: wrapNvmlDeviceGetCudaComputeCapability
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor)
{
    nvmlDevice_t realDevice = NULL;
    nvmlReturn_t status = queryOnlineDevice(device, realDevice);
    if (NVML_SUCCESS == status) {
        *major = realDevice->prop->major;
        *minor = realDevice->prop->minor;
    }
    return status;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 8: nvmlDeviceGetMinorNumber, nccl internal symbol: wrapNvmlDeviceGetMinorNumber
 * Not supported as macOS don't have /dev/Nvidia[number] like linux
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int *minorNumber)
{
    return NVML_ERROR_NOT_SUPPORTED;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 9: nvmlDeviceGetIndex, nccl internal symbol: wrapNvmlDeviceGetIndex
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index)
{
    nvmlReturn_t status = queryOnlineDevice(device, *index);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 10: nvmlDeviceGetHandleByIndex_v2, nccl internal symbol: wrapNvmlDeviceGetHandleByIndex
 * This API will be remapped to nvmlDeviceGetHandleByIndex in nvml.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device)
{
    if (index >= devices.size()) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    device = &(devices[index]);
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Not Used in implementation
 * Exported function 11: nvmlDeviceGetPciInfo_v3, nccl internal symbol: wrapNvmlDeviceGetPciInfo
 * This API will be remapped to nvmlDeviceGetPciInfo in nvml.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci)
{
    nvmlDevice_t realDevice = NULL;
    nvmlReturn_t status = queryOnlineDevice(device, realDevice);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    copy(realDevice->prop, pci);
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 12: nvmlDeviceGetComputeRunningProcesses, pytorch internal symbol: deviceGetComputeRunningProcesses
 * This API will be remapped to deviceGetComputeRunningProcesses in Line 33 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetComputeRunningProcesses(nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos)
{
    // for macOS version, we can only retrieve the total memory consumption for certain device, leaving process id = -1 to indicate the aggregated processes
    // TODO: for further follow-ups, esp. how to trace memory consumption for individual process, orlando still need further investigations
    if (NULL == infoCount || NULL == infos){
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    
    nvmlDevice_t realDevice;
    int indexOfDevice;
    nvmlReturn_t status = queryOnlineDeviceIndex(device, realDevice, indexOfDevice);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_NOT_SUPPORTED;
    }
    cudaSetDevice(indexOfDevice);
    size_t memUsed, memTotal;
    cudaMemGetInfo(&memUsed, &memTotal);
    *infoCount = -1;
    infos->pid = -1;
    infos->usedGpuMemory = memUsed;
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 13: nvmlDeviceGetCount_v2, pytorch internal symbol: deviceGetCount_v2
 * This API will be remapped to deviceGetCount_v2 in Line 37 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 * API document in https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1ga93623b195bff04bbe3490ca33c8a42d
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    int cnt;

    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync);

    cudaError_t cures = cudaGetDeviceCount(&cnt);
    if (cures)
    {
        WARN("can't retrieve gpu device number, failed to successfully load driver");
        return NVML_ERROR_UNINITIALIZED;
    }
    *deviceCount = cnt;
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 14: nvmlDeviceGetHandleByIndex_v2, pytorch internal symbol: deviceGetHandleByIndex_v2
 * This API will be remapped to deviceGetHandleByIndex_v2 in Line 38 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device)
{
    if (device_count <= index || NULL == *device) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    device = &devices[index];
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 15: nvmlDeviceGetHandleByUUID, pytorch internal symbol: deviceGetHandleByUUID
 * This API will be remapped to deviceGetHandleByUUID in Line 42 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device)
{
    for (vector<nvmlDevice_t>::iterator it=devices.begin(); it!=devices.end(); it++)
    {
        if (strcmp((*it)->prop->uuid.bytes, uuid)) {
            device = &(*it);
            return NVML_SUCCESS;
        }
    }
    
    return NVML_ERROR_NOT_FOUND;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 16: nvmlDeviceGetP2PStatus, pytorch internal symbol: deviceGetP2PStatus
 * This API will be remapped to deviceGetP2PStatus in Line 46 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex,nvmlGpuP2PStatus_t *p2pStatus)
{
    if (p2pIndex == NVML_P2P_CAPS_INDEX_NVLINK || p2pIndex == NVML_P2P_CAPS_INDEX_UNKNOWN) {
        return NVML_ERROR_UNKNOWN;
    }
    if (p2pStatus == NULL) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    nvmlDevice_t realDevice0 = NULL;
    int realDevice0Index;
    nvmlReturn_t status = queryOnlineDeviceIndex(device1, realDevice0, realDevice0Index);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    nvmlDevice_t realDevice1 = NULL;
    int realDevice1Index;
    status = queryOnlineDeviceIndex(device2, realDevice1, realDevice1Index);
    if (status != NVML_SUCCESS) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (p2pIndex == NVML_P2P_CAPS_INDEX_READ ||p2pIndex == NVML_P2P_CAPS_INDEX_WRITE) {
        int can_access_peer_0_1 = 0;
        int can_access_peer_1_0 = 0;
        cudaDeviceCanAccessPeer(&can_access_peer_0_1, realDevice0Index, realDevice1Index);
        cudaDeviceCanAccessPeer(&can_access_peer_1_0, realDevice1Index, realDevice0Index);
        if (can_access_peer_0_1 && can_access_peer_1_0) {
            *p2pStatus = NVML_P2P_STATUS_OK;
        }
        else {
            *p2pStatus = NVML_P2P_STATUS_NOT_SUPPORTED;
        }
    }
    if (p2pIndex == NVML_P2P_CAPS_INDEX_ATOMICS) {
        int atomicAttribute;
        int can_access_peer_0_1 = 0;
        cudaError_t retValue = cudaDeviceGetP2PAttribute(&atomicAttribute, cudaDeviceP2PAttr::cudaDevP2PAttrNativeAtomicSupported, realDevice0Index, realDevice1Index);
        if (retValue == cudaSuccess) {
            can_access_peer_0_1 = 1;
        }
        int can_access_peer_1_0 = 0;  
        retValue = cudaDeviceGetP2PAttribute(&atomicAttribute, cudaDeviceP2PAttr::cudaDevP2PAttrNativeAtomicSupported, realDevice1Index, realDevice0Index);
        if (retValue == cudaSuccess) {
            can_access_peer_1_0 = 1;
        }
        if (can_access_peer_0_1 && can_access_peer_1_0) {
            *p2pStatus = NVML_P2P_STATUS_OK;
        }
        else {
            *p2pStatus = NVML_P2P_STATUS_NOT_SUPPORTED;
        }
    }
    if (p2pIndex == NVML_P2P_CAPS_INDEX_PROP) {
        //see: orlando supposed that it means cudaDevP2PAttrPerformanceRank
        int atomicAttribute;
        int can_access_peer_0_1 = 0;
        cudaError_t retValue = cudaDeviceGetP2PAttribute(&atomicAttribute, cudaDeviceP2PAttr::cudaDevP2PAttrPerformanceRank, realDevice0Index, realDevice1Index);
        if (retValue == cudaSuccess) {
            can_access_peer_0_1 = 1;
        }
        int can_access_peer_1_0 = 0;  
        retValue = cudaDeviceGetP2PAttribute(&atomicAttribute, cudaDeviceP2PAttr::cudaDevP2PAttrPerformanceRank, realDevice1Index, realDevice0Index);
        if (retValue == cudaSuccess) {
            can_access_peer_1_0 = 1;
        }
        if (can_access_peer_0_1 && can_access_peer_1_0) {
            *p2pStatus = NVML_P2P_STATUS_OK;
        }
        else {
            *p2pStatus = NVML_P2P_STATUS_NOT_SUPPORTED;
        }
    }
    return NVML_SUCCESS;
}

/**
 * Mandatory to load && Used in https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h
 * Exported function 17: nvmlDeviceGetUUID, pytorch internal symbol: deviceGetUUID
 * This API will be remapped to deviceGetUUID in Line 50 of https://github.com/pytorch/tensorpipe/blob/52791a2fd214b2a9dc5759d36725909c1daa7f2e/tensorpipe/common/dl.h.
 */
EXPORT // Symbol to export
    nvmlReturn_t
    nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length)
{
    if (length <= 0) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    for (vector<nvmlDevice_t>::iterator it=devices.begin(); it!=devices.end(); it++)
    {
        if (&device == &(*it)) {
            if (strlen((*it)->prop->uuid.bytes) < length) {
                return NVML_ERROR_INVALID_ARGUMENT;   
            }
            else {
                strncpy(uuid, (*it)->prop->uuid.bytes, length);
                return NVML_SUCCESS;
            }
        }
    }

    return NVML_ERROR_GPU_IS_LOST;
}

#endif
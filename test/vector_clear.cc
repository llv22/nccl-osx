#include <stdio.h>
#include <dlfcn.h>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

// Helper function 1
void destory(void* & methodHandle)
{
    methodHandle = NULL;
}

int main() {
    void* nvmlhandle = dlopen("../build/lib/libnvidia-ml.so.1", RTLD_NOW);
    if (!nvmlhandle) {
        printf("error for loading library\n");
        return -1;
    }
    const char *symbols[]   = {"nvmlInit",  "nvmlShutdown", "nvmlDeviceGetHandleByPciBusId", "nvmlDeviceGetIndex", "nvmlDeviceGetHandleByIndex", "nvmlErrorString",
    "nvmlDeviceGetPciInfo", "nvmlDeviceGetMinorNumber", "nvmlDeviceGetNvLinkState", "nvmlDeviceGetNvLinkRemotePciInfo", "nvmlDeviceGetNvLinkCapability", "nvmlDeviceGetCudaComputeCapability"};
    vector<void*> methods;
    for(int i = 0; i < 12; i++) {
      void* tmp = dlsym(nvmlhandle, symbols[i]);
      methods.push_back(tmp);                      
      if (tmp == NULL) {                                   
        printf("dlsym failed on %s - %s\n", symbols[i], dlerror());
        return -1;
      } 
    }
    for_each(methods.begin(), methods.end(), destory);
    dlclose(nvmlhandle);
    return 0;
}
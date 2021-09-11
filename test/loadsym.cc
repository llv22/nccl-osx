#include <stdio.h>
#include <dlfcn.h>

int main() {
    void* nvmlhandle = dlopen("../build/lib/libnvidia-ml.so.1", RTLD_NOW);
    if (!nvmlhandle) {
        printf("error for loading library\n");
        return -1;
    }
    const char* symbol = "nvmlInit";
    void* tmp = dlsym(nvmlhandle, symbol);                         
    if (tmp == NULL) {                                   
      printf("dlsym failed on %s - %s\n", symbol, dlerror());
      return -1;
    } 
    return 0;
}
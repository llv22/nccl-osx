#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>

using namespace std;

void initialize(int& value){
  value = 5;
}

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
    int input = 0;
    initialize(input);
    printf("final value of input : %d\n", input);
    char resolved_name[120];
    char *path = realpath("/sys/class/pci_bus/0000:45/../../0000:45:00.0", resolved_name);
    printf("real path = %s\n", path);
    printf("resolved_name = %s\n", resolved_name);

    const char* busId = "0000:45:00.0";
    char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
    memset(busPath+sizeof("/sys/class/")-1, 0, sizeof("0000:00:00.0"));
    memcpy(busPath+sizeof("/sys/class/")-1, busId, sizeof("0000:00:00.0")-1);
    int len = sizeof(busPath);
    char* _path = (char*)malloc(len * sizeof(char));
    memcpy(_path, busPath, len);
    printf("path :%s\n", _path);
    return 0;
}
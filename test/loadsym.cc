#include <stdio.h>
#include <dlfcn.h>

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
    int input = NULL;
    initialize(input);
    printf("final value of input : %d\n", input);
    return 0;
}
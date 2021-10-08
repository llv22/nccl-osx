#include <stdio.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <random>
#include <iomanip>

using std::cout;
using std::endl;
using std::setprecision;

using namespace std;

#define MAX_ASYNC_OPS 128
#if defined(__APPLE__) && defined(__MACH__)
#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_12
#include <pthread.h>
__inline__ pid_t gettid() {
  uint64_t tid64;
  pthread_threadid_np(NULL, &tid64);
  return (pid_t)tid64;
}
#else
#define _gettid() (pid_t) syscall(__NR_gettid)
#endif
#else
#define _gettid() (pid_t) syscall(SYS_gettid)
#endif

__thread pthread_t ncclGroupThreads[MAX_ASYNC_OPS];
__thread int ncclGroupIndex = 2;
__thread int ncclGroupMode = 0;
__thread int ncclGroupError = 0;

int FLOAT_MIN = 0;
int FLOAT_MAX = 1;

void *ncclAsyncThreadMain(void *args_)
{
    int *args = (int *)args_;
    int sleepTime = 2 + (double)((float)(rand()) / ((float)(RAND_MAX/(FLOAT_MAX - FLOAT_MIN))) * 15);
    sleep(sleepTime);
    printf("thread id: %d, sleep for %d, groupid : %d\n", gettid(), sleepTime, *args);
}

int main()
{
    int* groupIds = new int[ncclGroupIndex];
    for (int i = 0; i < ncclGroupIndex; i++)
    {
        groupIds[i] = i;
        pthread_create(ncclGroupThreads + i, NULL, ncclAsyncThreadMain, groupIds + i);
    }

    int done = ncclGroupIndex;
    while (done)
    {
        for (int i = 0; i < ncclGroupIndex; i++)
        {
            //see: orlando replace pthread_tryjoin_np with pthread_join, as no compatible version on mac for the former
            int err = pthread_join(ncclGroupThreads[i], NULL);
            if (err == EBUSY)
                continue;
            if (err != 0)
            {
                fprintf(stderr, "pthread_join called failed, return code: %d (translated to ncclSystemError)", err);
            }
            done--;
        }
    }
    return 0;
}
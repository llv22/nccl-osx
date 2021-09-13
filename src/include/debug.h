/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"

#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include "nccl_net.h"

//see: https://github.com/google/glog/issues/185
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

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...);

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
#define NOWARN(a, ret) do { \
  ncclDebugNoWarn = 1; \
  ret = a; \
  ncclDebugNoWarn = 0; \
} while (0)

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

#endif

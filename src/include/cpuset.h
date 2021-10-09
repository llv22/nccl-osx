/*************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CPUSET_H_
#define NCCL_CPUSET_H_

#include <stdio.h>

typedef struct cpu_set {
  uint32_t    count;
} cpu_set_t;
// Convert local_cpus, e.g. 0003ff,f0003fff to cpu_set_t

#if defined(__APPLE__) && defined(__MACH__)
/////////// for macos x ////////
#include <pthread.h>
#include <mach/thread_act.h>
#include <thread>
#include <sys/sysctl.h>

static inline void
CPU_AND(cpu_set_t *destset, cpu_set_t *srcset1, cpu_set_t *srcset2) { 
  destset->count = srcset1->count & srcset2->count; 
}
static inline int 
CPU_COUNT(cpu_set_t *cs) {return cs->count;}

static inline void
CPU_ZERO(cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_ZERO_S(int size, cpu_set_t *cs) { cs->count = 0; }

static inline void
CPU_SET(int num, cpu_set_t *cs) { cs->count |= (1 << num); }

static inline int
CPU_ISSET(int num, cpu_set_t *cs) { return (cs->count & (1 << num)); }

int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set)
{
  int32_t core_count = 0;
  // size_t  len = sizeof(core_count);
  /*int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, 0, 0);
  if (ret) {
    printf("error while get core count %d\n", ret);
    return -1;
  }*/
  core_count = std::thread::hardware_concurrency();
  cpu_set->count = 0;
  for (int i = 0; i < core_count; i++) {
    cpu_set->count |= (1 << i);
  }

  return 0;
}

int sched_setaffinity(pthread_t thread, size_t cpu_size,
                           cpu_set_t *cpu_set)
{
  thread_port_t mach_thread;
  int core = 0;

  for (core = 0; core < 8 * cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  thread_affinity_policy_data_t policy = { core };
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                    (thread_policy_t)&policy, 1);
  uint64_t thread_id;
  pthread_threadid_np(thread, &thread_id);
  return 0;
}

////////////////////////////////
#endif

static int hexToInt(char c) {
  int v = c - '0';
  if (v < 0) return -1;
  if (v > 9) v = 10 + c - 'a';
  if ((v < 0) || (v > 15)) return -1;
  return v;
}

#define CPU_SET_N_U32 (sizeof(cpu_set_t)/sizeof(uint32_t))

static ncclResult_t ncclStrToCpuset(const char* str, cpu_set_t* mask) {
  uint32_t cpumasks[CPU_SET_N_U32];
  int m = CPU_SET_N_U32-1;
  cpumasks[m] = 0;
  for (int o=0; o<strlen(str); o++) {
    char c = str[o];
    if (c == ',') {
      m--;
      cpumasks[m] = 0;
    } else {
      int v = hexToInt(c);
      if (v == -1) break;
      cpumasks[m] <<= 4;
      cpumasks[m] += v;
    }
  }
  // Copy cpumasks to mask
  for (int a=0; m<CPU_SET_N_U32; a++,m++) {
    memcpy(((uint32_t*)mask)+a, cpumasks+m, sizeof(uint32_t));
  }
  return ncclSuccess;
}

static ncclResult_t ncclCpusetToStr(cpu_set_t* mask, char* str) {
  int c = 0;
  uint8_t* m8 = (uint8_t*)mask;
  for (int o=sizeof(cpu_set_t)-1; o>=0; o--) {
    if (c == 0 && m8[o] == 0) continue;
    sprintf(str+c, "%02x", m8[o]);
    c+=2;
    if (o && o%4 == 0) {
      sprintf(str+c, ",");
      c++;
    }
  }
  str[c] = '\0';
  return ncclSuccess;
}

#endif

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

///// for mac os x
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <cmath>
#include <string.h>

using namespace std;

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

bool posix_fallocate(int fd, const int aLength)
{
  fstore_t store = {F_ALLOCATECONTIG, F_PEOFPOSMODE, 0, aLength};
  // Try to get a continuous chunk of disk space
  int ret = fcntl(fd, F_PREALLOCATE, &store);
  if (-1 == ret)
  {
    fprintf(stderr, "error reason : %s\n", strerror(errno));
    // OK, perhaps we are too fragmented, allocate non-continuous
    store.fst_flags = F_ALLOCATEALL;
    ret = fcntl(fd, F_PREALLOCATE, &store);
    if (-1 == ret)
    {
      fprintf(stderr, "error reason : %s\n", strerror(errno));
      struct stat mapstat;
      if (-1 != fstat(fd, &mapstat) && mapstat.st_size == 0)
      {
        //see: https://stackoverflow.com/questions/29682880/preallocate-storage-with-fcntl-doesnt-work-as-expected
        return 0 == ftruncate(fd, aLength);
      }
      return false;
    }
  }
  struct stat mapstat;
  if (-1 != fstat(fd, &mapstat) && mapstat.st_size == 0)
  {
    return 0 == ftruncate(fd, aLength);
  }
  return false;
}

static int shm_allocate(int fd, const int shmsize)
{
  bool succeed = posix_fallocate(fd, shmsize);
  if (!succeed)
  {
    errno = succeed;
    return -1;
  }
  return 0;
}

static int shm_map(int fd, const size_t shmsize, void **ptr)
{
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

int main()
{
  void *ptr = MAP_FAILED;
  int fd = -1;
  const char *shmname = "sh-recv-8b1d3bb2bb49be24-0-0-1";
  // see: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/getpagesize.3.html
  // https://developer.apple.com/forums/thread/672804
  // https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemAdvancedPT/MappingFilesIntoMemory/MappingFilesIntoMemory.html
  const int _shmsize = 9637888;
  const int pagesize = getpagesize();
  const int shmsize = ((int)ceil((double)_shmsize / (double)pagesize)) * pagesize;
  fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  int retcode = shm_allocate(fd, shmsize);
  if (retcode != 0)
  {
    printf("can't allocate memory\n");
    return -1;
  }
  retcode = shm_map(fd, shmsize, &ptr);
  printf("start shared memory mapping with fd=%d, shmsize=%d, ptr=%p, then shm_map result: %d\n", fd, shmsize, ptr, retcode);
  if (retcode != 0)
  {
    fprintf(stderr, "error reason : %s\n", strerror(errno));
    int outErr = errno;
    switch (outErr)
    {
    case EACCES:
      fprintf(stderr, "The flag PROT_READ was specified as part of the prot argument and fd was not open for reading\n");
      break;
    case EBADF:
      fprintf(stderr, "The fd argument is not a valid open file descriptor\n");
      break;
    case EINVAL:
      fprintf(stderr, "MAP_FIXED was specified and the addr argument was not page aligned, or part of the desired address space resides out of the valid address space for a user process; flags does not include either MAP_PRIVATE or MAP_SHARED; The len argument was negative or zero; The offset argument was not page-aligned based on the page size as returned by getpagesize(3)\n");
      break;
    case ENODEV:
      fprintf(stderr, "MAP_ANON has not been specified and the file fd refers to does not support mapping\n");
      break;
    case ENOMEM:
      fprintf(stderr, "MAP_FIXED was specified and the addr argument was not available\n");
      break;
    case ENXIO:
      fprintf(stderr, "Addresses in the specified range are invalid for fd");
      break;
    case EOVERFLOW:
      fprintf(stderr, "Addresses in the specified range exceed the maximum offset set for fd");
      break;
    default:
      fprintf(stderr, "unknown reason\n");
      break;
    }
  }
  shm_unlink(shmname);
  if (ptr != MAP_FAILED)
  {
    munmap(ptr, shmsize);
  }
  close(fd);
  fd = -1;
  printf("inline function ttid: %d\n", gettid());
  return 0;
}
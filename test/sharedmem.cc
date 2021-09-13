#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>

///// for mac os x
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>

using namespace std;

bool posix_fallocate(int fd, const int aLength)
{
  fstore_t store = {F_ALLOCATECONTIG, F_PEOFPOSMODE, 0, aLength};
  // Try to get a continuous chunk of disk space
  int ret = fcntl(fd, F_PREALLOCATE, &store);
  if (-1 == ret)
  {
    // OK, perhaps we are too fragmented, allocate non-continuous
    store.fst_flags = F_ALLOCATEALL;
    ret = fcntl(fd, F_PREALLOCATE, &store);
    if (-1 == ret)
      return false;
  }
  return 0 == ftruncate(fd, aLength);
}

static int shm_allocate(int fd, const int shmsize)
{
  int err = posix_fallocate(fd, shmsize);
  if (err)
  {
    errno = err;
    return -1;
  }
  return 0;
}

static int shm_map(int fd, const int shmsize, void **ptr)
{
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

int main()
{
  void **ptr;
  int fd = -1;
  const char *shmname = "sh-recv-8b1d3bb2bb49be24-0-0-1";
  const int shmsize = 9637888;
  fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  int retcode = shm_allocate(fd, shmsize);
  if (retcode != 0)
  {
    printf("can't allocate memory\n");
    return -1;
  }
  retcode = shm_map(fd, shmsize, ptr);
  printf("start shared memory mapping with fd=%d, shmsize=%d, ptr=%p, then shm_map result: %d\n", fd, shmsize, ptr, retcode);
  switch (errno)
  {
  case EACCES:
    printf("The flag PROT_READ was specified as part of the prot argument and fd was not open for reading\n");
    break;
  case EBADF:
    printf("The fd argument is not a valid open file descriptor\n");
    break;
  case EINVAL:
    printf("MAP_FIXED was specified and the addr argument was not page aligned, or part of the desired address space resides out of the valid address space for a user process; flags does not include either MAP_PRIVATE or MAP_SHARED; The len argument was negative or zero; The offset argument was not page-aligned based on the page size as returned by getpagesize(3)\n");
    break;
  case ENODEV:
    printf("MAP_ANON has not been specified and the file fd refers to does not support mapping\n");
    break;
  case ENOMEM:
    printf("MAP_FIXED was specified and the addr argument was not available\n");
    break;
  case ENXIO:
    printf("Addresses in the specified range are invalid for fd");
    break;
  case EOVERFLOW:
    printf("Addresses in the specified range exceed the maximum offset set for fd");
    break;
  default:
    printf("unknown reason\n");
    break;
  }
  close(fd);
  fd = -1;
  return 0;
}
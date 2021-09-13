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
    if(-1 == ret){
    // OK, perhaps we are too fragmented, allocate non-continuous
    store.fst_flags = F_ALLOCATEALL;
    ret = fcntl(fd, F_PREALLOCATE, &store);
    if (-1 == ret)
      return false;
  }
  return 0 == ftruncate(fd, aLength);
}

static int shm_allocate(int fd, const int shmsize) {
  int err = posix_fallocate(fd, shmsize);
  if (err) { errno = err; return -1; }
  return 0;
}

static int shm_map(int fd, const int shmsize, void** ptr) {
  *ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? -1 : 0;
}

int main()
{
    void** ptr;
    int fd = -1;
    const char* shmname = "sh-recv-8b1d3bb2bb49be24-0-0-1";
    const int shmsize = 9637888;
    fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    shm_allocate(fd, shmsize);
    printf("fd=%d, shmsize=%d, ptr=%p\n", fd, shmsize, ptr);
    shm_map(fd, shmsize, ptr);
    close(fd);
    fd = -1;
    return 0;
}
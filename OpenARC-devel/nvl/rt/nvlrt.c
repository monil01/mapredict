// Fully home-grown version of the NVL runtime. See ../README for details.

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const unsigned HEAP_SIZE_INIT_MIN = 100;

typedef struct {
  char *name;
  int fd;
  unsigned long size;
  void *addr;
} nvlrt_heap_t;

#define nvlrt_perror(Msg) perror("NVL-RT: " Msg)

static nvlrt_heap_t *nvlrt_create_or_open(
  const char *name, bool create, size_t initSize, unsigned long mode);

// The type of the initSize and mode parameters for nvlrt_create and
// nvlrt_create_or_open must be kept in sync with the compiler front end's
// type for those parameters to __builtin_nvl_create. See comments there.
nvlrt_heap_t *nvlrt_create(const char *name, size_t initSize,
                           unsigned long mode)
{
  return nvlrt_create_or_open(name, true, initSize, mode);
}

nvlrt_heap_t *nvlrt_open(const char *name) {
  return nvlrt_create_or_open(name, false, 0/*ignored*/, 0/*ignored*/);
}

static nvlrt_heap_t *nvlrt_create_or_open(
  const char *name, bool create, size_t initSize, unsigned long mode)
{
  // Allocate heap data structure and set file name.
  nvlrt_heap_t *heap = malloc(sizeof(nvlrt_heap_t));
  if (heap == NULL)
    return NULL;
  heap->name = malloc(strlen(name) + 1);
  if (heap->name == NULL) {
    free(heap);
    return NULL;
  }
  strcpy(heap->name, name);

  // Open file.
  heap->fd = create
             ? open(name, O_RDWR|O_CREAT|O_EXCL, (mode_t)mode)
             : open(name, O_RDWR);
  if (heap->fd == -1) {
    free(heap->name);
    free(heap);
    return NULL;
  }

  // Get its size and initialize it if it's a new file.
  if (create) {
    heap->size = initSize < HEAP_SIZE_INIT_MIN ? HEAP_SIZE_INIT_MIN
                                               : initSize;
    if (-1 == lseek(heap->fd, heap->size, SEEK_SET)) {
      close(heap->fd);
      free(heap->name);
      free(heap);
      return NULL;
    }
    if (-1 == write(heap->fd, "", 1)) {
      close(heap->fd);
      free(heap->name);
      free(heap);
      return NULL;
    }
  }
  else {
    struct stat statInit;
    if (-1 == fstat(heap->fd, &statInit)) {
      close(heap->fd);
      free(heap->name);
      free(heap);
      return NULL;
    }
    heap->size = statInit.st_size;
  }

  // Map it to our address space.
  heap->addr = mmap(0, heap->size, PROT_READ|PROT_WRITE, MAP_SHARED,
                    heap->fd, 0);
  if (heap->addr == MAP_FAILED) {
    close(heap->fd);
    free(heap->name);
    free(heap);
    return NULL;
  }

  return heap;
}

void nvlrt_close(nvlrt_heap_t *heap) {
  // The order of munmap and close doesn't seem to matter.
  if (-1 == munmap(heap->addr, heap->size)) {
    nvlrt_perror("munmap failed");
    exit(EXIT_FAILURE);
  }
  if (-1 == close(heap->fd)) {
    nvlrt_perror("close failed");
    exit(EXIT_FAILURE);
  }
  free(heap->name);
  free(heap);
}

const char *nvlrt_get_name(nvlrt_heap_t *heap) {
  return heap->name;
}

// This will go away once load/store is implemented.
void *nvlrt_get_addr(nvlrt_heap_t *heap) {
  return heap->addr;
}

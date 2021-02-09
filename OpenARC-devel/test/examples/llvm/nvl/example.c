// This example uses the under-developed libnvlrt and thus is very minimal.
// Use example-pmemobj.c instead.

#include <nvl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

// This will go away once load/store is implemented.
void *nvlrt_get_addr(nvl_heap_t *heap);

void writeNVL(const char *name) {
  nvl_heap_t *heap = nvl_open(name);
  if (heap == NULL && errno == ENOENT)
    heap = nvl_create(name, 0, 0600);
  void *addr = nvlrt_get_addr(heap);
  strcpy(addr, "hello world");
  printf("%s: wrote: %s\n", nvl_get_name(heap), addr);
  nvl_close(heap);
}

void readNVL(const char *name) {
  nvl_heap_t *heap = nvl_open(name);
  void *addr = nvlrt_get_addr(heap);
  printf("%s: read: %s\n", nvl_get_name(heap), addr);
  nvl_close(heap);
}

int main() {
  writeNVL("example.nvl");
  readNVL("example.nvl");
  return 0;
}

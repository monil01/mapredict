// This example uses libnvlrt-pmemobj, which is the version of the NVL
// runtime on which we are currently focusing our development.  To build it,
// you must define PMEM_LIBDIR in make.header.

#include <nvl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define STR "hello world"
void writeNVL(const char *name) {
  nvl_heap_t *heap = nvl_open(name);
  if (heap == NULL && errno == ENOENT)
    heap = nvl_create(name, 0, 0600);
  nvl char *root = nvl_alloc_nv(heap, 1, char);
  nvl_set_root(heap, root);
  for (int i = 0; i < sizeof STR; ++i)
    root[i] = STR[i];
  printf("%s: wrote: %s\n", nvl_get_name(heap), nvl_bare_hack(root));
  nvl_close(heap);
}

void readNVL(const char *name) {
  nvl_heap_t *heap = nvl_open(name);
  nvl char *root = nvl_get_root(heap, char);
  char str[sizeof STR];
  for (int i = 0; i < sizeof STR; ++i)
    str[i] = root[i];
  printf("%s: read: %s\n", nvl_get_name(heap), str);
  nvl_close(heap);
}

int main() {
  writeNVL("example.nvl");
  readNVL("example.nvl");
  return 0;
}

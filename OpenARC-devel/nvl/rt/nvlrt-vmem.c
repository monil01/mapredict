// A component of the NVL runtime that wraps Intel's libvmem. See ../README
// for details.

#include "nvl-vheap.h"
#include <libvmem.h>

nvl_vheap_t *nvl_vcreate(const char *dir, size_t size) {
  return vmem_create(dir, size < VMEM_MIN_POOL ? VMEM_MIN_POOL : size);
}

void nvl_vclose(nvl_vheap_t *heap) {
  vmem_delete(heap);
}

void *nvl_vmalloc(nvl_vheap_t *heap, size_t size) {
  return vmem_malloc(heap, size);
}

void nvl_vfree(nvl_vheap_t *heap, void *ptr) {
  vmem_free(heap, ptr);
}

void *nvl_vcalloc(nvl_vheap_t *heap, size_t nmemb, size_t size) {
  return vmem_calloc(heap, nmemb, size);
}

void *nvl_vrealloc(nvl_vheap_t *heap, void *ptr, size_t size) {
  return vmem_realloc(heap, ptr, size);
}

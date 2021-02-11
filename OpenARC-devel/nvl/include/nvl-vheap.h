// The following C API is for using NVM as extended volatile memory. It is
// independent of the NVL-C language extensions and corresponding API in
// nvl.h, and it only requires a standard C compiler.
//
// To use both systems, include nvl.h, but your compiler must support NVL-C.
// To use only this system, include nvl-vheap.h.

#ifndef NVL_VMEM_H
#define NVL_VMEM_H

#include <stddef.h>

typedef struct vmem nvl_vheap_t;
nvl_vheap_t *nvl_vcreate(const char *dir, size_t size);
void nvl_vclose(nvl_vheap_t *heap);
void *nvl_vmalloc(nvl_vheap_t *heap, size_t size);
void nvl_vfree(nvl_vheap_t *heap, void *ptr);
void *nvl_vcalloc(nvl_vheap_t *heap, size_t nmemb, size_t size);
void *nvl_vrealloc(nvl_vheap_t *heap, void *ptr, size_t size);

#endif

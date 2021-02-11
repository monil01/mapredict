#ifndef NVL_H
#define NVL_H 1

#define nvl __nvl__
#define nvl_wp __nvl_wp__
typedef __builtin_nvl_heap nvl_heap_t;

// Recover an existing NVM heap, clobbering the file if the file exists and
// cannot be opened as an NVM heap, or create a new NVM heap if the file
// doesn't exist yet.
//
// nvl_recover encapsulates an idiom for the first step of NVM heap
// initialization within an application that might be run in either of two
// modes: (1) create an NVM heap for the first time, or (2) resume use of an
// existing NVM heap, possibly after a previous failure of the application
// running in either mode. A successful nvl_recover returns a non-NULL
// nvl_heap_t*, and then the next step in NVM heap initialization is the
// responsibility of the caller: immediately run a transaction that, if and
// only if the NVM heap's root is still NULL, sets the root and the initial
// state of the NVM heap's application data. Thus, after both steps have
// completed successfully, in future runs of the application (possibly
// intermixed with application failures), nvl_recover will merely call
// nvl_open, and the NVM heap's application data will be recovered not
// reinitialized. For example:
//
//   nvl_heap_t *heap = nvl_recover(heapFileName, heapSize, 0600);
//   if (!heap) {
//     fprintf(stderr, "%s: ", heapFileName);
//     perror("failed to recover or create NVM heap");
//     exit(1);
//   }
//   nvl struct root *root = nvl_get_root(heap, struct root);
//   if (!root) {
//     #pragma nvl atomic heap(heap)
//     {
//       root = nvl_alloc_nv(heap, 1, struct root);
//       if (!root) {
//         perror("failed to allocate NVM heap root");
//         exit(1);
//       }
//       nvl_set_root(heap, root);
//       root->iter = 0;
//       root->data = 2;
//     }
//   }
//
// It is tempting to skip the pragma and move the nvl_set_root call to the
// end. For application failures that preserve NVM write order, that indeed
// would be sufficient. However, at least power loss can corrupt NVM write
// order due to caching.
//
// Calling nvl_recover is nearly the same as calling nvl_open and then, if
// nvl_open fails, calling nvl_create, but there is one important
// difference: nvl_recover will attempt to clobber the specified file if it
// exists but is not a valid NVM heap or is unreadable. For example, the
// file might be missing read permission, but the parent directory might
// have write permission, and so nvl_recover successfully clobbers the file.
// The reason for this clobbering behavior is that, if the application
// happened to fail when previously attempting to create the NVM heap, the
// file might have been left in any one of many such invalid states, so
// nvl_recover assumes any failure to open an existing file is the result of
// such an application failure. nvl_open and nvl_create never clobber
// existing files, so the application programmer should prefer them over
// nvl_recover when he either expects the NVM heap to have already been
// created successfully (nvl_open) or expects the file not to exist at all
// (nvl_create).
#define nvl_recover(name, initSize, mode) \
  __builtin_nvl_recover(name, initSize, mode)

#define nvl_create(name, initSize, mode) \
  __builtin_nvl_create(name, initSize, mode)
#define nvl_open(name) \
  __builtin_nvl_open(name)

#define nvl_recover_mpi(name, initSize, mode, mpiTxWorld) \
  __builtin_nvl_recover_mpi(name, initSize, mode, mpiTxWorld)
#define nvl_create_mpi(name, initSize, mode, mpiTxWorld) \
  __builtin_nvl_create_mpi(name, initSize, mode, mpiTxWorld)
#define nvl_open_mpi(name, mpiTxWorld) \
  __builtin_nvl_open_mpi(name, mpiTxWorld)

#define nvl_close(heap)             __builtin_nvl_close(heap)
#define nvl_get_name(heap)          __builtin_nvl_get_name(heap)
#define nvl_get_root(heap, type)    __builtin_nvl_get_root(heap, type)
#define nvl_set_root(heap, root)    __builtin_nvl_set_root(heap, root)
#define nvl_alloc_nv(heap, n, type) __builtin_nvl_alloc_nv(heap, n, type)
#define nvl_alloc_length(p)         __builtin_nvl_alloc_length(p)

// TODO: These will not remain in the API once we implement a more careful
// nvl_bare block construct. See todos in SrcSymbolTable.
#define nvl_bare_hack(ptr)          __builtin_nvl_bare_hack(ptr)
#define nvl_persist_hack(ptr, n)    __builtin_nvl_persist_hack(ptr, n)
#define nvl_nv2nv_to_v2nv_hack(ptr, ptrPtr) \
  __builtin_nvl_nv2nv_to_v2nv_hack(ptr, ptrPtr)

// TODO: When we introduce nvl_alloc_v into the nvl.h API (as a safe way to
// dynamically allocate volatile memory that contains NVM-stored pointers),
// it might be worthwhile to overload it to accept an nvl_vheap_t* as the
// location of the allocation. Without that argument, it would just allocate
// using the system malloc.
#include "nvl-vheap.h"

#endif

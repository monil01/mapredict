// This interface supports internal NVL runtime functionality for testing
// and profiling as used in the NVL-C test suite and benchmarks. This
// interface should not be considered stable across NVL-C releases.
//
// In NVL-C applications, this file should always be included after nvl.h
// so that types like nvl_heap_t are defined. In the NVL runtime, nvl.h must
// never be included.

#ifndef NVLRT_TEST_H
#define NVLRT_TEST_H

# if !NVLRT && !NVL_H
#  error nvlrt-test.h must be included after nvl.h
# endif

// Get the number of NVM allocations created and freed in this process. That
// is, how many times have nvlrt_alloc_nv and nvlrt_free_nv been called?
//
// Implemented only in versions of the NVL runtime that support automatic
// reference counting.
size_t nvlrt_get_numAllocNV(void);
size_t nvlrt_get_numFreeNV(void);

// Get the total real time (in seconds) this process has spent in
// nvlrt_tx_add.
//
// Implemented only in versions of the NVL runtime that support
// transactions.
double nvlrt_get_txAddTime(void);

// Get the total real time (in seconds) this process has spent in the
// nvlrt_alloc_nv call within nvlrt_tx_tryShadowUpdate.
//
// Implemented only in versions of the NVL runtime that support transactions
// and automatic reference counting.
double nvlrt_get_shadowUpdateAllocTime(void);

// Reset all statistics mentioned above that are implemented, or do nothing
// if none are implemented.
void nvlrt_resetStats(void);

// Get sizeof(nvl_heap_t).
//
// This is useful in the test suite when trying to force nvlrt_open to
// allocate an nvl_heap_t at a new memory location.
size_t nvlrt_get_sizeofNvlHeapT(void);

// Did pmem_is_pmem return false for the specified NVM heap?
# if NVLRT
size_t nvlrt_usesMsync(struct nvlrt_heap_t *heap);
# else
size_t nvlrt_usesMsync(nvl_heap_t *heap);
# endif

// From now on in this process, zero-initialize the allocations created by
// shadow updates.
//
// Implemented only in versions of the NVL runtime that support transactions
// and automatic reference counting.
void nvlrt_zeroShadowUpdateAlloc(void);

typedef enum {
  NVLRT_COST_DEFAULT,
  NVLRT_COST_ZERO,
  NVLRT_COST_COMPUTE,
  NVLRT_COST_INFINITE
} nvlrt_cost_mode_t;

// From now on in this process, compute the cost of a shadow update
// according to the specified cost mode:
//
// - NVLRT_COST_DEFAULT: revert to the default cost mode
// - NVLRT_COST_ZERO: perform shadow updates even if the cost model predicts
//   they will harm performance (but not if behavior would be incorrect)
// - NVLRT_COST_COMPUTE: perform shadow updates only if the cost model
//   predicts they will not harm performance
// - NVLRT_COST_INFINITE: never perform shadow updates
//
// Implemented only in versions of the NVL runtime that support transactions
// and automatic reference counting.
void nvlrt_setShadowUpdateCostMode(nvlrt_cost_mode_t cost);

// Get the current shadow update cost mode, which will never be
// NVLRT_COST_DEFAULT.
nvlrt_cost_mode_t nvlrt_getShadowUpdateCostMode(void);

#endif

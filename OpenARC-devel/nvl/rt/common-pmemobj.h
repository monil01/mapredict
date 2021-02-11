#if !NVLRT_NOREFS
# define NVLRT_PREMSG "nvlrt-pmemobj: "
#else
# define NVLRT_PREMSG "nvlrt-pmemobj-norefs: "
#endif
#define NVLRT_PMEM_TYPE_NUM 0

#if !NVLRT_NOTXS
# define NVLRT_TX_BEGIN(Heap) \
         nvlrt_tx_begin(Heap)
# define NVLRT_TX_END(Heap) \
         nvlrt_tx_end(Heap)
# define NVLRT_TX_ADD(P, NumElements, ElementSize) \
         nvlrt_tx_add(P, NumElements, ElementSize)
# define NVLRT_TX_ADD_BARE(P, NumElements, ElementSize) \
         nvlrt_tx_add_bare(P, NumElements, ElementSize)
# define NVLRT_PMEMOBJ_ZALLOC(Pool, Oidp, Size, TypeNum) \
         OID_IS_NULL(*(Oidp) = pmemobj_tx_zalloc(Size, TypeNum))
# define NVLRT_PMEMOBJ_ALLOC(Pool, Oidp, Size, TypeNum) \
         OID_IS_NULL(*(Oidp) = pmemobj_tx_alloc(Size, TypeNum))
# define NVLRT_PMEMOBJ_FREE(Oidp) \
         (pmemobj_tx_free(*(Oidp)) \
          ? (void)0 \
          : (void)((Oidp)->pool_uuid_lo = 0, (Oidp)->off = 0))
void nvlrt_tx_add_bare(void *p, size_t numElements, size_t elementSize);
#else
# define NVLRT_TX_BEGIN(Heap)
# define NVLRT_TX_END(Heap)
# define NVLRT_TX_ADD(P, NumElements, ElementSize)
# define NVLRT_TX_ADD_BARE(P, NumElements, ElementSize)
# define NVLRT_PMEMOBJ_ZALLOC(Pool, Oidp, Size, TypeNum) \
         pmemobj_zalloc(Pool, Oidp, Size, TypeNum)
# define NVLRT_PMEMOBJ_ALLOC(Pool, Oidp, Size, TypeNum) \
         pmemobj_alloc(Pool, Oidp, Size, TypeNum, NULL, NULL)
# define NVLRT_PMEMOBJ_FREE(Oidp) \
         pmemobj_free(Oidp)
#endif

#if !NVLRT_NOTXS && NVLRT_PERSIST
# error NVLRT_NOTXS must be non-zero if NVLRT_PERSIST is non-zero
#endif

#if NVLRT_PERSIST
# define NVLRT_PMEMOBJ_PERSIST(Pool, Ptr) \
         pmemobj_persist(Pool, Ptr, sizeof *(Ptr))
# define NVLRT_PMEMOBJ_PERSIST_SIZE(Pool, Ptr, Size) \
         pmemobj_persist(Pool, Ptr, Size)
#else
# define NVLRT_PMEMOBJ_PERSIST(Pool, Ptr)
# define NVLRT_PMEMOBJ_PERSIST_SIZE(Pool, Ptr, Size)
#endif

// These encapsulate a fast version of pmemobj_direct and similar functions.
// The caller is responsible for handling null pointers.
static inline void *nvlrt_offToDirect(PMEMobjpool *pop, uintptr_t off) {
  return (void*)((uintptr_t)pop + off);
}
static inline uintptr_t nvlrt_directToOff(PMEMobjpool *pop, void *direct) {
  return (uintptr_t)direct - (uintptr_t)pop;
}
static inline PMEMoid nvlrt_offToOID(uint64_t pool_uuid_lo, uintptr_t off) {
  PMEMoid oid = {pool_uuid_lo, off};
  return oid;
}
static inline uintptr_t nvlrt_oidToOff(PMEMoid oid) {
  return oid.off;
}
static inline PMEMoid nvlrt_directToOID(PMEMobjpool *pop,
                                        uint64_t pool_uuid_lo, void *direct)
{
  return nvlrt_offToOID(pool_uuid_lo, nvlrt_directToOff(pop, direct));
}
static inline void *nvlrt_oidToDirect(PMEMobjpool *pop, PMEMoid oid) {
  return nvlrt_offToDirect(pop, nvlrt_oidToOff(oid));
}

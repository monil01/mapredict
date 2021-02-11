// A version of the NVL runtime that wraps Intel's libpmemobj.
//
// There several cpp macros that affect the build:
//
// * -DNVLRT_NOREFS disables runtime support for reference counting.
// * -DNVLRT_NOTXS disables runtime support for transactions, but then
//   vrefs-pmemobj.c should also be built with -DNVLRT_NOTXS.
// * -DNVLRT_PERSIST enables runtime support for persist calls after stores,
//   but then vrefs-pmemobj.c should also be built with -DNVLRT_PERSIST.
//   -DNVLRT_PERSIST cannot be used without -DNVLRT_NOTXS.
//
// See ../README for further details.

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <libpmem.h>
#include <libpmemobj.h>
#include "common-pmemobj.h"
#if !NVLRT_NOREFS
# include "vrefs-pmemobj.h"
#endif
#include "killpos.h"
#define NVLRT 1
#include "../include/nvlrt-test.h"
#if !NVLRT_NOMPI
# include <mpi.h>
#endif

// This must be kept in sync with the compiler front end's choice of
// checksum algorithm.
#define TYPE_CHECKSUM_NBYTES 16

// We use pmemobj's layout setting to indicate whether a heap's NV-to-NV
// pointers have nextPtrOff fields. If we're performing automatic reference
// counting, we do not correctly handle heaps whose NV-to-NV pointers don't
// have nextPtrOff fields, and pmemobj rejects opening a heap with an
// unexpected layout.
#if !NVLRT_NOREFS
#define PMEM_LAYOUT "nvlrt-pmemobj"
#else
#define PMEM_LAYOUT "nvlrt-pmemobj-norefs"
#endif

// The first field must be the base virtual address of the heap from which
// all alloc and obj fields are computed. That way, the NVLLowerPointers
// pass can convert between V-to-NV pointers and NV-to-NV pointers without
// knowing any more about the layout of nvlrt_heap_t.
typedef struct nvlrt_heap_t {
  PMEMobjpool *pop; // becomes null upon nvlrt_close
  char *name;
  // TODO: If pmemobj were to expose the is_pmem field of PMEMobjpool
  // (perhaps via a get function), we wouldn't have to call pmem_is_pmem to
  // compute it. Because pmem_is_pmem does not always perform well, we call
  // it once when opening/creating the NVM heap and store it here.
  bool usesMsync;
  uint64_t pool_uuid_lo; // pool_uuid_lo from PMEMoid
#if !NVLRT_NOREFS
  size_t vrefs; // total number of V-to-NV references into this heap
  VrefTable vrefTable;
#endif
#if !NVLRT_NOMPI
  // Convenient access to the world and group encoded in the nvlrt_root_t
  // for this heap.
  // MPI_GROUP_NULL iff heap is in local tx mode.
  MPI_Group mpiTxWorld;
  // MPI_GROUP_NULL iff no MPI group tx is active (including when heap is in
  // local tx mode).
  MPI_Group mpiTxGroup;
  // 0 means no current transactions.
  // 1 means outermost transaction.
  unsigned long txNestLevel;
#endif
} nvlrt_heap_t;

typedef struct {
#if !NVLRT_NOREFS
  // Off field from PMEMoid to VrefOnlyNode.
  uint64_t vrefOnlyNodeOff;
  // Off field from PMEMoid to first ptr in allocation.
  uint64_t firstPtrOff;
  // Bits are in one of the following formats:
  // - {0,COUNT} where COUNT is a 63-bit count of the NV-to-NV pointers to
  //   the allocation. In this case, the addresses of those NV-to-NV
  //   pointers have not been recorded. We can be sure we'll never need the
  //   the 64th bit because pointers are 8 bytes wide, so the maximum number
  //   of pointers that can be addressed in memory at a single time is
  //   (2^64 bytes) / 8 (bytes/ptr) = 2^61 ptrs, which can be counted with
  //   just 62 bits.
  // - {1,OFF} where OFF is the high-order 63 bits of the off field from the
  //   PMEMoid for the only NV-to-NV pointer to the allocation. The
  //   lowest-order bit of the off field is 0. We normally will not
  //   encounter off fields for which the lowest-order bit is 1 because the
  //   addresses of pointers are normally aligned in memory (probably to 8
  //   bytes but at least to 2 bytes). Otherwise, we must use the first
  //   format instead.
  // If there is only 1 NV-to-NV pointer to the allocation, either format is
  // possible. When the count rises to 1, the second format is used. When
  // the count falls to 1, the first format is used because the OFF for the
  // sole remaining NV-to-NV pointer was discarded when the count previously
  // rose above 1.
  uint64_t nvrefs;
#endif
  // Number of elements and size of an element for the allocated type, as
  // specified to nvlrt_alloc_nv. For multi-dimensional arrays, that's the
  // outermost array's element type, so do not mix these with a numElements
  // and elementSize that might be specified in terms of a nested array's
  // element type. For example, nvlrt_tx_add's liked-named arguments could
  // be specified in terms of any of the arrays' element types.
  size_t numElements;
  size_t elementSize;
  // Flexible array member with largest possible alignment.
  uint64_t data[];
} nvlrt_alloc_t;
// nvlrt_alloc_t.nvrefs with only the highest-order bit as 1.
#define NVLRT_NVREFS_H1 ((uint64_t)1 << 63)

// Fields must be kept in sync with the nvl-lower-pointers pass.
// Field sizes must be the same as in nvlrt_nv2nv_t.
typedef struct nvlrt_v2nv_t {
  nvlrt_heap_t *heap;
  nvlrt_alloc_t *alloc; // absolute virtual address of the allocation
  void *obj;            // absolute virtual address of the object
} nvlrt_v2nv_t;

// Happens to be the same as OID_NULL plus alloc=0.
static const nvlrt_v2nv_t nvlrt_v2nv_null = {0, 0, 0};

// Fields must be kept in sync with the nvl-lower-pointers pass.
// Field sizes must be the same as in nvlrt_v2nv_t.
typedef struct {
#if !NVLRT_NOREFS
  uint64_t nextPtrOff; // off field from PMEMoid to next ptr in allocation
#else
  uint64_t unused;
#endif
  uint64_t allocOff; // off field from PMEMoid to the enclosing allocation
  uint64_t objOff;   // off field from PMEMoid to object
} nvlrt_nv2nv_t;

static inline double nvlrt_time() {
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + time.tv_usec / 1000000.0;
}

static inline bool nvlrt_v2nv_isNull(nvlrt_v2nv_t v2nv) {
  assert(!v2nv.alloc == !v2nv.heap && !v2nv.alloc == !v2nv.obj);
  return !v2nv.alloc;
}
static inline bool nvlrt_nv2nv_isNull(nvlrt_nv2nv_t nv2nv) {
  assert(!nv2nv.allocOff == !nv2nv.objOff);
  return !nv2nv.allocOff;
}

// Convert an nv2nv to a v2nv pointer. Call nvlrt_nv2nv_toV2nv instead if
// the nv2nv pointer might be null. It's the caller's responsibility to
// ensure heap is not null or closed.
static inline nvlrt_v2nv_t nvlrt_nv2nv_toV2nvNonNull(nvlrt_nv2nv_t nv2nv,
                                                     nvlrt_heap_t *heap)
{
  PMEMobjpool *pop = heap->pop;
  nvlrt_v2nv_t v2nv = {heap,
                       nvlrt_offToDirect(pop, nv2nv.allocOff),
                       nvlrt_offToDirect(pop, nv2nv.objOff)};
  return v2nv;
}

// Convert an nv2nv to a v2nv pointer. Call nvlrt_nv2nv_toV2nvNonNull
// instead if the nv2nv pointer is guaranteed not to be null. It's the
// caller's responsibility to ensure heap is not null or closed.
static inline nvlrt_v2nv_t nvlrt_nv2nv_toV2nv(nvlrt_nv2nv_t nv2nv,
                                              nvlrt_heap_t *heap)
{
  if (nvlrt_nv2nv_isNull(nv2nv))
    return nvlrt_v2nv_null;
  return nvlrt_nv2nv_toV2nvNonNull(nv2nv, heap);
}

#if !NVLRT_NOMPI
// It will probably always be int based on the MPI standard, but the typedef
// makes the intent more obvious.
typedef int nvlrt_mpiRank_t;
const MPI_Datatype NVLRT_MPI_RANK_MPI_TYPE = MPI_INT;
# define NVLRT_MPI_RANK_FMT "d"

// A type to store an array of MPI ranks in NVM. Don't access the fields
// except through the nvlrt_mpiRanks_* functions.
//
// arrOID points to an array with elements of type nvlrt_mpiRank_t. capacity
// is the known number of elements in the array, and size is the number of
// leading elements that are currently in use. Thus, it is always true that
// size <= capacity. Zero-initialization (by pmemobj_root) yields no array
// allocated.
//
// We say that capacity is the array's known number of elements because the
// actual number of elements might be larger. That is, it is possible for
// the application to terminate unexpectedly after enlarging the array and
// before enlarging capacity. The worst effect is that the array might later
// be reallocated when it doesn't need to be. Also, when capacity=0,
// OID_IS_NULL(arrOID) might or might not be true, so you must test the
// latter to determine whether there's anything to free.
//
// Operations on this type are performed outside of any transactions, so
// they use non-transactional versions of pmemobj functions to make sure any
// failure (such as a power outage) do not corrupt NVM data. We might think
// to instead run them within transactions, but that could confuse recovery,
// which wouldn't be able to determine whether the recorded MPI group is
// active or still being set up. At least when the array doesn't need to be
// reallocated, the non-transactional approach requires less syncs to
// physical NVM because there are no undo logs.
typedef struct {
  // Don't access the fields except through the nvlrt_mpiRanks_* functions.
  PMEMoid arrOID;
  nvlrt_mpiRank_t capacity;
  nvlrt_mpiRank_t size;
} nvlrt_mpiRanks_t;

static inline void nvlrt_mpiRanks_clear(PMEMobjpool *pop,
                                        nvlrt_mpiRanks_t *mpiRanks)
{
  if (mpiRanks->size) {
    mpiRanks->size = 0;
    pmemobj_persist(pop, &mpiRanks->size, sizeof mpiRanks->size);
  }
}

static inline nvlrt_mpiRank_t nvlrt_mpiRanks_getSize(
  PMEMobjpool *pop, nvlrt_mpiRanks_t *mpiRanks)
{
  return mpiRanks->size;
}

static inline nvlrt_mpiRank_t *nvlrt_mpiRanks_getArray(
  PMEMobjpool *pop, nvlrt_mpiRanks_t *mpiRanks)
{
  return nvlrt_oidToDirect(pop, mpiRanks->arrOID);
}

// Enlarges capacity (if necessary) but not the size, which should be set
// after the data.
static void nvlrt_mpiRanks_enlarge(
  PMEMobjpool *pop, nvlrt_mpiRanks_t *mpiRanks, nvlrt_mpiRank_t capacity)
{
  if (mpiRanks->capacity >= capacity)
    return;
  // Atomically (re)allocate and set OID so there's no memory leak.
  if (pmemobj_realloc(pop, &mpiRanks->arrOID,
                      capacity * sizeof(nvlrt_mpiRank_t),
                      NVLRT_PMEM_TYPE_NUM))
  {
    fprintf(stderr, NVLRT_PREMSG"error: MPI group ranks array"
                                " (re)allocation failed: %s\n",
            strerror(errno));
    exit(1);
  }
  // Now that the actual capacity has increased, update the known capacity.
  // Do so before before updating the size, or else we might end up with
  // capacity<size in physical NVM if there's a failure.
  mpiRanks->capacity = capacity;
  pmemobj_persist(pop, &mpiRanks->capacity, sizeof mpiRanks->capacity);
}

// Storage for returned array is reused by subsequent calls and should not
// be freed or realloced by caller. It is fine to overwrite its contents.
static nvlrt_mpiRank_t *nvlrt_mpiGroupToWorldRanks(
  MPI_Group grp, nvlrt_mpiRank_t *grpSize)
{
  MPI_Group worldGroup;
  MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
  MPI_Group_size(grp, grpSize);
  static nvlrt_mpiRank_t *ranks = NULL;
  static nvlrt_mpiRank_t *id = NULL;
  static nvlrt_mpiRank_t size = 0;
  if (size < *grpSize) {
    ranks = realloc(ranks, *grpSize * sizeof *ranks);
    id    = realloc(id,    *grpSize * sizeof *id);
    if (!ranks || !id) {
      fprintf(stderr, NVLRT_PREMSG"error: allocation of arrays for MPI"
                      " group translation failed: %s\n",
              strerror(errno));
      exit(1);
    }
    for (; size < *grpSize; ++size)
      id[size] = size;
  }
  MPI_Group_translate_ranks(grp, size, id, worldGroup, ranks);
  MPI_Group_free(&worldGroup);
  return ranks;
}

// Enlarges, persists the data, and then persists the new size.
static void nvlrt_mpiRanks_setGroup(
  PMEMobjpool *pop, nvlrt_mpiRanks_t *mpiRanks,
  const nvlrt_mpiRank_t *newRanks, nvlrt_mpiRank_t newSize)
{
  nvlrt_mpiRanks_enlarge(pop, mpiRanks, newSize);
  nvlrt_mpiRank_t *ranks = nvlrt_oidToDirect(pop, mpiRanks->arrOID);
  memcpy(ranks, newRanks, newSize * sizeof *ranks);
  pmemobj_persist(pop, ranks, newSize * sizeof *ranks);
  mpiRanks->size = newSize;
  pmemobj_persist(pop, &mpiRanks->size, sizeof mpiRanks->size);
}

typedef struct {
  // false=TX_STATE_NONE, true=TX_STATE_COMMITTED
  bool committed;
  // We never compare two counts whose difference is more than 1. Thus,
  // counts are incremented modulo 3 to avoid overflow. To compute i < j, we
  // instead compute (i+1) % 3 == j.
  nvlrt_mpiRanks_t counts;
} nvlrt_mpiTxState_t;
#endif

typedef struct {
  nvlrt_nv2nv_t nv2nv;
  char typeChecksum[TYPE_CHECKSUM_NBYTES];
#if !NVLRT_NOREFS
  VrefOnlyList vrefOnlyList;
#endif
#if !NVLRT_NOMPI
  // Whether this heap is in local tx mode (false) or MPI group tx mode
  // (true).
  //
  // If true, then the remaining mpi* fields specify the MPI metadata.
  //
  // If false, then the remaining mpi* fields are meaningful only if
  // mpiTxWorld is non-empty, as follows:
  // - If mpiTxWorld is empty (in size not necessarily capacity), then
  //   mpiTxWorld and the subsequent mpi* fields are not meaningful.
  //   mpiTxWorld becomes empty in two cases. (1) mpiTxWorld is initially
  //   empty when nvlrt_create is called. (2) After nvlrt_close is
  //   successfully called throughout an MPI tx world and then nvlrt_open or
  //   nvlrt_recover is later called with a new MPI tx world, mpiTxWorld and
  //   subsequent mpi* fields are rewritten with the new MPI tx world. The
  //   first step of this rewrite is to set mpiTxWorld to empty so that, in
  //   the case of a failure during this rewrite, partially rewritten data
  //   in mpiTxWorld and subsequent mpi* fields does not appear meaningful
  //   during recovery. In both of these cases, an empty mpiTxWorld means
  //   that all transactions for any previous MPI tx world completed
  //   successfully and that old MPI tx counts are lost. Thus, MPI tx counts
  //   will be consistently zeroed throughout any new MPI tx world.
  // - If mpiTxWorld is non-empty (in size and thus capacity), then
  //   mpiTxWorld and the subsequent mpi* fields contain the MPI metadata
  //   for the previous MPI tx world. This MPI metadata is useful when
  //   nvlrt_close is successful on this heap (thus setting
  //   mpiGroupTxMode=false) but at least one other member of the MPI tx
  //   world failed. In the case of any failure in a MPI tx world, it's up
  //   to the user to guarantee that nvlrt_open or nvlrt_recover is called
  //   with the same MPI tx world, which permits communication with other
  //   members to learn of their previous failure. Then, nvlrt_open or
  //   nvlrt_recover knows to restore the MPI metadata stored here in order
  //   to have consistent MPI tx counts throughout the MPI tx world.
  //
  // For the sake of nvlrt_create, it's important that zero-initialization
  // sets local tx mode.
  bool mpiGroupTxMode;
  // The MPI_COMM_WORLD ranks of all MPI processes (including this MPI
  // process) with which this MPI process might participate in an MPI group
  // transaction.
  //
  // For the sake of nvlrt_create, it's important that zero-initialization
  // leaves this empty. See comments on mpiGroupTxMode.
  nvlrt_mpiRanks_t mpiTxWorld;
  // This MPI process's MPI_COMM_WORLD rank.
  nvlrt_mpiRank_t mpiWorldRank;
  // The indices of the MPI tx group members within mpiTxWorld.
  // Empty if the current transaction is local or there's no transaction.
  // Non-empty if the current transaction is an MPI group transaction.
  // According to my original design, it's not actually necessary to clear
  // the MPI tx group at the end of transactions or during recovery.
  // However, doing so probably helps with debugging.
  nvlrt_mpiRanks_t mpiTxGroupToWorld;
  // Selects active element of mpiTxStates. To atomically update tx
  // committed state and tx counts together, write the new state to the
  // inactive element and sync it to physical NVM, and then flip
  // mpiTxStateActive and sync it to physical NVM. Sometimes, we only need
  // to update the tx committed state, and then there's no need to switch
  // active elements because it can be written atomically.
  bool mpiTxStateActive;
  // The counts array sizes are the same as mpiTxWorld's when mpiTxWorld
  // is non-empty.
  nvlrt_mpiTxState_t mpiTxStates[2];
#endif
} nvlrt_root_t;

void nvlrt_report_heapAlloc() {
  fprintf(stderr, NVLRT_PREMSG"error: illegal operation on pointers to"
                  " different NVM allocations\n");
}

void nvlrt_report_heap() {
  fprintf(stderr, NVLRT_PREMSG"error: creation of interheap NV-to-NV"
                  " pointer\n");
}

#if !NVLRT_NOTXS
void nvlrt_tx_begin(nvlrt_heap_t *heap);
void nvlrt_tx_end(nvlrt_heap_t *heap);
nvlrt_v2nv_t nvlrt_tx_add(nvlrt_v2nv_t p, size_t numElements,
                          size_t elementSize, bool tryShadowUpdate,
                          bool writeFirst);
#endif

static inline nvlrt_root_t *nvlrt_get_popPmemRoot(PMEMobjpool *pop) {
  PMEMoid pmemRootOID = pmemobj_root(pop, sizeof(nvlrt_root_t));
  assert(!OID_IS_NULL(pmemRootOID)
         && "pmemobj pool root not allocated properly");
  return nvlrt_oidToDirect(pop, pmemRootOID);
}
// The caller is responsible for calling nvlrt_check_closedHeap.
static inline nvlrt_root_t *nvlrt_get_pmemRoot(nvlrt_heap_t *heap) {
  return nvlrt_get_popPmemRoot(heap->pop);
}

#if !NVLRT_NOREFS
static size_t numAllocNV = 0;
static size_t numFreeNV = 0;
size_t nvlrt_get_numAllocNV() { return numAllocNV; }
size_t nvlrt_get_numFreeNV()  { return numFreeNV; }

static inline void nvlrt_check_closedHeap(nvlrt_heap_t *heap) {
  if (heap->pop)
    return;
  fprintf(stderr, NVLRT_PREMSG"error: access to closed heap\n");
  exit(1);
}

static void nvlrt_free_nv(nvlrt_heap_t *heap, uint64_t allocOff,
                          nvlrt_alloc_t *alloc);

// Does not modify NVM, so caller need not worry about enclosing call in a
// transaction. (The pmemobj_root call here would modify NVM if it changed
// the root size, but that would be a bug in the NVL runtime.)
static VrefOnlyList *nvlrt_vrefOnly_getList(nvlrt_heap_t *heap) {
  return &nvlrt_get_pmemRoot(heap)->vrefOnlyList;
}

// Caller must NVLRT_TX_ADD/persist the vrefOnlyNodeOff field for changes
// made here. Caller must enclose call in a transaction.
static void nvlrt_vrefOnly_add(nvlrt_heap_t *heap, uint64_t allocOff,
                               nvlrt_alloc_t *alloc)
{
  VrefOnlyList *vrefOnlyList = nvlrt_vrefOnly_getList(heap);
  assert(!alloc->vrefOnlyNodeOff);
  alloc->vrefOnlyNodeOff = VrefOnlyList_put(vrefOnlyList, heap->pop,
                                            heap->pool_uuid_lo, allocOff);
}

// If freeing, then the caller must properly free the allocation afterward.
// Otherwise, the caller must NVLRT_TX_ADD/persist the vrefOnlyNodeOff field
// for changes made here. Either way, caller must enclose call in a
// transaction.
static void nvlrt_vrefOnly_remove(nvlrt_heap_t *heap, nvlrt_alloc_t *alloc,
                                  bool freeing)
{
  VrefOnlyList *vrefOnlyList = nvlrt_vrefOnly_getList(heap);
  assert(alloc->vrefOnlyNodeOff);
  VrefOnlyList_remove(vrefOnlyList, heap->pop, heap->pool_uuid_lo,
                      alloc->vrefOnlyNodeOff);
  if (!freeing)
    alloc->vrefOnlyNodeOff = 0;
}

// Body contains transactions, so caller need not worry about enclosing call
// in a transaction.
static void nvlrt_vrefOnly_freeAll(nvlrt_heap_t *heap) {
  PMEMobjpool *pop = heap->pop;
  uint64_t pool_uuid_lo = heap->pool_uuid_lo;
  VrefOnlyList *vrefOnlyList = nvlrt_vrefOnly_getList(heap);
  uint64_t vrefOnlyNodeOff = VrefOnlyList_get(vrefOnlyList, pop,
                                              pool_uuid_lo);
  while (vrefOnlyNodeOff) {
    VrefOnlyNode *vrefOnlyNode = nvlrt_offToDirect(pop, vrefOnlyNodeOff);
    uint64_t allocOff = vrefOnlyNode->allocOff;
    nvlrt_alloc_t *alloc = nvlrt_offToDirect(pop, allocOff);
    NVLRT_TX_BEGIN(heap);
    nvlrt_free_nv(heap, allocOff, alloc);
    VrefOnlyList_remove(vrefOnlyList, pop, pool_uuid_lo, vrefOnlyNodeOff);
    NVLRT_TX_END(heap);
    vrefOnlyNodeOff = VrefOnlyList_get(vrefOnlyList, pop, pool_uuid_lo);
  }
}
#endif

#if NVLRT_NOMPI
# define NVLRT_NAME_MPI_GROUP(Name) nvlrt_##Name
# define NVLRT_MPI_TX_WORLD_PARAM
# define NVLRT_MPI_TX_WORLD_ARG
# define NVLRT_MPI_TX_WORLD_NULL
#else
# define NVLRT_NAME_MPI_GROUP(Name) nvlrt_##Name##_mpiGroup
# define NVLRT_MPI_TX_WORLD_PARAM , MPI_Group mpiTxWorld
# define NVLRT_MPI_TX_WORLD_ARG   , mpiTxWorld
# define NVLRT_MPI_TX_WORLD_NULL  , MPI_GROUP_NULL
#endif

static nvlrt_heap_t *nvlrt_create_or_open(
  const char *name, bool create, size_t initSize, unsigned long mode
  NVLRT_MPI_TX_WORLD_PARAM);

// The type of the initSize and mode parameters for nvlrt_create,
// nvlrt_create_or_open, and nvlrt_recover (and their _mpi versions) must be
// kept in sync with the compiler front end's type for those parameters to
// __builtin_nvl_create and __builtin_nvl_recover. See comments there.
nvlrt_heap_t *NVLRT_NAME_MPI_GROUP(create)(
  const char *name, size_t initSize, unsigned long mode
  NVLRT_MPI_TX_WORLD_PARAM)
{
  return nvlrt_create_or_open(name, true, initSize, mode
                              NVLRT_MPI_TX_WORLD_ARG);
}

nvlrt_heap_t *NVLRT_NAME_MPI_GROUP(open)(const char *name
                                         NVLRT_MPI_TX_WORLD_PARAM)
{
  return nvlrt_create_or_open(name, false, 0/*ignored*/, 0/*ignored*/
                              NVLRT_MPI_TX_WORLD_ARG);
}

#if !NVLRT_NOMPI
nvlrt_heap_t *nvlrt_create(const char *name, size_t initSize,
                           unsigned long mode)
{
  return nvlrt_create_mpiGroup(name, initSize, mode, MPI_GROUP_NULL);
}

nvlrt_heap_t *nvlrt_open(const char *name) {
  return nvlrt_open_mpiGroup(name, MPI_GROUP_NULL);
}
#endif

nvlrt_heap_t *NVLRT_NAME_MPI_GROUP(recover)(
  const char *name, size_t initSize, unsigned long mode
  NVLRT_MPI_TX_WORLD_PARAM)
{
  {
    nvlrt_heap_t *heap = NVLRT_NAME_MPI_GROUP(open)(name
                                                    NVLRT_MPI_TX_WORLD_ARG);
    if (heap) {
      fprintf(stderr, NVLRT_PREMSG"warning: %s: recovered heap\n", name);
      return heap;
    }
  }
  // Couldn't open file as heap. If it exists, then try to clobber it.
  if (errno != ENOENT) {
    fprintf(stderr, NVLRT_PREMSG"warning: %s: ", name);
    perror("invalid heap file");
    fprintf(stderr, NVLRT_PREMSG"warning: %s: clobbering...\n", name);
    // If unlink fails, then its errno describes a problem that should not
    // result from merely an application failure during nvlrt_create, so
    // leave that errno instead of the errno from nvlrt_open.
    if (unlink(name))
      return NULL;
  }
  return NVLRT_NAME_MPI_GROUP(create)(name, initSize, mode
                                      NVLRT_MPI_TX_WORLD_ARG);
}

#if !NVLRT_NOMPI
nvlrt_heap_t *nvlrt_recover(const char *name, size_t initSize,
                            unsigned long mode)
{
  return nvlrt_recover_mpiGroup(name, initSize, mode, MPI_GROUP_NULL);
}

# ifndef NVLRT_DBG_MPI
#  define NVLRT_DBG_MPI 0
# endif
# if NVLRT_DBG_MPI
// Buffering helps to avoid overlapping output from different MPI processes.
// Using va_list would allow us to convert some macros to functions, but
// OpenARC+LLVM doesn't obey ABI rules for passing arguments as large as
// va_list. NVLRT_DBG_MPI_ARRAY must be a macro because its Vals parameter
// can be of a pointer to any type, and we must be able to index it and
// dereference it.
static char nvlrt_dbg_mpi_buf[2048];
static char *nvlrt_dbg_mpi_bufp = nvlrt_dbg_mpi_buf;
#  define NVLRT_DBG_MPI_BUFFER(Fmt, ...) \
  do { \
    size_t space = nvlrt_dbg_mpi_buf + (sizeof nvlrt_dbg_mpi_buf) \
                   - nvlrt_dbg_mpi_bufp; \
    int added = snprintf(nvlrt_dbg_mpi_bufp, space, (Fmt), __VA_ARGS__); \
    if (added >= space) { \
      fprintf(stderr, \
              NVLRT_PREMSG"warning: nvlrt_dbg_mpi_buf overflow\n"); \
      fflush(stderr); \
    } \
    nvlrt_dbg_mpi_bufp += added; \
  } while (0)
#  define NVLRT_DBG_MPI_ENDL() \
  do { \
    nvlrt_mpiRank_t rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    nvlrt_dbg_mpi_bufp = nvlrt_dbg_mpi_buf; \
    fprintf(stderr, "rank %"NVLRT_MPI_RANK_FMT": %s\n", \
            rank, nvlrt_dbg_mpi_buf); \
    fflush(stderr); \
  } while (0)
#  define NVLRT_DBG_MPI_VALUE(Var, TypeFmt) \
  do { \
    NVLRT_DBG_MPI_BUFFER("%s = ", #Var); \
    NVLRT_DBG_MPI_BUFFER("%"TypeFmt, (Var)); \
    NVLRT_DBG_MPI_ENDL(); \
  } while (0)
# define NVLRT_DBG_MPI_ARRAY(Var, TypeFmt, Ranks, Nranks) \
  do { \
    NVLRT_DBG_MPI_BUFFER("%s =", #Var); \
    for (nvlrt_mpiRank_t i = 0; i < (Nranks); ++i) { \
      NVLRT_DBG_MPI_BUFFER(" %"NVLRT_MPI_RANK_FMT":", (Ranks)[i]); \
      NVLRT_DBG_MPI_BUFFER("%"TypeFmt, (Var)[i]); \
    } \
    NVLRT_DBG_MPI_ENDL(); \
  } while (0)
# else
#  define NVLRT_DBG_MPI_BUFFER(Fmt, ...)
#  define NVLRT_DBG_MPI_ENDL()
#  define NVLRT_DBG_MPI_VALUE(Var, TypeFmt)
#  define NVLRT_DBG_MPI_ARRAY(Var, TypeFmt, Ranks, Nranks)
# endif

enum {NVLRT_MPI_MSG_GROUP_TX_MODE, NVLRT_MPI_MSG_TX_COMMITTED,
      NVLRT_MPI_MSG_TX_COUNTS, NVLRT_MPI_MSG_TX_GROUP_MEMBERSHIPS};

// Caller is responsible for freeing the returned array.
static void *nvlrt_mpiExchange(
  const void *val, bool valIsArray, MPI_Datatype type,
  const nvlrt_mpiRank_t *ranks, nvlrt_mpiRank_t nranks, int tag)
{
  nvlrt_mpiRank_t rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int typeSize;
  MPI_Type_size(type, &typeSize);

  static MPI_Request *reqs = NULL;
  static nvlrt_mpiRank_t reqsCapacity = 0;
  if (reqsCapacity < nranks) {
    free(reqs);
    reqs = malloc(nranks * sizeof *reqs);
    reqsCapacity = nranks;
  }
  void *vals = malloc(nranks * typeSize);
  if (!reqs || !vals) {
    fprintf(stderr, NVLRT_PREMSG"error: failed to allocate arrays for MPI"
                    " messages\n");
    exit(1);
  }
  // Includes a self-send, so must be non-blocking.
  for (nvlrt_mpiRank_t i = 0; i < nranks; ++i)
    MPI_Isend((char*)val + (valIsArray?i*typeSize:0), 1, type, ranks[i],
              tag, MPI_COMM_WORLD, &reqs[i]);
  for (nvlrt_mpiRank_t i = 0; i < nranks; ++i)
    MPI_Recv((char*)vals + i*typeSize, 1, type, ranks[i], tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Waitall(nranks, reqs, MPI_STATUSES_IGNORE);
  return vals;
}

typedef struct {
  MPI_Group mpiTxWorld;
  bool allFromLocalTxMode;
  bool txGroupCommittedOrNoGroupTx;
} nvlrt_mpiTxGroupIsCommitted_t;

static bool nvlrt_mpiTxGroupIsCommitted(PMEMobjpool *pop, void *data) {
  // WARNING: With one exception (see below) NVM modifications should not be
  // made in this function because recovery has not yet run. Allocating and
  // freeing NVM (including the PMEMobjpool root) is specifically
  // problematic.
  //
  // The only NVM modifications that should be made in this function are
  // local resets of MPI tx counts to zero for other MPI tx world members
  // that are transitioning from local tx mode to MPI group tx mode. Those
  // modifications must be made here so that they are durably recorded in
  // NVM before they are (here) sent to the corresponding MPI tx world
  // members, which must not record MPI group tx mode in NVM until they
  // receive the counts. Otherwise, a failure could mean those members would
  // not report the transition again and thus this MPI process would never
  // durably record the reset. We could probably rewrite this code not to
  // perform the resets in this function by delaying those sends, but I'm
  // pretty sure it's OK in pmemobj to make non-transactional durable writes
  // to previously allocated and unfreed NVM before performing recovery.
  MPI_Group mpiTxWorld;
  bool *allFromLocalTxMode;
  bool *txGroupCommittedOrNoGroupTx;
  {
    nvlrt_mpiTxGroupIsCommitted_t *p = data;
    mpiTxWorld = p->mpiTxWorld;
    allFromLocalTxMode = &p->allFromLocalTxMode;
    txGroupCommittedOrNoGroupTx = &p->txGroupCommittedOrNoGroupTx;
  }
  // Getting the root when it hasn't been allocated causes it to be
  // allocated, but allocations must not be created before pmemobj recovery
  // has run.
  nvlrt_root_t *pmemRoot = pmemobj_root_size(pop) == sizeof(nvlrt_root_t)
                           ? nvlrt_get_popPmemRoot(pop) : NULL;
  NVLRT_DBG_MPI_VALUE(pmemRoot, "p");

  // If local tx mode is requested, complain if the heap is in MPI group tx
  // mode, and otherwise revert to normal local tx recovery.
  if (mpiTxWorld == MPI_GROUP_NULL) {
    if (pmemRoot && pmemRoot->mpiGroupTxMode) {
      fprintf(stderr,
              NVLRT_PREMSG"error: opening heap in local transaction mode"
              " when previously in MPI group transaction mode\n");
      exit(1);
    }
    return true;
  }

  // MPI group tx mode is requested. Complain if the specified MPI tx world
  // is empty or otherwise does not include this MPI process.
  nvlrt_mpiRank_t rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  nvlrt_mpiRank_t txWorldSize;
  nvlrt_mpiRank_t *txWorldRanks = nvlrt_mpiGroupToWorldRanks(mpiTxWorld,
                                                             &txWorldSize);
  {
    nvlrt_mpiRank_t i;
    for (i = 0; i < txWorldSize; ++i)
      if (rank == txWorldRanks[i])
        break;
    if (i == txWorldSize) {
      fprintf(stderr, NVLRT_PREMSG"error: new MPI transaction world does"
                      " not include the process's own MPI rank\n");
      exit(1);
    }
  }

  // Exchange previous modes with other members of the newly specified MPI
  // tx world.
  bool groupTxMode = pmemRoot && pmemRoot->mpiGroupTxMode;
  NVLRT_DBG_MPI_VALUE(groupTxMode, "d");
  bool *allGroupTxModes = nvlrt_mpiExchange(&groupTxMode, false, MPI_C_BOOL,
                                            txWorldRanks, txWorldSize,
                                            NVLRT_MPI_MSG_GROUP_TX_MODE);
  NVLRT_DBG_MPI_ARRAY(allGroupTxModes, "d", txWorldRanks, txWorldSize);

  // Compute *allFromLocalTxMode, which is true iff all members (including
  // this process) of the newly specified MPI tx world were in local tx
  // mode. If so, we'll record the new MPI metadata after returning and
  // after performing local tx recovery. We cannot skip exchanging MPI tx
  // counts and other MPI metadata in this case because there might exist
  // some member of the MPI tx world for which *allFromLocalTxMode=false and
  // that thus might need to exhange MPI metadata. We cannot have that other
  // member simply assume MPI tx counts of zero for this process because the
  // other member must durably reset the MPI tx count to zero for this
  // process before this process records MPI group tx mode or else, if
  // there's a failure, the other member will not know to reset the MPI tx
  // count to zero during the next recovery. Exchanging MPI metadata ensures
  // between the reset and the recording of MPI group tx mode ensures this
  // sequence.
  *allFromLocalTxMode = true; // close succeeded throughout MPI tx world
  for (nvlrt_mpiRank_t i = 0; i < txWorldSize; ++i) {
    if (allGroupTxModes[i]) {
      *allFromLocalTxMode = false;
      break;
    }
  }
  NVLRT_DBG_MPI_VALUE(*allFromLocalTxMode, "d");

  // If we do not have valid MPI metadata recorded locally, then either (1)
  // the user failed to reopen all heaps with the same MPI tx world after a
  // failure, or (2) this heap never successfully initialized with the MPI
  // tx world. We cannot distinguish these cases, so we assume the latter
  // case. Tx counts are then zero because, if this heap never initialized
  // with the MPI tx world, then it certainly never performed a transaction.
  // Also, if all members of the newly specified MPI tx world were in local
  // tx mode, then reset MPI tx counts to zero.
  bool localHasMpiMetadata
    = pmemRoot && nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxWorld);
  bool txCommittedOrNoGroupTx;
  nvlrt_mpiRank_t *txCounts;
  nvlrt_mpiRank_t txGroupSize;
  if (!localHasMpiMetadata || *allFromLocalTxMode) {
    txCommittedOrNoGroupTx = true;
    txCounts = calloc(txWorldSize, sizeof *txCounts);
    txGroupSize = 0;
  }
  else {
    // Complain if the new MPI_COMM_WORLD rank or MPI tx world is not
    // consistent with those recorded in NVM.
    if (rank != pmemRoot->mpiWorldRank) {
      fprintf(stderr, NVLRT_PREMSG"error: opening heap with a different"
                      " MPI_COMM_WORLD rank: old=%d, new=%d\n",
              pmemRoot->mpiWorldRank, rank);
      exit(1);
    }
    if (txWorldSize != nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxWorld)
        || memcmp(nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxWorld),
                  txWorldRanks, txWorldSize * sizeof *txWorldRanks))
    {
      fprintf(stderr, NVLRT_PREMSG"error: opening heap with a different MPI"
                      " transaction world\n");
      exit(1);
    }
    txGroupSize = nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxGroupToWorld);
    nvlrt_mpiTxState_t *active = &pmemRoot->mpiTxStates[pmemRoot
                                                        ->mpiTxStateActive];
    txCommittedOrNoGroupTx = active->committed || !txGroupSize;
    txCounts = nvlrt_mpiRanks_getArray(pop, &active->counts);
    // For each remote member of the MPI tx world, if either the local or
    // remote member is in MPI group tx mode, continue using the old count
    // for that member. Otherwise, durably reset that count to zero because
    // the remote member will too.
    bool persistTxCounts = false;
    for (nvlrt_mpiRank_t i = 0; i < txWorldSize; ++i) {
      if (!allGroupTxModes[i] && !groupTxMode && txCounts[i]) {
        txCounts[i] = 0;
        persistTxCounts = true;
      }
    }
    if (persistTxCounts)
      pmemobj_persist(pop, txCounts, txWorldSize * sizeof *txCounts);
  }
  NVLRT_DBG_MPI_VALUE(txCommittedOrNoGroupTx, "d");
  NVLRT_DBG_MPI_ARRAY(txCounts, "d", txWorldRanks, txWorldSize);

  bool *allTxCommittedOrNoGroupTx = nvlrt_mpiExchange(
    &txCommittedOrNoGroupTx, false, MPI_C_BOOL, txWorldRanks, txWorldSize,
    NVLRT_MPI_MSG_TX_COMMITTED);
  NVLRT_DBG_MPI_ARRAY(allTxCommittedOrNoGroupTx, "d",
                      txWorldRanks, txWorldSize);

  nvlrt_mpiRank_t *allTxCounts = nvlrt_mpiExchange(
    txCounts, true, NVLRT_MPI_RANK_MPI_TYPE, txWorldRanks, txWorldSize,
    NVLRT_MPI_MSG_TX_COUNTS);
  NVLRT_DBG_MPI_ARRAY(allTxCounts, "d", txWorldRanks, txWorldSize);

  bool *txGroupMemberships = calloc(txWorldSize,
                                    sizeof *txGroupMemberships);
  if (txGroupSize) {
    nvlrt_mpiRank_t *txGroupToWorld
      = nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxGroupToWorld);
    for (nvlrt_mpiRank_t i = 0; i < txGroupSize; ++i)
      txGroupMemberships[txGroupToWorld[i]] = true;
  }
  NVLRT_DBG_MPI_ARRAY(txGroupMemberships, "d", txWorldRanks, txWorldSize);

  bool *allTxGroupMemberships = nvlrt_mpiExchange(
    txGroupMemberships, true, MPI_C_BOOL, txWorldRanks, txWorldSize,
    NVLRT_MPI_MSG_TX_GROUP_MEMBERSHIPS);
  NVLRT_DBG_MPI_ARRAY(allTxGroupMemberships, "d",
                      txWorldRanks, txWorldSize);

  // Determine whether to roll back (*txGroupCommittedOrNoGroupTx=false) or
  // discard undo logs (*txGroupCommittedOrNoGroupTx=true).
  if (txCommittedOrNoGroupTx)
    *txGroupCommittedOrNoGroupTx = true;
  else {
    assert(txGroupSize);
    *txGroupCommittedOrNoGroupTx = false;
    nvlrt_mpiRank_t *txGroupToWorld
      = nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxGroupToWorld);
    for (nvlrt_mpiRank_t i = 0; i < txGroupSize; ++i) {
      nvlrt_mpiRank_t j = txGroupToWorld[i];
      if ((allTxCounts[j] == txCounts[j] && allTxCommittedOrNoGroupTx[j]
           && allTxGroupMemberships[j])
          || allTxCounts[j] == (txCounts[j]+1)%3)
          //|| allTxCounts[j] > txCounts[j])
      {
        *txGroupCommittedOrNoGroupTx = true;
# if NDEBUG
        // Skip assertions for other members.
        break;
# endif
      }
      else {
        //assert(allTxCounts[j] < txCounts[j]
        assert((allTxCounts[j]+1)%3 == txCounts[j]
               || (allTxCounts[j] == txCounts[j]
                   && !allTxCommittedOrNoGroupTx[j])
               || (allTxCounts[j] == txCounts[j]
                   && allTxCommittedOrNoGroupTx[j]
                   && !allTxGroupMemberships[j]));
      }
    }
  }
  NVLRT_DBG_MPI_VALUE(*txGroupCommittedOrNoGroupTx, "d");

  free(allGroupTxModes);
  if (!localHasMpiMetadata || *allFromLocalTxMode)
    free(txCounts);
  free(allTxCommittedOrNoGroupTx);
  free(allTxCounts);
  free(txGroupMemberships);
  free(allTxGroupMemberships);
  // In case of local tx, normal recovery determines if committed.
  return *txGroupCommittedOrNoGroupTx;
}

static void nvlrt_mpiTxGroupIncCounts(nvlrt_heap_t *heap) {
  PMEMobjpool *pop = heap->pop;
  nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
  nvlrt_mpiTxState_t *old = &pmemRoot->mpiTxStates[pmemRoot
                                                   ->mpiTxStateActive];
  nvlrt_mpiTxState_t *new = &pmemRoot->mpiTxStates[!pmemRoot
                                                   ->mpiTxStateActive];
  nvlrt_mpiRank_t *oldCounts = nvlrt_mpiRanks_getArray(pop, &old->counts);
  nvlrt_mpiRank_t *newCounts = nvlrt_mpiRanks_getArray(pop, &new->counts);
  nvlrt_mpiRank_t *groupToWorld
    = nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxGroupToWorld);
  nvlrt_mpiRank_t groupSize
    = nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxGroupToWorld);
  assert(groupSize); // otherwise, why was this called?
  nvlrt_mpiRank_t worldSize = nvlrt_mpiRanks_getSize(pop,
                                                     &pmemRoot->mpiTxWorld);
  new->committed = false;
  memcpy(newCounts, oldCounts, worldSize * sizeof *newCounts);
  for (nvlrt_mpiRank_t i = 0; i < groupSize; ++i)
    newCounts[groupToWorld[i]] = (newCounts[groupToWorld[i]] + 1) % 3;
  pmemobj_persist(pop, &new->committed, sizeof new->committed);
  pmemobj_persist(pop, newCounts, worldSize * sizeof *newCounts);
  pmemRoot->mpiTxStateActive = !pmemRoot->mpiTxStateActive;
  pmemobj_persist(pop, &pmemRoot->mpiTxStateActive,
                  sizeof pmemRoot->mpiTxStateActive);
}
#endif

static nvlrt_heap_t *nvlrt_create_or_open(
  const char *name, bool create, size_t initSize, unsigned long mode
  NVLRT_MPI_TX_WORLD_PARAM)
{
  // Once we have an NVM-stored heap, we must be careful not to corrupt the
  // heap upon a failure. For recoverable application failures, we carefully
  // clean up. For unexpected application termination (unless we've disabled
  // transactions and thus don't care about this case), including power
  // loss, we need to guarantee that all NVM modifications will resume or
  // replay properly when nvlrt_create_or_open is called again for this
  // heap.
  //
  // Consider the case when MPI features are disabled. If create=true, then
  // the only NVM modification is the pmemobj_root call, which merely sets
  // the root size and zeroes the root, so that modification will replay
  // correctly next time. If create=false, then the only NVM modification is
  // nvlrt_vrefOnly_freeAll, which is performed after recovery and which
  // contains transactions as needed.
  //
  // MPI features make things much more challenging for two reasons. First,
  // they must perform NVM write sets whose atomicity must be protected from
  // failures but that cannot easily be protected using pmemobj
  // transactions, which correspond to user transactions, because (1)
  // multiple write sets are required per user transaction, and (2) some
  // write sets are performed outside user transactions. Second, if
  // create=false, they call nvlrt_mpiTxGroupIsCommitted via
  // pmemobj_open_hook before pmemobj recovery has been performed. Thus,
  // nvlrt_mpiTxGroupIsCommitted is careful not to perform any NVM
  // modifications, such as alloc or free, that require any of pmemobj's
  // other mechanisms (redo logging) for recovery. All such NVM
  // modifications must be made after pmemobj_open_hook has returned and
  // must be made in a way that leaves a consistent state both before and
  // after pmemobj recovery runs again.

  // Allocate heap data structure.
  nvlrt_heap_t *heap = malloc(sizeof(nvlrt_heap_t));
  if (heap == NULL)
    return NULL;

  // Set file name.
  heap->name = malloc(strlen(name) + 1);
  if (heap->name == NULL) {
    free(heap);
    return NULL;
  }
  strcpy(heap->name, name);

  // Open pool.
#if !NVLRT_NOMPI
  nvlrt_mpiTxGroupIsCommitted_t mpiTxGroupIsCommitted = {mpiTxWorld};
#endif
  if (create) {
    heap->pop = pmemobj_create(name, PMEM_LAYOUT,
                               initSize < PMEMOBJ_MIN_POOL
                               ? PMEMOBJ_MIN_POOL : initSize,
                               (mode_t)mode);
    if (!heap->pop) {
      free(heap->name);
      free(heap);
      return NULL;
    }
#if !NVLRT_NOMPI
    nvlrt_mpiTxGroupIsCommitted(heap->pop, &mpiTxGroupIsCommitted);
#endif
  }
  else {
#if NVLRT_NOMPI
    heap->pop = pmemobj_open(name, PMEM_LAYOUT);
#else
    heap->pop = pmemobj_open_hook(name, PMEM_LAYOUT,
                                  nvlrt_mpiTxGroupIsCommitted,
                                  &mpiTxGroupIsCommitted);
#endif
    if (!heap->pop) {
      free(heap->name);
      free(heap);
      return NULL;
    }
  }
  PMEMobjpool *pop = heap->pop;
#if !NVLRT_NOMPI
  bool allFromLocalTxMode = mpiTxGroupIsCommitted.allFromLocalTxMode;
  bool txGroupCommittedOrNoGroupTx = mpiTxGroupIsCommitted
                                     .txGroupCommittedOrNoGroupTx;
#endif

  // If new pool, add space for NVL root pointer's offset. If old pool,
  // that's already been done, but call pmemobj_root anyway to get
  // pool_uuid_lo.
  errno = 0;
  PMEMoid pmemRootOID = pmemobj_root(pop, sizeof(nvlrt_root_t));
  if (OID_IS_NULL(pmemRootOID)) {
    // pmemobj_root doesn't always set (or never sets?) errno upon an
    // error, so we do it. It seems that errors here should be impossible
    // if PMEMOBJ_MIN_POOL is a reasonable size.
    if (errno == 0)
      errno = ENOMEM;
    pmemobj_close(pop);
    free(heap->name);
    free(heap);
    return NULL;
  }
  heap->pool_uuid_lo = pmemRootOID.pool_uuid_lo;
  nvlrt_root_t *pmemRoot = nvlrt_oidToDirect(pop, pmemRootOID);
  heap->usesMsync = !pmem_is_pmem(pmemRoot, 1);

#if !NVLRT_NOMPI
  if (mpiTxWorld != MPI_GROUP_NULL) {
    bool localHasMpiMetadata = nvlrt_mpiRanks_getSize(pop, &pmemRoot
                                                           ->mpiTxWorld);
    if (!localHasMpiMetadata || allFromLocalTxMode) {
      nvlrt_mpiRank_t rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      assert(!pmemRoot->mpiGroupTxMode);
      if (localHasMpiMetadata)
        nvlrt_mpiRanks_clear(pop, &pmemRoot->mpiTxWorld);
      pmemRoot->mpiWorldRank = rank;
      nvlrt_mpiRank_t size;
      nvlrt_mpiRank_t *ranks = nvlrt_mpiGroupToWorldRanks(mpiTxWorld, &size);
      nvlrt_mpiRanks_enlarge(pop, &pmemRoot->mpiTxStates[0].counts, size);
      nvlrt_mpiRanks_enlarge(pop, &pmemRoot->mpiTxStates[1].counts, size);
      pmemobj_persist(pop, &pmemRoot->mpiWorldRank, sizeof(nvlrt_mpiRank_t));
      pmemobj_memset_persist(
        pop, nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxStates[0].counts),
        0, size * sizeof(nvlrt_mpiRank_t));
      pmemobj_memset_persist(
        pop, nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxStates[1].counts),
        0, size * sizeof(nvlrt_mpiRank_t));
      // Must be set and persisted after the above MPI metadata is persisted
      // as this indicates all MPI metadata is now in a consistent state.
      nvlrt_mpiRanks_setGroup(pop, &pmemRoot->mpiTxWorld, ranks, size);
      // Must be set and persisted after all MPI metadata is persisted as MPI
      // group tx mode cannot be active until then.
      bool *mpiGroupTxMode = &pmemRoot->mpiGroupTxMode;
      *mpiGroupTxMode = true;
      pmemobj_persist(pop, mpiGroupTxMode, sizeof *mpiGroupTxMode);
    }
    else if (txGroupCommittedOrNoGroupTx
             && nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxGroupToWorld))
    {
      assert(pmemRoot->mpiGroupTxMode);
      nvlrt_mpiTxGroupIncCounts(heap);
    }
    else if (!pmemRoot->mpiGroupTxMode) {
      // localHasMpiMetadata is true because either (1) nvlrt_close
      // previously succeeded locally but not throughout the MPI tx world or
      // (2) the above initialization of MPI metadata previously failed
      // right before setting mpiGroupTxMode.
      bool *mpiGroupTxMode = &pmemRoot->mpiGroupTxMode;
      *mpiGroupTxMode = true;
      pmemobj_persist(pop, mpiGroupTxMode, sizeof *mpiGroupTxMode);
    }
  }
  nvlrt_mpiRanks_clear(pop, &pmemRoot->mpiTxGroupToWorld);
  heap->mpiTxWorld = mpiTxWorld;
  heap->mpiTxGroup = MPI_GROUP_NULL;
  heap->txNestLevel = 0;
#endif

#if !NVLRT_NOREFS
  // Handle ref counting.
  heap->vrefs = 0;
  if (VrefTable_init(&heap->vrefTable)) {
    pmemobj_close(pop);
    free(heap->name);
    free(heap);
    return NULL;
  }
  if (!create)
    // Contains transactions, so must come after the heap's transaction
    // metadata (such as heap->mpiTxGroup and heap->txNestLevel) is
    // initialized above.
    nvlrt_vrefOnly_freeAll(heap);
#endif

  return heap;
}

static void nvlrt_free(nvlrt_heap_t *heap) {
#if !NVLRT_NOREFS
  VrefTable_deinit(&heap->vrefTable);
#endif
  free(heap->name);
  free(heap);
}

void nvlrt_close(nvlrt_heap_t *heap) {
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
  nvlrt_vrefOnly_freeAll(heap);
#endif
#if !NVLRT_NOMPI
  if (heap->mpiTxWorld != MPI_GROUP_NULL) {
    bool *mpiGroupTxMode = &nvlrt_get_pmemRoot(heap)->mpiGroupTxMode;
    *mpiGroupTxMode = false;
    pmemobj_persist(heap->pop, mpiGroupTxMode, sizeof *mpiGroupTxMode);
  }
#endif
  pmemobj_close(heap->pop);
#if !NVLRT_NOREFS
  heap->pop = NULL;
  if (heap->vrefs == 0)
#endif
    nvlrt_free(heap);
}

const char *nvlrt_get_name(nvlrt_heap_t *heap) {
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
#endif
  return heap->name;
}

#if !NVLRT_NOREFS
void nvlrt_inc_nv(nvlrt_v2nv_t p, nvlrt_nv2nv_t *pp);
void nvlrt_inc_v(nvlrt_v2nv_t p, nvlrt_v2nv_t *pp);
void nvlrt_dec_nv(nvlrt_v2nv_t p, nvlrt_nv2nv_t *pp);
void nvlrt_dec_v(nvlrt_v2nv_t p, nvlrt_v2nv_t *pp);
#endif

void nvlrt_set_root(nvlrt_heap_t *heap, nvlrt_v2nv_t root,
                    const char *typeChecksum)
{
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
#endif
  PMEMobjpool *pop = heap->pop;
  nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
  if (!nvlrt_v2nv_isNull(root) && root.heap != heap) {
    nvlrt_report_heap();
    exit(1);
  }
#if !NVLRT_NOREFS
  nvlrt_nv2nv_t oldRootNv2nv = pmemRoot->nv2nv;
#endif
  NVLRT_TX_BEGIN(heap);
  NVLRT_TX_ADD_BARE(pmemRoot, 1, sizeof *pmemRoot);
  if (nvlrt_v2nv_isNull(root)) {
    pmemRoot->nv2nv.allocOff = 0;
    pmemRoot->nv2nv.objOff   = 0;
  }
  else {
    pmemRoot->nv2nv.allocOff = nvlrt_directToOff(pop, root.alloc);
    pmemRoot->nv2nv.objOff   = nvlrt_directToOff(pop, root.obj);
  }
  strncpy(pmemRoot->typeChecksum, typeChecksum, TYPE_CHECKSUM_NBYTES);
  NVLRT_PMEMOBJ_PERSIST(pop, pmemRoot);
#if !NVLRT_NOREFS
  // inc before dec in case it's the same pointer
  nvlrt_inc_nv(root, &pmemRoot->nv2nv);
  nvlrt_v2nv_t oldRoot = nvlrt_nv2nv_toV2nv(oldRootNv2nv, heap);
  nvlrt_dec_nv(oldRoot, &pmemRoot->nv2nv);
#endif
  NVLRT_TX_END(heap);
}

nvlrt_v2nv_t nvlrt_get_root(nvlrt_heap_t *heap, const char *typeChecksum) {
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
#endif
  nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
  // Skip type check (and conversion and increment) for null pointer.
  if (nvlrt_nv2nv_isNull(pmemRoot->nv2nv))
    return nvlrt_v2nv_null;
  if (strncmp(pmemRoot->typeChecksum, typeChecksum, TYPE_CHECKSUM_NBYTES)) {
    fprintf(stderr, NVLRT_PREMSG"error: root type checksum mismatch\n");
    exit(1);
  }
  nvlrt_v2nv_t root = nvlrt_nv2nv_toV2nvNonNull(pmemRoot->nv2nv, heap);
#if !NVLRT_NOREFS
  nvlrt_inc_v(root, NULL);
#endif
  return root;
}

static nvlrt_v2nv_t nvlrt_alloc_nv_optZero(
  nvlrt_heap_t *heap, size_t numElements, size_t elementSize,
  nvlrt_v2nv_t initNextPtrsFn(nvlrt_v2nv_t, nvlrt_v2nv_t), bool zero)
{
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
#endif
  PMEMoid allocOID;
  size_t size = sizeof(nvlrt_alloc_t) + numElements * elementSize;
  PMEMobjpool *pop = heap->pop;
  NVLRT_TX_BEGIN(heap);
  // The entire allocation will be rolled back if the transaction does not
  // commit, so we don't need to NVLRT_TX_ADD the parts of the allocation
  // we write here.
  if (zero ? NVLRT_PMEMOBJ_ZALLOC(pop, &allocOID, size, NVLRT_PMEM_TYPE_NUM)
           : NVLRT_PMEMOBJ_ALLOC(pop, &allocOID, size, NVLRT_PMEM_TYPE_NUM))
  {
    NVLRT_TX_END(heap);
    return nvlrt_v2nv_null;
  }
  nvlrt_alloc_t *alloc = nvlrt_oidToDirect(pop, allocOID);
  alloc->numElements = numElements;
  alloc->elementSize = elementSize;
  nvlrt_v2nv_t v2nv = {heap, alloc, &alloc->data};
#if !NVLRT_NOREFS
  ++numAllocNV;
  if (!zero) {
    alloc->vrefOnlyNodeOff = 0;
    alloc->firstPtrOff = 0;
    alloc->nvrefs = 0;
  }
  if (initNextPtrsFn) {
    nvlrt_v2nv_t prev = {heap, alloc, &alloc->firstPtrOff};
    nvlrt_v2nv_t ele = v2nv;
    size_t i;
    for (i = 0; i < numElements; ++i) {
      prev = initNextPtrsFn(prev, ele);
      ele.obj = (void*)((uintptr_t)ele.obj + elementSize);
    }
  }
  nvlrt_vrefOnly_add(heap, nvlrt_oidToOff(allocOID), alloc);
  // Persist the allocation header and data.
  NVLRT_PMEMOBJ_PERSIST_SIZE(pop, alloc, size);
  nvlrt_inc_v(v2nv, NULL);
#else
  if (initNextPtrsFn) {
    fprintf(stderr, NVLRT_PREMSG"error: nvlrt_alloc_nv received non-null"
                    " initNextPtrsFn, but reference counting is not"
                    " built\n");
    exit(1);
  }
#endif
  NVLRT_TX_END(heap);
  return v2nv;
}

// numElements and elementSize are specified in terms of the allocated type.
// When allocating a multi-dimensional array, that's the outermost array's
// element type. For example, nvl_alloc_nv(heap, 2, int[3]) translates such
// that numElements=2 and elementSize=3*sizeof(int).
//
// The types of the numElements and elementSize parameters for
// nvlrt_alloc_nv must be kept in sync with the compiler front end's types
// for these parameters to __builtin_nvl_alloc_nv. See comments there.
nvlrt_v2nv_t nvlrt_alloc_nv(
  nvlrt_heap_t *heap, size_t numElements, size_t elementSize,
  nvlrt_v2nv_t initNextPtrsFn(nvlrt_v2nv_t, nvlrt_v2nv_t))
{
  return nvlrt_alloc_nv_optZero(heap, numElements, elementSize,
                                initNextPtrsFn, true);
}

// The result is numElements as specified to nvlrt_alloc_nv.
//
// The return type of nvlrt_alloc_length must be kept in sync with the
// compiler front end's return type for __builtin_nvl_alloc_length. See
// comments there.
size_t nvlrt_alloc_length(nvlrt_v2nv_t p) {
  // Like malloc_usable_size and pmemobj_alloc_usable_size (which are not
  // semantically equivalent), return 0 for null pointer.
  if (nvlrt_v2nv_isNull(p))
    return 0;
#if !NVLRT_NOREFS
  nvlrt_check_closedHeap(p.heap);
#endif
  return p.alloc->numElements;
}

#if !NVLRT_NOREFS
static bool nvrefsAlignFail = false;

// Caller must enclose call in a transaction that includes the store
// operation for which the inc is being performed.
void nvlrt_inc_nv(nvlrt_v2nv_t p, nvlrt_nv2nv_t *pp) {
  if (nvlrt_v2nv_isNull(p))
    return;
  if (!pp) {
    fprintf(stderr, NVLRT_PREMSG"error: NV-to-NV inc with null address of"
                    " NV-to-NV pointer\n");
    exit(1);
  }
  nvlrt_check_closedHeap(p.heap);
  nvlrt_alloc_t *alloc = p.alloc;
  if (alloc->nvrefs == 0) {
    // NVLRT_TX_ADD/persist only the allocation header not the data, which
    // we don't change.
    NVLRT_TX_ADD_BARE(alloc, 1, sizeof *alloc);
    nvlrt_vrefOnly_remove(p.heap, alloc, false);
    alloc->nvrefs = nvlrt_directToOff(p.heap->pop, pp);
    if (alloc->nvrefs & 1) {
      if (!nvrefsAlignFail) {
        fprintf(stderr,
                NVLRT_PREMSG"warning: cannot record NV-to-NV pointer in"
                " nvrefs because pointer address offset is not 2-byte"
                " aligned\n");
        nvrefsAlignFail = true;
      }
      alloc->nvrefs = 1;
    }
    else {
      alloc->nvrefs >>= 1;
      alloc->nvrefs |= NVLRT_NVREFS_H1;
    }
    NVLRT_PMEMOBJ_PERSIST(p.heap->pop, alloc);
  }
  else {
    NVLRT_TX_ADD_BARE(&alloc->nvrefs, 1, sizeof alloc->nvrefs);
    if (alloc->nvrefs & NVLRT_NVREFS_H1)
      alloc->nvrefs = 1;
    ++alloc->nvrefs;
    NVLRT_PMEMOBJ_PERSIST(p.heap->pop, &alloc->nvrefs);
  }
}

// nvlrt_inc_v never modifies NVM, so the caller need not worry about
// enclosing call in a transaction.
void nvlrt_inc_v(nvlrt_v2nv_t p, nvlrt_v2nv_t *pp) {
  // This function must not fail after the heap is closed. That is, the only
  // harm that should be possible from copying a dangling pointer is
  // delaying freeing the nvlrt_heap_t.
  if (nvlrt_v2nv_isNull(p))
    return;
  VrefEntry *vrefEntry = VrefTable_get(&p.heap->vrefTable, p.alloc);
  ++vrefEntry->vrefs;
  if (!vrefEntry->v2nvPtr)
    vrefEntry->v2nvPtr = pp;
  ++p.heap->vrefs;
}

// Caller must enclose call in a transaction and ensure alloc/allocOff are
// not null.
static void nvlrt_free_nv(nvlrt_heap_t *heap, uint64_t allocOff,
                          nvlrt_alloc_t *alloc)
{
  PMEMobjpool *pop = heap->pop;
  uint64_t ptrOff = alloc->firstPtrOff;
  while (ptrOff) {
    nvlrt_nv2nv_t *ptr2nv2nv = nvlrt_offToDirect(pop, ptrOff);
    nvlrt_nv2nv_t nv2nv = *ptr2nv2nv;
    nvlrt_v2nv_t v2nv = nvlrt_nv2nv_toV2nv(nv2nv, heap);
    nvlrt_dec_nv(v2nv, ptr2nv2nv);
    ptrOff = nv2nv.nextPtrOff;
  }
  PMEMoid allocOID = nvlrt_offToOID(heap->pool_uuid_lo, allocOff);
  NVLRT_PMEMOBJ_FREE(&allocOID); // sets allocOID to OID_NULL
  ++numFreeNV;
}

// Caller must enclose call in a transaction that includes the store
// operation for which the dec is being performed.
void nvlrt_dec_nv(nvlrt_v2nv_t p, nvlrt_nv2nv_t *pp) {
  if (nvlrt_v2nv_isNull(p))
    return;
  if (!pp) {
    fprintf(stderr, NVLRT_PREMSG"error: NV-to-NV dec with null address of"
                    " NV-to-NV pointer\n");
    exit(1);
  }
  nvlrt_heap_t *heap = p.heap;
  nvlrt_check_closedHeap(heap);
  PMEMobjpool *pop = heap->pop;
  nvlrt_alloc_t *alloc = p.alloc;
  assert(alloc->nvrefs);
  if (alloc->nvrefs == 1 || (alloc->nvrefs & NVLRT_NVREFS_H1)) {
    assert(!(alloc->nvrefs & NVLRT_NVREFS_H1)
           || pp == nvlrt_offToDirect(pop, alloc->nvrefs << 1));
    VrefEntry *vrefEntry = VrefTable_get(&heap->vrefTable, alloc);
    // While p is a V-to-NV reference, it is not included in vrefs count
    // because p was loaded from NVM solely in order to call nvlrt_dec_nv.
    // Thus, vrefs can be zero here.
    uint64_t allocOff = nvlrt_directToOff(pop, alloc);
    if (vrefEntry->vrefs == 0)
      nvlrt_free_nv(heap, allocOff, alloc);
    else {
      // NVLRT_TX_ADD/persist only the allocation header not the data, which
      // we don't change here.
      NVLRT_TX_ADD_BARE(alloc, 1, sizeof *alloc);
      alloc->nvrefs = 0;
      nvlrt_vrefOnly_add(heap, allocOff, alloc);
      NVLRT_PMEMOBJ_PERSIST(pop, alloc);
    }
  }
  else {
    NVLRT_TX_ADD_BARE(&alloc->nvrefs, 1, sizeof alloc->nvrefs);
    --alloc->nvrefs;
    NVLRT_PMEMOBJ_PERSIST(pop, &alloc->nvrefs);
  }
}

// The only case in which nvlrt_dec_v modifies NVM is when it frees an
// allocation, and it encloses that free in a transaction internally, so the
// caller need not worry about enclosing call in a transaction.
void nvlrt_dec_v(nvlrt_v2nv_t p, nvlrt_v2nv_t *pp) {
  // This function must not fail after the heap is closed. Otherwise, all
  // V-to-NV pointers would be required to go out of scope before closing
  // the heap, and that's an unintuitive requirement for C programs.
  if (nvlrt_v2nv_isNull(p))
    return;
  nvlrt_heap_t *heap = p.heap;
  nvlrt_alloc_t *alloc = p.alloc;
  VrefEntry *vrefEntry = VrefTable_get(&heap->vrefTable, alloc);
  assert(vrefEntry->vrefs && vrefEntry->vrefs <= heap->vrefs);
  --vrefEntry->vrefs;
  if (vrefEntry->v2nvPtr == pp)
    vrefEntry->v2nvPtr = 0;
  --heap->vrefs;
  if (vrefEntry->vrefs == 0) {
    assert(!vrefEntry->v2nvPtr);
    PMEMobjpool *pop = heap->pop;
    if (!pop) {
      if (heap->vrefs == 0)
        nvlrt_free(heap);
    }
    else if (alloc->nvrefs == 0) {
      NVLRT_TX_BEGIN(heap);
      nvlrt_vrefOnly_remove(heap, alloc, true);
      nvlrt_free_nv(heap, nvlrt_directToOff(pop, alloc), alloc);
      NVLRT_TX_END(heap);
    }
  }
}
#endif

#if !NVLRT_NOTXS || NVLRT_PERSIST
// The numElements and elementSize parameter types must be kept in sync with
// the nvl-lower-pointers LLVM pass.
void nvlrt_persist(nvlrt_v2nv_t p, size_t numElements, size_t elementSize) {
  if (nvlrt_v2nv_isNull(p))
    return;
# if !NVLRT_NOREFS
  nvlrt_check_closedHeap(p.heap);
# endif
  pmemobj_persist(p.heap->pop, p.obj, numElements*elementSize);
}
#endif

#if !NVLRT_NOTXS
// A nvlrt_tx_begin_ptr call always immediately precedes a store instruction
// for its p, so we can be sure that it is safe to pass writeFirst=true to
// nvlrt_tx_add.
// TODO: For simplicity, consider removing these _ptr functions (and their
// corresponding LLVM intrinsics) and having the compiler generate the code
// to implement them instead.
void nvlrt_tx_begin_ptr(nvlrt_v2nv_t p, size_t targetSize) {
  if (nvlrt_v2nv_isNull(p)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction begin for heap from null NVM"
            " pointer\n");
    exit(1);
  }
  nvlrt_tx_begin(p.heap);
  nvlrt_tx_add(p, 1, targetSize, false, true);
}
void nvlrt_tx_end_ptr(nvlrt_v2nv_t p) {
  if (nvlrt_v2nv_isNull(p)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction end for heap from null NVM"
            " pointer\n");
    exit(1);
  }
  nvlrt_tx_end(p.heap);
}

# if !NVLRT_NOMPI
void nvlrt_tx_begin_mpiGroup(nvlrt_heap_t *heap, MPI_Group mpiTxGroup) {
#  if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
#  endif

  // If specified, record MPI tx group.
  if (mpiTxGroup != MPI_GROUP_NULL) {
    if (heap->txNestLevel) {
      fprintf(stderr, NVLRT_PREMSG"error: nested MPI group transaction\n");
      exit(1);
    }
    assert(heap->mpiTxGroup == MPI_GROUP_NULL);
    if (heap->mpiTxWorld == MPI_GROUP_NULL) {
      fprintf(stderr, NVLRT_PREMSG"error: MPI group transaction on heap"
                      " that is in local transaction mode\n");
      exit(1);
    }
    // Set the MPI tx group, which should currently be empty because ending
    // a transaction or recovering after a failure guarantees this.
    PMEMobjpool *pop = heap->pop;
    nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
    assert(!nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxGroupToWorld));
    nvlrt_mpiRank_t groupSize;
    // Compute as MIP_COMM_WORLD ranks.
    nvlrt_mpiRank_t *groupRanks = nvlrt_mpiGroupToWorldRanks(mpiTxGroup,
                                                             &groupSize);
    // Convert to indices into mpiTxWorld.
    nvlrt_mpiRank_t worldSize
      = nvlrt_mpiRanks_getSize(pop, &pmemRoot->mpiTxWorld);
    nvlrt_mpiRank_t *worldRanks
      = nvlrt_mpiRanks_getArray(pop, &pmemRoot->mpiTxWorld);
    bool foundSelf = false;
    for (nvlrt_mpiRank_t i = 0; i < groupSize; ++i) {
      nvlrt_mpiRank_t j;
      for (j = 0; j < worldSize; ++j) {
        if (groupRanks[i] == worldRanks[j]) {
          groupRanks[i] = j;
          if (worldRanks[j] == pmemRoot->mpiWorldRank)
            foundSelf = true;
          break;
        }
      }
      if (j == worldSize) {
        fprintf(stderr,
                NVLRT_PREMSG"error: transaction's MPI group is not a subset"
                " of the heap's MPI transaction world\n");
        exit(1);
      }
    }
    if (!foundSelf) {
      fprintf(stderr, NVLRT_PREMSG"error: transaction's MPI group does not"
                      " include the process's own MPI rank\n");
      exit(1);
    }
    nvlrt_mpiRanks_setGroup(pop, &pmemRoot->mpiTxGroupToWorld, groupRanks,
                            groupSize);
    heap->mpiTxGroup = mpiTxGroup;
  }

  // Start transaction. This must come after MPI data modifications in NVM,
  // which shouldn't be rolled back with the user transaction.
  nvlrt_tx_begin(heap);
}
void nvlrt_tx_mpiCommitBarrier(void *data) {
  nvlrt_heap_t *heap = data;

  // Barrier across tx group.
  MPI_Comm comm;
  MPI_Comm_create(MPI_COMM_WORLD, heap->mpiTxGroup, &comm);
  // TODO: Set comm error handler to fail?
  // TODO: Cache comm in nvlrt_heap_t until group changes? Will that help
  // performance?
  MPI_Barrier(comm);
  MPI_Comm_free(&comm);

  // Set tx state to committed.
  PMEMobjpool *pop = heap->pop;
  nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
  nvlrt_mpiTxState_t *active = &pmemRoot->mpiTxStates[pmemRoot
                                                     ->mpiTxStateActive];
  active->committed = true;
  pmemobj_persist(pop, &active->committed, sizeof active->committed);
}
# endif

void nvlrt_tx_begin(nvlrt_heap_t *heap) {
# if !NVLRT_NOREFS
  nvlrt_check_closedHeap(heap);
# endif
  if (pmemobj_tx_begin(heap->pop, 0, TX_PARAM_NONE)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction begin failed: %s\n",
            pmemobj_errormsg());
    exit(1);
  }
# if !NVLRT_NOMPI
  ++heap->txNestLevel;
# endif
}
void nvlrt_tx_end(nvlrt_heap_t *heap) {
  if (pmemobj_tx_stage() == TX_STAGE_ONABORT) {
    // TODO: It is not clear from the libpmemobj man page that
    // pmemobj_errormsg will always give the right error message here, but
    // it has diagnosed out-of-memory cases for us here.
    fprintf(stderr, NVLRT_PREMSG"error: transaction aborted: %s\n",
            pmemobj_errormsg());
    exit(1);
  }
  if (pmemobj_tx_stage() != TX_STAGE_WORK) {
    fprintf(stderr, NVLRT_PREMSG"error: transaction in invalid state\n");
    exit(1);
  }
# if !NVLRT_NOMPI
  if (heap->txNestLevel == 1 && heap->mpiTxGroup != MPI_GROUP_NULL) {
    pmemobj_tx_commit_hook(nvlrt_tx_mpiCommitBarrier, heap);
    nvlrt_mpiTxGroupIncCounts(heap);
    if (pmemobj_tx_end()) {
      fprintf(stderr, NVLRT_PREMSG"error: transaction end failed: %s\n",
              pmemobj_errormsg());
      exit(1);
    }
    nvlrt_root_t *pmemRoot = nvlrt_get_pmemRoot(heap);
    assert(pmemRoot);
    nvlrt_mpiRanks_clear(heap->pop, &pmemRoot->mpiTxGroupToWorld);
    heap->mpiTxGroup = MPI_GROUP_NULL;
  }
  else
# endif
  {
    pmemobj_tx_commit();
    if (pmemobj_tx_end()) {
      fprintf(stderr, NVLRT_PREMSG"error: transaction end failed: %s\n",
              pmemobj_errormsg());
      exit(1);
    }
  }
# if !NVLRT_NOMPI
  --heap->txNestLevel;
# endif
}

# if !NVLRT_NOREFS
#  include "prof/prof-tx-add.h"
static bool zeroShadowUpdateAlloc = false;
static double shadowUpdateAllocTime = 0;
// Always set this via nvlrt_setShadowUpdateCostMode.
static nvlrt_cost_mode_t shadowUpdateCostMode
  = SHADOW_UPDATE_COST_MODE_DEFAULT;

nvlrt_cost_mode_t nvlrt_getShadowUpdateCostMode() {
  return shadowUpdateCostMode;
}
void nvlrt_setShadowUpdateCostMode(nvlrt_cost_mode_t cost) {
  if (cost == NVLRT_COST_DEFAULT)
    shadowUpdateCostMode = SHADOW_UPDATE_COST_MODE_DEFAULT;
  else
    shadowUpdateCostMode = cost;
}
void nvlrt_zeroShadowUpdateAlloc() {
  zeroShadowUpdateAlloc = true;
}
double nvlrt_get_shadowUpdateAllocTime() {
  return shadowUpdateAllocTime;
}

//#  define DBG_SHADOW(...) fprintf(stderr, __VA_ARGS__)
#  define DBG_SHADOW(...)
static nvlrt_v2nv_t nvlrt_tx_tryShadowUpdate(
  nvlrt_v2nv_t p, size_t numElements, size_t elementSize, bool writeFirst)
{
  nvlrt_heap_t *heap = p.heap;
  PMEMobjpool *pop = heap->pop;
  DBG_SHADOW("SHADOW: --------------------------\n");
  DBG_SHADOW("SHADOW: tx_add p.obj: %p\n", p.obj);
  DBG_SHADOW("SHADOW: tx_add numElements: %zu\n", numElements);
  DBG_SHADOW("SHADOW: tx_add elementSize: %zu\n", elementSize);

  // Skip shadow update if we approximate it would have more overhead than
  // undo logging.
  //
  // Constants used below are specified in prof-tx-add.h, included above.
  // TODO: Currently, those constants are hardcoded in that file for the
  // ioScale device on megatron. Instead, we need some way to detect the
  // device on which an NVM file is stored and then use an associated
  // profile or set shadowUpdateCostMode = NVLRT_COST_INFINITE if there
  // isn't one yet. Of course, the profile might vary based on the machine
  // in which the device is installed rather than just on the type of
  // device.
  //
  // TODO: Handle the case where a substantial portion of the data to be
  // logged was previously logged. In that case, undo logging has
  // dramatically less overhead because there's no new undo log to create.
  //
  // TODO: The cost model might ought to consider that msync is always
  // called at page boundaries. These web pages suggest the right way to
  // compute the page size:
  //
  //     http://linux.die.net/man/2/getpagesize
  //     http://pubs.opengroup.org/onlinepubs/7908799/xsh/sysconf.html
  DBG_SHADOW("SHADOW: p.alloc->numElements: %zu\n", p.alloc->numElements);
  DBG_SHADOW("SHADOW: p.alloc->elementSize: %zu\n", p.alloc->elementSize);
  size_t logSize = numElements * elementSize;
  size_t allocSize = p.alloc->numElements * p.alloc->elementSize;
  switch (shadowUpdateCostMode) {
  case NVLRT_COST_DEFAULT:
    assert(!"shadowUpdateCostMode=NVLRT_COST_DEFAULT");
    break;
  case NVLRT_COST_ZERO:
    DBG_SHADOW("SHADOW: shadowUpdateCostMode=NVLRT_COST_ZERO\n");
    break;
  case NVLRT_COST_INFINITE:
    DBG_SHADOW("SHADOW: shadowUpdateCostMode=NVLRT_COST_INFINITE\n");
    DBG_SHADOW("SHADOW: -------------------\n");
    return nvlrt_v2nv_null;
  case NVLRT_COST_COMPUTE: {
    DBG_SHADOW("SHADOW: shadowUpdateCostMode=NVLRT_COST_COMPUTE\n");
    bool m = heap->usesMsync;
    DBG_SHADOW("SHADOW: %s msync\n", m ? "uses" : "does not use");
    size_t cutoff = m ? shadowUpdateMsyncCutoff : shadowUpdateClflushCutoff;
    if (allocSize <= cutoff) {
      DBG_SHADOW("SHADOW: allocSize (%zu) <= cutoff (%zu)\n",
                 allocSize, cutoff);
      DBG_SHADOW("SHADOW: -------------------\n");
      return nvlrt_v2nv_null;
    }
    // Fits are to the form y=a*x^b+c where x is allocation size in bytes
    // and y is the overhead for one of the following:
    // - Undo: undo logging
    // - Sbwf: shadow updating when writeFirst=true
    // - Write: writes to NVM
    // Write is added to Sbwf to compute overhead of shadow updating when
    // writeFirst=false.
    double undoA = m ? shadowUpdateMsyncUndoA : shadowUpdateClflushUndoA;
    double undoB = m ? shadowUpdateMsyncUndoB : shadowUpdateClflushUndoB;
    double undoC = m ? shadowUpdateMsyncUndoC : shadowUpdateClflushUndoC;
    double sbwfA = m ? shadowUpdateMsyncSbwfA : shadowUpdateClflushSbwfA;
    double sbwfB = m ? shadowUpdateMsyncSbwfB : shadowUpdateClflushSbwfB;
    double sbwfC = m ? shadowUpdateMsyncSbwfC : shadowUpdateClflushSbwfC;
    double undo = undoA * pow(logSize, undoB) + undoC;
    double shdw = sbwfA * pow(allocSize, sbwfB) + sbwfC;
    DBG_SHADOW("SHADOW: writeFirst=%s\n", writeFirst ? "true" : "false");
    if (!writeFirst) {
      double writeA = m?shadowUpdateMsyncWriteA : shadowUpdateClflushWriteA;
      double writeB = m?shadowUpdateMsyncWriteB : shadowUpdateClflushWriteB;
      double writeC = m?shadowUpdateMsyncWriteC : shadowUpdateClflushWriteC;
      shdw += ((double)logSize) / allocSize
              * writeA * pow(allocSize, writeB) + writeC;
    }
    DBG_SHADOW("SHADOW: undo logging  cost: %g sec\n", undo);
    DBG_SHADOW("SHADOW: shadow update cost: %g sec\n", shdw);
    if (undo < shdw) {
      DBG_SHADOW("SHADOW: shadow update is too costly\n");
      DBG_SHADOW("SHADOW: --------------------------\n");
      return nvlrt_v2nv_null;
    }
    break;
  }
  }

  // Skip shadow update if we can't update all pointers to this allocation.
  //
  // TODO: For now, we also skip if the allocation contains pointers. We
  // need to copy over the linked list starting at firstPtrOff but adjust
  // it for the new allocation address. Use a loop like the one in
  // nvlrt_free_nv. Something like:
  //
  //   ptrdiff_t newMinusOld = (uintptr_t)pNew.alloc - (uintptr_t)p.alloc;
  //   uint64_t *ptrOffOld = &p.alloc->firstPtrOff;
  //   uint64_t *ptrOffNew = &pNew.alloc->firstPtrOff;
  //   while (*ptrOffOld) {
  //     *ptrOffNew = *ptrOffOld + newMinusOld;
  //     nvlrt_nv2nv_t *nv2nvOld = nvlrt_offToDirect(pop, *ptrOffOld);
  //     nvlrt_nv2nv_t *nv2nvNew = nvlrt_offToDirect(pop, *ptrOffNew);
  //     ptrOffOld = &nv2nvOld->nextPtrOff;
  //     ptrOffNew = &nv2nvNew->nextPtrOff;
  //   }
  //
  // However, there are a few issues. First, we must do this after any
  // memcpy that might overwrite the nextPtrOff fields. Second, we also need
  // to increment the NV-to-NV reference counts on these pointers. However,
  // we currently free the old allocation before memcpy, and that's already
  // decremented the NV-to-NV reference counts on these pointers and thus
  // potentially freed their target allocations. Third, if we inc before dec
  // instead to overcome that problem, then we'll lose the encoding of the
  // pointers within nvrefs fields of the target allocations. We might
  // temporarily inc V-to-NV ref counts to make dec before inc safe, or we
  // might manually rewrite nvrefs fields. The latter seems harder to
  // maintain as code evolves, and the former seems logical because we're
  // temporarily working with those allocations locally. Finally, do we need
  // special consideration for a self-reference or any other existing
  // NV-to-NV pointer to this allocation?
  uint64_t nvrefs = p.alloc->nvrefs;
  VrefEntry *vrefEntry = VrefTable_get(&heap->vrefTable, p.alloc);
  DBG_SHADOW("SHADOW: tx_add nvrefs: 0x%016lx\n", nvrefs);
  DBG_SHADOW("SHADOW: tx_add vrefs:  0x%016lx\n", vrefEntry->vrefs);
  DBG_SHADOW("SHADOW: tx_add v2nvPtr:  0x%016lx\n", vrefEntry->v2nvPtr);
  if ((nvrefs > 0 && !(nvrefs & NVLRT_NVREFS_H1))
      || vrefEntry->vrefs > 2
      || (vrefEntry->vrefs == 2 && !vrefEntry->v2nvPtr)
      || p.alloc->firstPtrOff)
  {
    DBG_SHADOW("SHADOW: too many pointers or contained pointers\n");
    DBG_SHADOW("SHADOW: --------------------------\n");
    return nvlrt_v2nv_null;
  }
  // vrefs=0 is impossible because of p.
  assert(vrefEntry->vrefs);
  nvlrt_nv2nv_t *nv2nvPtr = nvrefs ? nvlrt_offToDirect(pop, nvrefs << 1)
                                   : NULL;
  nvlrt_v2nv_t *v2nvPtr = vrefEntry->v2nvPtr;

  // Skip shadow update if we don't have enough memory for a new allocation.
  //
  // We do not bother to initialize the allocation because that can waste a
  // lot of time that normal undo logging wouldn't waste. However, we can
  // get away with not initializing only because the allocation is
  // guaranteed not to contain pointers. Also, below we'll memcpy over any
  // (non-pointer) bytes that the transaction body won't write before
  // reading.
  double allocTimeStart = nvlrt_time();
  nvlrt_v2nv_t pNew = nvlrt_alloc_nv_optZero(heap, p.alloc->numElements,
                                             p.alloc->elementSize, NULL,
                                             zeroShadowUpdateAlloc);
  shadowUpdateAllocTime += nvlrt_time() - allocTimeStart;
  if (nvlrt_v2nv_isNull(pNew)) {
    DBG_SHADOW("SHADOW: out of memory on new allocation\n");
    DBG_SHADOW("SHADOW: --------------------------\n");
    return nvlrt_v2nv_null;
  }

  // We have decided to perform the shadow update and we have a new
  // allocation.
  DBG_SHADOW("SHADOW: shadow update\n");

  // Free the old allocation.
  //
  // Because the transaction has not yet committed, the allocation's data
  // should remain in place even though it's freed, so we can still read
  // from it below.
  //
  // TODO: Originally, we decided to free the allocation before copying its
  // data to the new allocation so that an indiscriminate msync wouldn't
  // have to sync the new allocation's data when sync'ing the undo log for
  // the free.  That is, we had read that, on Linux, msync syncs everything
  // in the file that's changed. However, our performance measurements imply
  // that's not true, at least with the installation of Linux on megatron.
  if (nv2nvPtr)
    nvlrt_dec_nv(p, nv2nvPtr);
  if (v2nvPtr)
    nvlrt_dec_v(p, v2nvPtr);
  nvlrt_dec_v(p, NULL);

  // Compute addresses for various parts of the allocation.
  //
  // Most of this is currently unused, but dead-code eliminate should remove
  // it.
  uintptr_t begDatOld = (uintptr_t)p.alloc->data;
  uintptr_t begDatNew = (uintptr_t)pNew.alloc->data;
  uintptr_t begLogOld = (uintptr_t)p.obj;
  ptrdiff_t begGap = begLogOld - begDatOld;
  uintptr_t begLogNew = begDatNew + begGap;
  uintptr_t endLogOld = begLogOld + logSize;
  uintptr_t endLogNew = begLogNew + logSize;
  uintptr_t endDatOld = begDatOld + allocSize;
  ptrdiff_t endGap = endDatOld - endLogOld;
  DBG_SHADOW("SHADOW: offsetof data: %u\n", offsetof(nvlrt_alloc_t, data));
  DBG_SHADOW("SHADOW: begDatOld: %p\n", (void*)begDatOld);
  DBG_SHADOW("SHADOW: begLogOld: %p\n", (void*)begLogOld);
  DBG_SHADOW("SHADOW: endLogOld: %p\n", (void*)endLogOld);
  DBG_SHADOW("SHADOW: endDatOld: %p\n", (void*)endDatOld);
  DBG_SHADOW("SHADOW: begDatNew: %p\n", (void*)begDatNew);
  DBG_SHADOW("SHADOW: begLogNew: %p\n", (void*)begLogNew);
  DBG_SHADOW("SHADOW: endLogNew: %p\n", (void*)endLogNew);
  DBG_SHADOW("SHADOW: begGap: %tu\n", begGap);
  DBG_SHADOW("SHADOW: endGap: %tu\n", endGap);
  assert(begGap >= 0);
  assert(endGap >= 0);

  // Adjust new obj field to beginning of log. This currently isn't useful,
  // but it conceptually makes sense that the pointer returned (pNew) is the
  // replacement for the pointer received (p).
  pNew.obj = (void*)begLogNew;

  // Update the NV-to-NV pointer and V-to-NV pointer, if any, to point to
  // the new allocation.
  if (nv2nvPtr) {
    ptrdiff_t objMinusAlloc = nv2nvPtr->objOff - nv2nvPtr->allocOff;
    NVLRT_TX_ADD_BARE(nv2nvPtr, 1, sizeof *nv2nvPtr);
    nv2nvPtr->allocOff = nvlrt_directToOff(pop, pNew.alloc);
    nv2nvPtr->objOff = nv2nvPtr->allocOff + objMinusAlloc;
    nvlrt_inc_nv(pNew, nv2nvPtr);
  }
  if (v2nvPtr) {
    ptrdiff_t objMinusAlloc = (uintptr_t)v2nvPtr->obj
                              - (uintptr_t)v2nvPtr->alloc;
    v2nvPtr->alloc = pNew.alloc;
    v2nvPtr->obj = (void*)((uintptr_t)v2nvPtr->alloc + objMinusAlloc);
    nvlrt_inc_v(pNew, v2nvPtr);
  }

  DBG_SHADOW("SHADOW: tx_add p.obj: %p\n", p.obj);
  DBG_SHADOW("SHADOW: tx_add nvrefs: 0x%016lx\n", p.alloc->nvrefs);
  DBG_SHADOW("SHADOW: tx_add vrefs:  0x%016lx\n", vrefEntry->vrefs);
  DBG_SHADOW("SHADOW: tx_add v2nvPtr:  0x%016lx\n", vrefEntry->v2nvPtr);
  DBG_SHADOW("SHADOW: tx_add pNew.obj: %p\n", pNew.obj);
  DBG_SHADOW("SHADOW: tx_add nvrefs: 0x%016lx\n", pNew.alloc->nvrefs);
  DBG_SHADOW("SHADOW: tx_add vrefs:  0x%016lx\n",
             VrefTable_get(&heap->vrefTable, pNew.alloc)->vrefs);
  DBG_SHADOW("SHADOW: tx_add v2nvPtr:  0x%016lx\n",
             VrefTable_get(&heap->vrefTable, pNew.alloc)->v2nvPtr);

  // Copy over data.
  //
  // To improve performance, we avoid copying data that's about to be
  // overwritten anyway. How do we determine what data is about to be
  // overwritten? We might assume that's exactly the logged data. However,
  // there are two challenging problems here. (1) Due to log aggregation,
  // some of the logged data might not be written by the transaction. (2)
  // Some of the logged data might be read before written. Fortunately,
  // writeFirst=true means that either the compiler or a user clause has
  // specified that neither of those cases is true: after this tx.add, all
  // logged data is written and is not read before being written. If
  // writeFirst=false instead, we play it safe by copying all of the old
  // allocation's data to the new allocation. TODO: The most important
  // benefit of the shadow update is reducing msync and, according to our
  // experiments so far, is achieved well even if writeFirst=false. If
  // performance measurements show we need something better for some apps,
  // especially in the case of future NVM technologies that don't require an
  // msync, we might want to find better ways to handle writeFirst=false.
  // See cost model above.
  if (writeFirst) {
    DBG_SHADOW("SHADOW: writeFirst=true\n");
    memcpy((void*)begDatNew, (void*)begDatOld, begGap);
    memcpy((void*)endLogNew, (void*)endLogOld, endGap);
  }
  else {
    DBG_SHADOW("SHADOW: writeFirst=false\n");
    memcpy((void*)begDatNew, (void*)begDatOld, endDatOld-begDatOld);
  }

  DBG_SHADOW("SHADOW: --------------------------\n");
  return pNew;
}
# endif

static double txAddTime = 0;
double nvlrt_get_txAddTime() { return txAddTime; }

// In the case of multi-dimensional arrays, numElements and elementSize
// might be in terms of a nested array's element type, but
// p.alloc->numElements and p.alloc->elementSize are always in terms of the
// outermost array element type.
nvlrt_v2nv_t nvlrt_tx_add(nvlrt_v2nv_t p, size_t numElements,
                          size_t elementSize, bool tryShadowUpdate,
                          bool writeFirst)
{
  double txAddTimeStart = nvlrt_time();
  if (nvlrt_v2nv_isNull(p)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction add for null NVM pointer\n");
    exit(1);
  }
# if !NVLRT_NOREFS
  nvlrt_check_closedHeap(p.heap);
# endif
# if !NVLRT_NOREFS
  if (tryShadowUpdate) {
    nvlrt_v2nv_t pNew = nvlrt_tx_tryShadowUpdate(p, numElements,
                                                 elementSize, writeFirst);
    if (!nvlrt_v2nv_isNull(pNew)) {
      txAddTime += nvlrt_time() - txAddTimeStart;
      return pNew;
    }
  }
# endif
  PMEMoid allocOID = nvlrt_directToOID(p.heap->pop, p.heap->pool_uuid_lo,
                                       p.alloc);
  uintptr_t allocObj = (uintptr_t)p.obj - (uintptr_t)p.alloc;
  // The PMEMoid passed to pmemobj_tx_add_range must point to the start of a
  // pmemobj object (that is, an NVM allocation), and the second parameter
  // must be the offset within the pmemobj object.  Otherwise, as of pmem
  // 0.3, pmemobj_tx_add_range won't recognize the PMEMoid as being for an
  // pmemobj object whose allocation (that is, the action of allocating it)
  // has been marked as already committed, so it will assume it hasn't been
  // committed, and so it will assume that adding its memory to the
  // transaction again is redundant.  pmemobj_tx_add_range_direct won't have
  // this problem because it doesn't receive an PMEMoid and so it just
  // always blindly assumes the add is necessary.
  if (pmemobj_tx_add_range(allocOID, allocObj, numElements*elementSize)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction add failed for NVM pointer:"
                        " %s\n",
            pmemobj_errormsg());
    exit(1);
  }
  txAddTime += nvlrt_time() - txAddTimeStart;
  return p;
}

nvlrt_v2nv_t nvlrt_tx_add_alloc(nvlrt_v2nv_t p, bool tryShadowUpdate,
                                bool writeFirst)
{
  // nvlrt_tx_add also checks nvlrt_v2nv_isNull, but that would fail an
  // assertion for dataV2nv, which always has a non-null obj field.
  if (nvlrt_v2nv_isNull(p)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction add for null NVM pointer\n");
    exit(1);
  }
  nvlrt_v2nv_t dataV2nv = {p.heap, p.alloc, &p.alloc->data};
  return nvlrt_tx_add(dataV2nv, p.alloc->numElements, p.alloc->elementSize,
                      tryShadowUpdate, writeFirst);
}

// TODO: Should we eliminate uses of this function (via NVLRT_TX_ADD_BARE)
// because pmemobj_tx_add_range_direct misses cases where a range need not
// be added to the tx? See comments in nvlrt_tx_add for details.  This
// function is prototyped in common-pmemobj.h.
void nvlrt_tx_add_bare(void *p, size_t numElements, size_t elementSize) {
  if (p == 0) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction add for null bare NVM"
            " pointer\n");
    exit(1);
  }
  if (pmemobj_tx_add_range_direct(p, numElements*elementSize)) {
    fprintf(stderr,
            NVLRT_PREMSG"error: transaction add failed for bare NVM"
                        " pointer: %s\n",
            pmemobj_errormsg());
    exit(1);
  }
}
#endif

void nvlrt_resetStats() {
#if !NVLRT_NOREFS
  numAllocNV = numFreeNV = 0;
#endif
#if !NVLRT_NOTXS
  txAddTime = 0;
# if !NVLRT_NOREFS
  shadowUpdateAllocTime = 0;
# endif
#endif
}

size_t nvlrt_get_sizeofNvlHeapT(void) {
  return sizeof(nvlrt_heap_t);
}

size_t nvlrt_usesMsync(nvlrt_heap_t *heap) {
  return heap->usesMsync;
}

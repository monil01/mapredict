#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <libpmemobj.h>

typedef struct VrefTable {
  struct cuckoo *cuckoo;
  struct VrefEntryNode *head;
} VrefTable;

typedef struct VrefEntry {
  // The total V-to-NV pointer count.
  size_t vrefs;
  // The virtual memory address at which one of those V-to-NV pointers is
  // stored, or null if none of their addresses are recorded.
  struct nvlrt_v2nv_t *v2nvPtr;
} VrefEntry;

int VrefTable_init(VrefTable *table);
void VrefTable_deinit(VrefTable *table);
VrefEntry *VrefTable_get(VrefTable *table, void *alloc);

typedef struct VrefOnlyNode VrefOnlyNode;
struct VrefOnlyNode {
  uint64_t prevOff; // off field of prev node OID
  uint64_t nextOff; // off field of next node OID
  uint64_t allocOff; // off field of allocation that has only vrefs
};

// The correct initialization of this data structure is all bits zero.
// nvlrt_create_or_open currently requires this property so that
// nvlrt_create behaves correctly in the case of unexpected application
// termination. See comments within nvlrt_create_or_open for details.
typedef struct VrefOnlyList {
  uint64_t headOff; // off field of head node OID
  // TODO: Allocating one VrefOnlyNode at a time probably wastes significant
  // memory given that pmemobj_alloc doesn't create small allocations.
  // Moreover, allocating and deallocating VrefOnlyNode objects every time
  // a nvrefs inc/dec from/to zero probably wastes time. We should memory
  // pool: allocate large arrays of VrefOnlyNode objects instead of
  // individual objects, and keep a linked list of free objects.
} VrefOnlyList;

// Caller must enclose call in a transaction.
uint64_t VrefOnlyList_put(VrefOnlyList *list, PMEMobjpool *pop,
                          uint64_t pool_uuid_lo, uint64_t allocOff);
// Does not modify NVM, so caller need not worry about enclosing call in a
// transaction.
uint64_t VrefOnlyList_get(VrefOnlyList *list, PMEMobjpool *pop,
                          uint64_t pool_uuid_lo);
// Caller must enclose call in a transaction.
void VrefOnlyList_remove(VrefOnlyList *list, PMEMobjpool *pop,
                         uint64_t pool_uuid_lo, uint64_t nodeOff);

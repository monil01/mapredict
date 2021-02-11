// Building with -DNVLRT_NOTXS disables runtime support for transactions,
// but then nvlrt-pmemobj.c should also be built with -DNVLRT_NOTXS.
// Building with -DNVLRT_PERSIST enables runtime support for persist calls
// after stores, but then nvlrt-pmemobj.c should also be built with
// -DNVLRT_PERSIST. See ../README.

#include "vrefs-pmemobj.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
// libpmemobj is not designed to expose its cuckoo hash table
// implementation, so we might have to provide our own at some point.
#include <../libpmemobj/cuckoo.h>
#include "common-pmemobj.h"
#include "killpos.h"

typedef struct VrefEntryNode {
  VrefEntry entry;
  struct VrefEntryNode *next;
} VrefEntryNode;

int VrefTable_init(VrefTable *table) {
  errno = 0;
  table->cuckoo = cuckoo_new();
  if (!table->cuckoo) {
    if (errno == 0)
      errno = ENOMEM;
    return errno;
  }
  table->head = NULL;
  return 0;
}

void VrefTable_deinit(VrefTable *table) {
  cuckoo_delete(table->cuckoo);
  VrefEntryNode *node = table->head;
  while (node) {
    VrefEntryNode *next = node->next;
    free(node);
    node = next;
  }
}

VrefEntry *VrefTable_get(VrefTable *table, void *alloc) {
  VrefEntry *entry = cuckoo_get(table->cuckoo, (uintptr_t)alloc);
  if (entry)
    return entry;
  VrefEntryNode *list = malloc(sizeof *list);
  if (!list) {
    fprintf(stderr, NVLRT_PREMSG"error: vref entry allocation failed: %s\n",
            strerror(errno));
    exit(1);
  }
  list->next = table->head;
  table->head = list;
  list->entry.vrefs = 0;
  list->entry.v2nvPtr = NULL;
  int err = cuckoo_insert(table->cuckoo, (uintptr_t)alloc, &list->entry);
  if (err) {
    fprintf(stderr, NVLRT_PREMSG"error: vref table insertion failed: %s\n",
            strerror(err));
    exit(1);
  }
  return &list->entry;
}

uint64_t VrefOnlyList_put(VrefOnlyList *list, PMEMobjpool *pop,
                          uint64_t pool_uuid_lo, uint64_t allocOff)
{
  PMEMoid nodeOID;
  // The entire allocation will be rolled back if the transaction does not
  // commit, so we don't need to NVLRT_TX_ADD the parts of the allocation
  // we write here.
  if (NVLRT_PMEMOBJ_ZALLOC(pop, &nodeOID, sizeof(VrefOnlyNode),
                           NVLRT_PMEM_TYPE_NUM))
  {
    fprintf(stderr,
            NVLRT_PREMSG"error: vref-only entry allocation failed: %s\n",
            strerror(errno));
    exit(1);
  }
  uint64_t nodeOff = nvlrt_oidToOff(nodeOID);
  VrefOnlyNode *node = nvlrt_offToDirect(pop, nodeOff);
  uint64_t oldHeadOff = list->headOff;
  NVLRT_TX_ADD_BARE(&list->headOff, 1, sizeof list->headOff);
  list->headOff = nodeOff;
  NVLRT_PMEMOBJ_PERSIST(pop, &list->headOff);
  node->nextOff = oldHeadOff;
  if (oldHeadOff) {
    VrefOnlyNode *oldHead = nvlrt_offToDirect(pop, oldHeadOff);
    NVLRT_TX_ADD_BARE(&oldHead->prevOff, 1, sizeof oldHead->prevOff);
    oldHead->prevOff = nodeOff;
    NVLRT_PMEMOBJ_PERSIST(pop, &oldHead->prevOff);
  }
  node->allocOff = allocOff;
  NVLRT_PMEMOBJ_PERSIST(pop, node);
  return nodeOff;
}

uint64_t VrefOnlyList_get(VrefOnlyList *list, PMEMobjpool *pop,
                          uint64_t pool_uuid_lo)
{
  return list->headOff;
}

void VrefOnlyList_remove(VrefOnlyList *list, PMEMobjpool *pop,
                         uint64_t pool_uuid_lo, uint64_t nodeOff)
{
  VrefOnlyNode *node = nvlrt_offToDirect(pop, nodeOff);
  uint64_t prevOff = node->prevOff;
  uint64_t nextOff = node->nextOff;
  if (!prevOff) {
    NVLRT_TX_ADD_BARE(&list->headOff, 1, sizeof list->headOff);
    list->headOff = nextOff;
    NVLRT_PMEMOBJ_PERSIST(pop, &list->headOff);
  }
  else {
    VrefOnlyNode *prev = nvlrt_offToDirect(pop, prevOff);
    NVLRT_TX_ADD_BARE(&prev->nextOff, 1, sizeof prev->nextOff);
    prev->nextOff = nextOff;
    NVLRT_PMEMOBJ_PERSIST(pop, &prev->nextOff);
  }
  if (nextOff) {
    VrefOnlyNode *next = nvlrt_offToDirect(pop, nextOff);
    NVLRT_TX_ADD_BARE(&next->prevOff, 1, sizeof next->prevOff);
    next->prevOff = prevOff;
    NVLRT_PMEMOBJ_PERSIST(pop, &next->prevOff);
  }
  PMEMoid nodeOID = nvlrt_offToOID(pool_uuid_lo, nodeOff);
  NVLRT_PMEMOBJ_FREE(&nodeOID);
}

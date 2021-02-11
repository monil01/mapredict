#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <nvl.h>
#include <nvlrt-test.h>
#include <libpmem.h>

#define NVLFILE "/opt/fio/scratch/jum/prof-tx-add.nvl"
#define TYPE float
#define TIME_FMT "%g"

struct timeval timer() {
  struct timeval time;
  gettimeofday(&time, 0);
  return time;
}
double subTimevals(struct timeval x, struct timeval y) {
  return x.tv_sec - y.tv_sec + (x.tv_usec - y.tv_usec)/1000000.;
}

enum Mode {UNDO, SHDW, WRITE, MODE_COUNT};
const char *modes[] = {"undo", "shdw", "write"};

void usage(const char *cmd) {
  fprintf(stderr, "Usage: %s N ", cmd);
  for (enum Mode mode = 0; mode < MODE_COUNT; ++mode) {
    if (mode != 0)
      fprintf(stderr, "|");
    fprintf(stderr, "%s", modes[mode]);
  }
  fprintf(stderr, "\n");
}

int main(int argc, const char *argv[]) {
  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }
  size_t n = atoi(argv[1]);
  enum Mode mode;
  for (mode = 0; mode < MODE_COUNT; ++mode) {
    if (0 == strcmp(argv[2], modes[mode]))
      break;
  }
  if (mode == MODE_COUNT) {
    usage(argv[0]);
    return 1;
  }
  size_t numElements = n*n;
  size_t allocSize = numElements * sizeof(TYPE);
  size_t heapSize = 5 * allocSize;

  // Print config.
  printf("N = %zu\n", n);
  printf("numElements = %zu\n", numElements);
  printf("elementSize = %zu bytes\n", sizeof(TYPE));
  printf("allocSize = %zu bytes\n", allocSize);
  printf("mode = %s\n", modes[mode]);

  // Create pool and allocations.
  nvl_heap_t *heap = nvl_create(NVLFILE, heapSize, 0600);
  if(!heap) {
    perror("nvl_create failed");
    return 1;
  }
  nvl TYPE *allocSrc = nvl_alloc_nv(heap, numElements, TYPE);
  if (!allocSrc) {
    perror("nvl_alloc_nv failed for source allocation");
    return 1;
  }
  nvl TYPE *allocDst = nvl_alloc_nv(heap, numElements, TYPE);
  if (!allocDst) {
    perror("nvl_alloc_nv failed for destination allocation");
    return 1;
  }
  printf("is_pmem = %d\n", pmem_is_pmem(nvl_bare_hack(allocDst), allocSize));

  // Create an NV-to-NV pointer to the destination allocation as that gives
  // the shadow update more work to do.
  nvl_set_root(heap, allocDst);

  // Initialize source allocation randomly.
  srand(timer().tv_usec);
  for (nvl TYPE *p = allocSrc, *e = p+numElements; p < e; ++p)
    *p = rand();

  // Run transaction twice and time the second run.
  //
  // The first run warms the cache and page buffering.
  //
  // In the case of modes UNDO and SHDW, warming makes sense because many
  // applications run the same transactions multiple times.
  //
  // In the case of WRITE mode, warming makes sense because the time is used
  // to estimate the time a transaction body spends writing to a
  // shadow-updated allocation for which backup_writeFirst is not in effect
  // and thus for which the shadow update already initialized the entire
  // allocation, thus warming it. In this case, warming can reduce the time
  // often by 75%, so it is especially critical to include.
  //
  // The way we use the time for WRITE is as follows. We assume WRITE time
  // is a fair approximation of the time for the full allocation write in
  // SHDW mode's transaction body, and we subtract WRITE time from SHDW
  // time. Thus, SHDW-WRITE time includes the one full allocation write
  // performed during the shadow update, but WRITE time includes the one
  // full allocation write performed by the transaction body.  However, we
  // assume the only significant difference between those two write times is
  // which came first (75% reduction for the second one) and not where in
  // the transaction it was performed. To estimate the time for a shadow
  // update with backup_writeFirst, which usually performs one full
  // allocation write, we just use SHDW-WRITE.  To estimate the time for a
  // shadow update without backup_writeFirst, which usually additionally
  // writes to logSize/allocSize of the allocation, we add
  // WRITE*logSize/allocSize to SHDW-WRITE.
  //
  // It might seem more reasonable to measure the time for the writes within
  // SHDW-mode transaction below instead of having a separate WRITE mode.
  // However, sometimes the OS performs syncs during writes, thus inflating
  // write time and usually deflating any subsequent sync time, such as the
  // sync time during the transaction commit. Adding backup_writeFirst to a
  // shadow update eliminates write time but usually does not eliminate sync
  // time: the entire allocation will still be sync'ed during the
  // transaction commit. Likewise for reducing the transaction body's writes
  // to logSize/allocSize when the entire allocation has already been
  // written as part of the shadow update. Thus, we want WRITE in our above
  // calculations to include only the actually write time not any sync time.
  // Somehow, performing memcpy twice in a row and measuring the second
  // memcpy seems to give us a reasonable estimate. Other approaches, such
  // as a for loop, have given us larger times apparently inflated by syncs.
  struct timeval start;
  struct timeval end;
  for (int i = 0; i < 2; ++i) {
    switch (mode) {
    case UNDO: {
      nvlrt_setShadowUpdateCostMode(NVLRT_COST_INFINITE);
      start = timer();
      #pragma nvl atomic heap(heap) default(readonly) \
                         backup(allocDst[0:numElements])
      for (nvl TYPE *pDst = allocDst, *eDst = pDst+numElements,
                    *pSrc = allocSrc;
           pDst < eDst; ++pDst, ++pSrc)
        *pDst = *pSrc;
      end = timer();
      break;
    }
    case SHDW: {
      nvlrt_setShadowUpdateCostMode(NVLRT_COST_ZERO);
      start = timer();
      #pragma nvl atomic heap(heap) default(readonly) \
                         backup(allocDst[0:numElements])
      for (nvl TYPE *pDst = allocDst, *eDst = pDst+numElements,
                    *pSrc = allocSrc;
           pDst < eDst; ++pDst, ++pSrc)
        *pDst = *pSrc;
      end = timer();
      break;
    }
    case WRITE: {
      start = timer();
      memcpy(nvl_bare_hack(allocDst), nvl_bare_hack(allocSrc),
             numElements*sizeof(TYPE));
      end = timer();
      break;
    }
    }
  }

  // Print time.
  printf("time = "TIME_FMT" sec\n", subTimevals(end, start));

  nvl_close(heap);
  return 0;
}

#include <assert.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#define PMEMOBJ 0
#define NVL 1
#define VHEAP 2
#if MEM == PMEMOBJ
# include <libpmemobj.h>
#elif MEM == NVL
# include <nvl.h>
# include <nvlrt-test.h>
# include <libpmem.h>
#elif MEM == VHEAP
# include <nvl-vheap.h>
#else
# error unknown MEM setting
#endif

#ifndef _N_
#define _N_ 2048
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/matmul.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/matmul.nvl"
#endif

#ifndef HEAPSIZE
#define HEAPSIZE 0
#endif

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#define N _N_
#define M _N_
#define P _N_
#pragma openarc #define N _N_
#pragma openarc #define M _N_
#pragma openarc #define P _N_

double
my_timer()
{
    struct timeval time;
    gettimeofday (&time, 0);
    return time.tv_sec + time.tv_usec / 1000000.0;
}

// Declaring MatrixMultiplication_* static significantly lowers the computed
// cost of inlining, which, in some builds of this code, LLVM does not
// perform otherwise.

#if MEM == PMEMOBJ

# if TXS
static void
MatrixMultiplication_nv(PMEMobjpool *pop, PMEMoid i_oid, PMEMoid a_oid,
                        PMEMoid b_oid, PMEMoid c_oid)
# else
static void
MatrixMultiplication_nv(PMEMoid a_oid, PMEMoid b_oid, PMEMoid c_oid)
# endif
{
# if !POOR
  float *a = pmemobj_direct(a_oid);
  float *b = pmemobj_direct(b_oid);
  float *c = pmemobj_direct(c_oid);
# endif
  int i, j, k;
# if TXS
  // Without this condition, the persist call and i_sub loop would have to
  // be adjusted for the last transaction.
  assert(M % ROWS_PER_TX == 0);
  for (i=*(int*)pmemobj_direct(i_oid); i<M;) {
    if (pmemobj_tx_begin(pop, 0, TX_LOCK_NONE)) {
      fprintf(stderr, "pmemobj_tx_begin failed: %s\n",
              pmemobj_errormsg());
      exit(1);
    }
    if (pmemobj_tx_add_range(i_oid, 0, sizeof(int))) {
      fprintf(stderr, "pmemobj_tx_add_range failed: %s\n",
              pmemobj_errormsg());
      exit(1);
    }
    for (int i_sub=0; i_sub<ROWS_PER_TX;
         ++i_sub, ++i, ++*(int*)pmemobj_direct(i_oid))
    {
# else
  for (i=0; i<M; i++) {
# endif
      for (j=0; j<N; j++) {
        float sum = 0.0;
        for (k=0; k<P; k++) {
# if POOR
          float *b = pmemobj_direct(b_oid);
          float *c = pmemobj_direct(c_oid);
# endif
          sum += b[i*P+k]*c[k*N+j];
        }
# if POOR
        float *a = pmemobj_direct(a_oid);
# endif
        a[i*N+j] = sum;
      }
# if TXS
    }
#  if POOR
    float *a = pmemobj_direct(a_oid);
#  endif
    pmemobj_persist(pop, &a[i*N], ROWS_PER_TX*N*sizeof(float));
    if (pmemobj_tx_stage() == TX_STAGE_ONABORT) {
      fprintf(stderr, "transaction aborted\n");
      exit(1);
    }
    pmemobj_tx_process();
    if (pmemobj_tx_end()) {
      fprintf(stderr, "pmemobj_tx_end failed: %s\n",
              pmemobj_errormsg());
      exit(1);
    }
# endif
  }
}

#elif MEM == NVL

#if TXS
static void
MatrixMultiplication_nv(nvl_heap_t *heap, nvl int *i_nv, nvl float * a_nv,
                        nvl float * b_nv, nvl float * c_nv)
#else
static void
MatrixMultiplication_nv(nvl float * a_nv, nvl float * b_nv,
                        nvl float * c_nv)
#endif
{
  // Because no NVM pointers are stored and no unknown functions are called,
  // it should be straight-forward for a compiler to recognize that all
  // V-to-NV incs and decs can be optimized away within this function.
# if !POOR
  float *a = nvl_bare_hack(a_nv);
  float *b = nvl_bare_hack(b_nv);
  float *c = nvl_bare_hack(c_nv);
# endif
  int i, j, k;
# if TXS
  // Without this condition, the clobber clause and i_sub loop would have to
  // be adjusted for the last transaction.
  assert(M % ROWS_PER_TX == 0);
  for (i=*i_nv; i<M;) {
#if (TXS == 1)
    #pragma nvl atomic heap(heap)
#elif (TXS == 2)
    #pragma nvl atomic heap(heap) default(readonly) \
            backup(i_nv[0:1], a_nv[i*N:ROWS_PER_TX*N])
#elif (TXS == 3)
    #pragma nvl atomic heap(heap) default(readonly) \
            backup(i_nv[0:1]) clobber(a_nv[i*N:ROWS_PER_TX*N])
#elif (TXS == 4)
    #pragma nvl atomic heap(heap) default(readonly) \
            backup(i_nv[0:1]), backup_writeFirst(a_nv[i*N:ROWS_PER_TX*N])
#endif
    for (int i_sub=0; i_sub<ROWS_PER_TX; ++i_sub, ++i, ++*i_nv) {
# else
  for (i=0; i<M; i++) {
# endif
      for (j=0; j<N; j++) {
        float sum = 0.0;
        for (k=0; k<P; k++) {
# if POOR
          sum += b_nv[i*P+k]*c_nv[k*N+j];
# else
          sum += b[i*P+k]*c[k*N+j];
# endif
        }
# if POOR || TXS
        a_nv[i*N+j] = sum;
# else
        a[i*N+j] = sum;
# endif
      }
# if TXS
    }
# endif
  }
# if !POOR && PERSIST
  nvl_persist_hack(a_nv, M*N);
  nvl_persist_hack(b_nv, M*P);
  nvl_persist_hack(c_nv, P*N);
# endif
}

#elif MEM == VHEAP

static void MatrixMultiplication_v(float * a, float * b, float * c);
static void MatrixMultiplication_nv(float * a, float * b, float * c) {
  MatrixMultiplication_v(a, b, c);
}

#endif

static void
MatrixMultiplication_v(float * a, float * b, float * c)
{
  int i, j, k;
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) {
      float sum = 0.0;
      for (k=0; k<P; k++) {
        sum += b[i*P+k]*c[k*N+j];
      }
      a[i*N+j] = sum;
    }
  }
}

#if MEM == NVL
struct root {
#if TXS
  int i;
#endif
  nvl float *a;
  nvl float *b;
  nvl float *c;
};
#endif

int
main()
{
#if MEM == NVL && TXS
    nvlrt_setShadowUpdateCostMode(SHADOW_UPDATE_COST_MODE);
#endif
#if MEM == PMEMOBJ
# if TXS
  PMEMoid i_nv_oid;
  int *i_nv;
# endif
  PMEMoid a_nv_oid, b_nv_oid, c_nv_oid;
  float *a_nv, *b_nv, *c_nv;
#elif MEM == NVL
# if TXS
  nvl int *i_nv = NULL;
# endif
  nvl float *a_nv = 0, *b_nv = 0, *c_nv = 0;
#elif MEM == VHEAP
  float *a_nv, *b_nv, *c_nv;
#endif
  float *a_v, *b_v, *c_v;
  int i, j;
  double elapsed_time;

#if MEM == PMEMOBJ
  PMEMobjpool *pop = pmemobj_create(NVLFILE, "",
                                    HEAPSIZE < PMEMOBJ_MIN_POOL
                                    ? PMEMOBJ_MIN_POOL : HEAPSIZE,
                                    0600);
  if (!pop) {
    perror("pmemobj_create failed");
    return 1;
  }
# if TXS
  if (pmemobj_zalloc(pop, &i_nv_oid, sizeof(int), 0)) {
    perror("pmemobj_zalloc failed");
    return 1;
  }
  i_nv = pmemobj_direct(i_nv_oid);
# endif
  if (pmemobj_alloc(pop, &a_nv_oid, M*N*sizeof(float), 0, NULL, NULL)
      || pmemobj_alloc(pop, &b_nv_oid, M*P*sizeof(float), 0, NULL, NULL)
      || pmemobj_alloc(pop, &c_nv_oid, P*N*sizeof(float), 0, NULL, NULL))
  {
    perror("pmemobj_alloc failed");
    return 1;
  }
  a_nv = pmemobj_direct(a_nv_oid);
  b_nv = pmemobj_direct(b_nv_oid);
  c_nv = pmemobj_direct(c_nv_oid);
#elif MEM == NVL
  // nvl_create uses min heap size if HEAPSIZE is less than min.
  nvl_heap_t *heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
  if (!heap) {
    perror("nvl_create failed");
    return 1;
  }
  nvl struct root *root_nv = 0;
  if (!(root_nv = nvl_alloc_nv(heap, 1, struct root))
      || !(a_nv = nvl_alloc_nv(heap, M*N, float))
      || !(b_nv = nvl_alloc_nv(heap, M*P, float))
      || !(c_nv = nvl_alloc_nv(heap, P*N, float)))
  {
    perror("nvl_alloc_nv failed");
    return 1;
  }
  nvl_set_root(heap, root_nv);
# if TXS
  i_nv = &root_nv->i;
# endif
  root_nv->a = a_nv;
  root_nv->b = b_nv;
  root_nv->c = c_nv;
#elif MEM == VHEAP
  nvl_vheap_t *vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
  if (!vheap) {
    perror("nvl_vcreate failed");
    return 1;
  }
  if (!(a_nv = (float *) nvl_vmalloc(vheap, M*N*sizeof(float)))
      || !(b_nv = (float *) nvl_vmalloc(vheap, M*P*sizeof(float)))
      || !(c_nv = (float *) nvl_vmalloc(vheap, P*N*sizeof(float))))
  {
    perror("nvl_vmalloc failed");
    return 1;
  }
#endif
  if (!(a_v = (float *) malloc(M*N*sizeof(float)))
      || !(b_v = (float *) malloc(M*P*sizeof(float)))
      || !(c_v = (float *) malloc(P*N*sizeof(float))))
  {
    perror("malloc failed");
    return 1;
  }

  {
#if MEM == NVL && PERSIST
    // The initialization loops aren't part of our timings, but they're
    // outrageously slow if persists are inserted within.
    float *a_nv_bare = nvl_bare_hack(a_nv);
    float *b_nv_bare = nvl_bare_hack(b_nv);
    float *c_nv_bare = nvl_bare_hack(c_nv);
    float *a_nv = a_nv_bare;
    float *b_nv = b_nv_bare;
    float *c_nv = c_nv_bare;
#endif
    for (i = 0; i <  M*N; i++) {
      a_nv[i] = (float) 0.0F;
      a_v[i]  = (float) 0.0F;
    }
    for (i = 0; i <  M*P; i++) {
      b_nv[i] = (float) i;
      b_v[i]  = (float) i;
    }
    for (i = 0; i <  P*N; i++) {
      c_nv[i] = (float) 1.0F;
      c_v[i]  = (float) 1.0F;
    }
  }
  printf("Input matrix size: M = %d, N = %d, P = %d\n", M, N, P);
#if MEM == PMEMOBJ
   printf("(pmemobj mode)\n");
#elif MEM == NVL
   printf("(NVL mode, is_pmem=%d, ",
          pmem_is_pmem(nvl_bare_hack(a_nv), M*N*sizeof(float)));
# if TXS
   printf("tx mode=%d, ROWS_PER_TX=%d, shdw cost mode=", TXS, ROWS_PER_TX);
   switch (nvlrt_getShadowUpdateCostMode()) {
   case NVLRT_COST_ZERO:     printf("zero");     break;
   case NVLRT_COST_COMPUTE:  printf("compute");  break;
   case NVLRT_COST_INFINITE: printf("infinite"); break;
   default: fprintf(stderr, "invalid cost mode\n"); exit(1);
   }
   printf(")\n");
# else
   printf("txs disabled)\n");
# endif
#elif MEM == VHEAP
   printf("(vheap mode)\n");
#endif

#if VERIFICATION == 1
  elapsed_time = my_timer();
  MatrixMultiplication_v(a_v, b_v, c_v);
  elapsed_time = my_timer() - elapsed_time;
  printf("Volatile Memory Elapsed time = %lf sec\n", elapsed_time);
#endif
  elapsed_time = my_timer();
#if MEM == PMEMOBJ
# if TXS
  MatrixMultiplication_nv(pop, i_nv_oid, a_nv_oid, b_nv_oid, c_nv_oid);
# else
  MatrixMultiplication_nv(a_nv_oid, b_nv_oid, c_nv_oid);
# endif
#elif MEM == NVL
# if TXS
  MatrixMultiplication_nv(heap, i_nv, a_nv, b_nv, c_nv);
# else
  MatrixMultiplication_nv(a_nv, b_nv, c_nv);
# endif
#elif MEM == VHEAP
  MatrixMultiplication_nv(a_nv, b_nv, c_nv);
#endif
  elapsed_time = my_timer() - elapsed_time;
  printf("NVM Elapsed time = %lf sec\n", elapsed_time);

#if VERIFICATION == 1
  {
    double sum_v = 0.0;
    double sum_nv = 0.0;
    double rel_err = 0.0;

    for (i=0; i<M*N; i++){
      sum_v += a_v[i]*a_v[i];
      sum_nv += a_nv[i]*a_nv[i];
    }

    sum_v = sqrt(sum_v);
    sum_nv = sqrt(sum_nv);
    if ( sum_v > sum_nv ) {
      rel_err = (sum_v-sum_nv)/sum_v;
    } else {
      rel_err = (sum_nv-sum_v)/sum_v;
    }

    if (rel_err < 1e-6) {
      printf("Verification Successful err = %e\n", rel_err);
    }
    else {
      printf("Verification Fail err = %e\n", rel_err);
    }
  }
#endif

  free(a_v);
  free(b_v);
  free(c_v);
#if MEM == PMEMOBJ
  pmemobj_close(pop);
#elif MEM == NVL
  nvl_close(heap);
#elif MEM == VHEAP
  nvl_vfree(vheap, a_nv);
  nvl_vfree(vheap, b_nv);
  nvl_vfree(vheap, c_nv);
  nvl_vclose(vheap);
#endif

  return 0;
}


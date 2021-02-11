#ifndef _COMMON_H
#define _COMMON_H

#include <assert.h>
#include <time.h>
#include <sys/time.h>

#define NULLZ 0

#define NVL 1
#define VHEAP 2
#if MEM == NVL
#include <nvl.h>
#define NVL_PREFIX nvl
extern nvl_heap_t *heap;
#elif MEM == VHEAP
#include <nvl-vheap.h>
#define NVL_PREFIX  
extern nvl_vheap_t *vheap;
#else
#define NVL_PREFIX  
#endif

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/lud.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/lud.nvl"
//#define NVLFILE "/tmp/f6l/lud.nvl"
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

#ifndef _M_SIZE
#define _M_SIZE 4096
#endif
#ifdef _OPENARC_
#pragma openarc #define _M_SIZE 4096
#endif

#ifndef ENABLE_OPENACC
#define ENABLE_OPENACC 0
#endif

#ifndef HEAPSIZE
#define HEAPSIZE (_M_SIZE * 4 * 3)
#endif


#ifdef __cplusplus
extern "C" {
#endif


#define GET_RAND_FP ( (float)rand() /   \
                     ((float)(RAND_MAX)+(float)(1)) )

#define MIN(i,j) ((i)<(j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

typedef struct __stopwatch_t{
    struct timeval begin;
    struct timeval end;
}stopwatch;

void 
stopwatch_start(stopwatch *sw);

void 
stopwatch_stop (stopwatch *sw);

double 
get_interval_by_sec(stopwatch *sw);

int 
get_interval_by_usec(stopwatch *sw);

func_ret_t
create_matrix_from_file(NVL_PREFIX float **mp, const char *filename, int *size_p);

func_ret_t
create_matrix_from_random(float **mp, int size);

func_ret_t
lud_verify(float *m, float *lu, int size);

void
matrix_multiply(float *inputa, float *inputb, float *output, int size);

void
matrix_duplicate(float *src, float **dst, int matrix_dim);

void
print_matrix(float *mm, int matrix_dim);

#ifdef __cplusplus
}
#endif

#endif

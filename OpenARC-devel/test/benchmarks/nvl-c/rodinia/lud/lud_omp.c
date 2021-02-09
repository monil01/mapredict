#include <stdio.h>
#include "common.h"

extern int omp_num_threads;

void lud_omp(NVL_PREFIX float * a, int size)
{
     int i,j,k;
     float sum;
#if (MEM == NVL) && !POOR
     float *a_v = nvl_bare_hack(a);
#endif
	 //printf("num of threads = %d\n", omp_num_threads);
#if TXS
	assert(size % ROWS_PER_TX == 0);
	for (i=*i_nv; i<size; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(i_nv[0:1], a[0:_M_SIZE])
#endif
	for (int i_sub=0; i_sub_ROWS_PER_TX; ++i_sub, ++i, ++*i_nv) {
#else
#if ENABLE_OPENACC == 1
#pragma acc data copy(a[0:_M_SIZE])
#endif
     for (i=0; i <size; i++){
#endif
#if ENABLE_OPENACC == 1
#pragma acc kernels loop gang, worker, private(j, k, sum)
#endif
         for (j=i; j <size; j++){
#if (MEM == NVL) && !POOR
             sum=a_v[i*size+j];
             for (k=0; k<i; k++) sum -= a_v[i*size+k]*a_v[k*size+j];
#if TXS
             a[i*size+j]=sum;
#else
             a_v[i*size+j]=sum;
#endif
#else
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
#endif
         }
#if PERSIST && (MEM == NVL) && !POOR
         nvl_persist_hack(a, size);
#endif

#if ENABLE_OPENACC == 1
#pragma acc kernels loop if(i+1<size) gang, worker, private(j, k, sum)
#endif
         for (j=i+1;j<size; j++){
#if (MEM == NVL) && !POOR
             sum=a_v[j*size+i];
             for (k=0; k<i; k++) sum -=a_v[j*size+k]*a_v[k*size+i];
#if TXS
             a[j*size+i]=sum/a_v[i*size+i];
#else
             a_v[j*size+i]=sum/a_v[i*size+i];
#endif
#else
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
#endif
         }
#if PERSIST && (MEM == NVL) && !POOR
         nvl_persist_hack(a, size);
#endif
#if TXS
     }
#endif
     }
}

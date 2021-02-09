#include <stdio.h>

#ifndef _M_SIZE
#define _M_SIZE 4096
#ifdef _OPENARC_
#pragma openarc #define _M_SIZE 4096
#endif
#endif

extern int omp_num_threads;

void lud_omp(float * a, int size)
{
     int i,j,k;
     float sum;
	 //printf("num of threads = %d\n", omp_num_threads);
#pragma omp target data device(0) map(tofrom:a[0:_M_SIZE])
     for (i=0; i <size; i++){
#pragma omp target teams distribute parallel for private(j, k, sum)
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }

#pragma omp target teams distribute parallel for private(j, k, sum) if(i+1<size)
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }
}

#include <stdio.h>
#include "resilience.h"

#ifndef _M_SIZE
#define _M_SIZE 4096
#ifdef _OPENARC_
#pragma openarc #define _M_SIZE 4096
#endif
#endif

#ifndef ENABLE_OPENACC
#define ENABLE_OPENACC 1
#endif

#ifndef R_MODE
#define R_MODE 0
#endif

#ifndef RES_REGION0
#define RES_REGION0 1
#endif
#ifndef RES_REGION1
#define RES_REGION1 0
#endif
#ifndef TOTAL_NUM_FAULTS
#define TOTAL_NUM_FAULTS    1
#endif
#ifndef NUM_FAULTYBITS
#define NUM_FAULTYBITS  1
#endif
#ifndef NUM_REPEATS
#define NUM_REPEATS 1
#endif
#ifndef _FTVAR
#define _FTVAR 0
#endif
#ifndef _FTKIND
#define _FTKIND 5
#endif
#ifndef _FTTHREAD
#define _FTTHREAD 0
#endif

#include "ftmacro.h"

extern int omp_num_threads;

void lud_omp(float * a, int size)
{
     int i,j,k;
     float sum;
#if (R_MODE == 2) || (R_MODE == 3) 
	long count0 = 0;
	long count1 = 0;
#endif
	 //printf("num of threads = %d\n", omp_num_threads);
#if ENABLE_OPENACC == 1
#pragma acc data copy(a[0:_M_SIZE])
#endif
     for (i=0; i <size; i++){
#if R_MODE == 0
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftdata(FTVAR0) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 1
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 2
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftprofile(count0) ftdata(FTVAR0) num_faults(0) num_ftbits(0)
#elif R_MODE == 3
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftprofile(count0) ftkind(FTKIND) num_faults(0) num_ftbits(0)
#elif R_MODE == 4
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftpredict(FTCNT0) ftdata(FTVAR0) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 5
#pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION0) ftpredict(FTCNT0) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#endif
#if ENABLE_OPENACC == 1
#pragma acc kernels loop gang, worker, private(j, k, sum)
#endif
         for (j=i; j <size; j++){
             sum=a[i*size+j];
             for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
             a[i*size+j]=sum;
         }

#if R_MODE == 0
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftdata(FTVAR1) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 1
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 2
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftprofile(count1) ftdata(FTVAR1) num_faults(0) num_ftbits(0)
#elif R_MODE == 3
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftprofile(count1) ftkind(FTKIND) num_faults(0) num_ftbits(0)
#elif R_MODE == 4
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftpredict(FTCNT1) ftdata(FTVAR1) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#elif R_MODE == 5
    #pragma openarc resilience ftregion FTTHREAD ftcond(RES_REGION1) ftpredict(FTCNT1) ftkind(FTKIND) num_faults(TOTAL_NUM_FAULTS) num_ftbits(NUM_FAULTYBITS)
#endif
#if ENABLE_OPENACC == 1
#pragma acc kernels loop if(i+1<size) gang, worker, private(j, k, sum)
#endif
         for (j=i+1;j<size; j++){
             sum=a[j*size+i];
             for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
             a[j*size+i]=sum/a[i*size+i];
         }
     }

#if (R_MODE == 2) || (R_MODE == 3) 
  printf("FT profile-count0 = %ld\n", count0);
  printf("FT profile-count1 = %ld\n", count1);
#endif
}

#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
//#define _OPENACCM
#ifdef _OPENACCM
#include <openacc.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "resilience.h"

#ifndef ENABLE_OPENACC
#define ENABLE_OPENACC 1
#endif

#ifndef _N_
#define _N_ 512
#endif

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

#ifndef R_MODE
#define R_MODE 0
#endif

#ifndef RES_REGION0
#define RES_REGION0 1
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

#ifdef _OPENARC_
#include "ftmacro.h"
#endif

int N = _N_;
int M = _N_;
int P = _N_;

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}


void
MatrixMultiplication_openacc(float * a, float * b, float * c)
{
  int i, j, k ;
#if (R_MODE == 2) || (R_MODE == 3) 
  long count0 = 0;
#endif

#ifdef _OPENACCM
  acc_init(acc_device_default);
#endif

#if ENABLE_OPENACC == 1
#pragma acc data copyout(a[0:(M*N)]), copyin(b[0:(M*P)],c[0:(P*N)])
#endif
  {
#ifdef _OPENARC_
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
#endif
#if ENABLE_OPENACC == 1
#pragma acc kernels loop independent gang
#endif
    for (i=0; i<M; i++){
#if ENABLE_OPENACC == 1
#pragma acc loop worker
#endif
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
#if ENABLE_OPENACC == 1
#pragma acc loop seq
#endif
	  for (k=0; k<P; k++) {
	    sum += b[i*P+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
  }
#if (R_MODE == 2) || (R_MODE == 3) 
  printf("FT profile-count0 = %ld\n", count0); 
#endif
#ifdef _OPENACCM
  acc_shutdown(acc_device_default);
#endif
}


void
MatrixMultiplication_openmp(float * a,float * b, float * c)
{
  int i, j, k ;
  int chunk = N/4;


#pragma omp parallel shared(a,b,c,chunk) private(i,j,k)
  {
#ifdef _OPENMP
	if(omp_get_thread_num() == 0) {
		printf("Number of OpenMP threads %d\n", omp_get_num_threads());
	}
#endif
#pragma omp for
    for (i=0; i<M; i++){
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
	  for (k=0; k<P; k++)
	    sum += b[i*P+k]*c[k*N+j] ;
	  a[i*N+j] = sum ;
        }
    }
  }
}


int main()
{
  float *a, *b, *c;
  float *a_CPU, *b_CPU, *c_CPU;
  int i,j;
  double elapsed_time;

  a = (float *) malloc(M*N*sizeof(float));
  b = (float *) malloc(M*P*sizeof(float));
  c = (float *) malloc(P*N*sizeof(float));
  a_CPU = (float *) malloc(M*N*sizeof(float));
  b_CPU = (float *) malloc(M*P*sizeof(float));
  c_CPU = (float *) malloc(P*N*sizeof(float));

  for (i = 0; i <  M*N; i++) {
    a[i] = (float) 0.0F;
    a_CPU[i] = (float) 0.0F;
  }
  for (i = 0; i <  M*P; i++) {
    b[i] = (float) i;
    b_CPU[i] = (float) i;
  }
  for (i = 0; i <  P*N; i++) {
    c[i] = (float) 1.0F;
    c_CPU[i] = (float) 1.0F;
  }

  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a_CPU,b_CPU,c_CPU);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
  elapsed_time = my_timer();
  MatrixMultiplication_openacc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

#if VERIFICATION == 1
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;

    	for (i=0; i<M*N; i++){
			cpu_sum += a_CPU[i]*a_CPU[i];
			gpu_sum += a[i]*a[i];
		}

		cpu_sum = sqrt(cpu_sum);
		gpu_sum = sqrt(gpu_sum);
		if( cpu_sum > gpu_sum ) {
			rel_err = (cpu_sum-gpu_sum)/cpu_sum;
		} else {
			rel_err = (gpu_sum-cpu_sum)/cpu_sum;
		}

		if(rel_err < 1e-9)
		{
	    	printf("Verification Successful err = %e\n", rel_err);
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#endif

  free(a_CPU);
  free(b_CPU);
  free(c_CPU);
  free(a);
  free(b);
  free(c);

  return 0;
} 


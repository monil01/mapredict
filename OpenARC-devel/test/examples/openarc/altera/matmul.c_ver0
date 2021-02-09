#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#if OMP == 1
#include <omp.h>
#endif

#ifndef _N_
#define _N_ 512
#endif

#ifndef HOST_MEM_ALIGNMENT
#define HOST_MEM_ALIGNMENT 1
#endif

#if HOST_MEM_ALIGNMENT == 1
#define AOCL_ALIGNMENT 64
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

#pragma acc data pcopyout(a[0:(M*N)]), copyin(b[0:(M*P)],c[0:(P*N)])
  {
#pragma acc kernels loop independent gang
    for (i=0; i<M; i++){
#pragma acc loop worker
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
#pragma acc loop seq
	  for (k=0; k<P; k++) {
	    sum += b[i*P+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
  }
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
  float *c, *b, *a_CPU, *a_ACC;
  float refV, testV;
  float diff;
  int error;
#if HOST_MEM_ALIGNMENT == 1
  void *p;
#endif
  int i;
  double elapsed_time;

#if HOST_MEM_ALIGNMENT == 1
  posix_memalign(&p, AOCL_ALIGNMENT, N*N*sizeof(float));
  c = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*N*sizeof(float));
  b = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*N*sizeof(float));
  a_CPU = (float *)p;
  posix_memalign(&p, AOCL_ALIGNMENT, N*N*sizeof(float));
  a_ACC = (float *)p;
#else
  c = (float *) malloc(M*N*sizeof(float));
  b = (float *) malloc(M*P*sizeof(float));
  a_CPU = (float *) malloc(P*N*sizeof(float));
  a_ACC = (float *) malloc(P*N*sizeof(float));
#endif

  for (i = 0; i <  M*N; i++) {
    a_CPU[i] = (float) 0.0;
    a_ACC[i] = (float) 0.0;
  }
  for (i = 0; i <  M*P; i++) {
    b[i] = (float) i;
  }
  for (i = 0; i <  P*N; i++) {
    c[i] = (float) 1.0;
  }

  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a_CPU,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
  elapsed_time = my_timer();
  MatrixMultiplication_openacc(a_ACC,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

  error = 0;
  for (i = 0; i <  M*N; i++) {
    refV = a_CPU[i];
    testV = a_ACC[i];
    if( refV == 0 ) {
      if( refV > testV ) { diff = refV - testV; }
      else { diff = testV - refV; }
    } else {
      if( refV > testV ) { diff = (refV - testV)/refV; }
      else { diff = (testV - refV)/refV; }
    }
    if( diff > 1E-6 ) {
      printf("Results are different: a_CPU[%d] = %lu, a_ACC[%d] = %lu\n", i, refV, i, testV);
      error = 1;
      break;
    }
  }
  if( error == 0 ) { printf("Verfication Success!\n"); }
  else { printf("Verification Fail!\n"); }

  free(a_CPU);
  free(a_ACC);
  free(b);
  free(c);

  return 0;
} 


#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
//#define _OPENACCM
//#ifdef _OPENACCM
#include <openacc.h>
//#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _N_
#define _N_ 512
#endif

#ifndef VERIFICATION
#define VERIFICATION 1
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
MatrixMultiplication_openacc(float ** a, float ** b, float ** c)
{
  int i, j, k ;

#ifdef _OPENACCM
  acc_init(acc_device_default);
#endif
  acc_create(a, M*sizeof(float *));
  acc_create(b, M*sizeof(float *));
  acc_create(c, P*sizeof(float *));
  for (i = 0; i <  M; i++) {
    acc_copyin(a[i], N*sizeof(float));
    //acc_attach() updates a pointer in device memory.
    acc_attach((h_void **)(a+i)); //acc_attach(&(a[i]);
    acc_copyin(b[i], P*sizeof(float));
    acc_attach((h_void **)(b+i));
  }
  for (i = 0; i <  P; i++) {
    acc_create(c[i], N*sizeof(float));
    acc_attach((h_void **)(c+i));
  }

#pragma acc kernels loop independent gang worker collapse(2) present(a[0:(M*N)]), present(b[0:(M*P)],c[0:(P*N)])
    for (i=0; i<M; i++){
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

  for (i = 0; i <  M; i++) {
    acc_update_self(a[i], N*sizeof(float));
  }
  //[INFO] detach/update operations below are unnecessary; 
  //they are included to show how to use acc_detach().
  for (i = 0; i <  M; i++) {
    //acc_detach() updates a pointer in device memory.
    acc_detach((h_void **)(a+i)); //acc_detach(&(a[i]);
    acc_detach((h_void **)(b+i)); //acc_detach(&(b[i]);
  }
  acc_update_self(a, M*sizeof(float *));
  acc_update_self(b, M*sizeof(float *));
  for (i = 0; i <  P; i++) {
    acc_detach((h_void **)(c+i)); //acc_detach(&(c[i]);
  }
  acc_update_self(c, P*sizeof(float *));
#ifdef _OPENACCM
  acc_shutdown(acc_device_default);
#endif
}


void
MatrixMultiplication_openmp(float ** a,float ** b, float ** c)
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
	    sum += b[i][k]*c[k][j] ;
	  a[i][j] = sum ;
        }
    }
  }
}

int main()
{
  // a[M][N], b[M][P], c[P][N];
  float **a, **b, **c;
  float **a_CPU, **b_CPU, **c_CPU;
  int i,j;
  double elapsed_time;

  a = (float **) malloc(M*sizeof(float *));
  b = (float **) malloc(M*sizeof(float *));
  c = (float **) malloc(P*sizeof(float *));
  a_CPU = (float **) malloc(M*sizeof(float *));
  b_CPU = (float **) malloc(M*sizeof(float *));
  c_CPU = (float **) malloc(P*sizeof(float *));

  for (i = 0; i <  M; i++) {
    a[i] = (float *) malloc(N*sizeof(float));
    a_CPU[i] = (float *) malloc(N*sizeof(float));
    b[i] = (float *) malloc(P*sizeof(float));
    b_CPU[i] = (float *) malloc(P*sizeof(float));
  }

  for (i = 0; i <  P; i++) {
    c[i] = (float *) malloc(N*sizeof(float));
    c_CPU[i] = (float *) malloc(N*sizeof(float));
  }

  for (i = 0; i <  M; i++) {
    for (j = 0; j <  N; j++) {
      a[i][j] = (float) 0.0F;
      a_CPU[i][j] = (float) 0.0F;
    }
  }
  for (i = 0; i <  M; i++) {
    for (j = 0; j <  P; j++) {
      b[i][j] = (float)(i*M+j);
      b_CPU[i][j] = (float)(i*M+j);
    }
  }
  for (i = 0; i <  P; i++) {
    for (j = 0; j <  N; j++) {
      c[i][j] = (float) 1.0F;
      c_CPU[i][j] = (float) 1.0F;
    }
  }

#if VERIFICATION == 1
  elapsed_time = my_timer();
  MatrixMultiplication_openmp(a_CPU,b_CPU,c_CPU);
  elapsed_time = my_timer() - elapsed_time;
  printf("CPU Elapsed time = %lf sec\n", elapsed_time);
#endif
  elapsed_time = my_timer();
  MatrixMultiplication_openacc(a,b,c);
  elapsed_time = my_timer() - elapsed_time;
  printf("Accelerator Elapsed time = %lf sec\n", elapsed_time);

#if VERIFICATION == 1
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;

    	for (i=0; i<M; i++){
    	  for (j=0; j<N; j++){
			cpu_sum += a_CPU[i][j]*a_CPU[i][j];
			gpu_sum += a[i][j]*a[i][j];
          }
		}

		cpu_sum = sqrt(cpu_sum);
		gpu_sum = sqrt(gpu_sum);
		if( cpu_sum > gpu_sum ) {
			rel_err = (cpu_sum-gpu_sum)/cpu_sum;
		} else {
			rel_err = (gpu_sum-cpu_sum)/cpu_sum;
		}

		if(rel_err < 1e-6)
		{
	    	printf("Verification Successful err = %e\n", rel_err);
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#endif

  for (i = 0; i <  M; i++) {
    free(a[i]);
    free(a_CPU[i]);
    free(b[i]);
    free(b_CPU[i]);
  }
  for (i = 0; i <  P; i++) {
    free(c[i]);
    free(c_CPU[i]);
  }
  free(a_CPU);
  free(b_CPU);
  free(c_CPU);
  free(a);
  free(b);
  free(c);

  return 0;
} 


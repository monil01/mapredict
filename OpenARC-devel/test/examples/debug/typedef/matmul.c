#include <stdio.h>
#ifndef _N_
#define _N_ 512
#endif

#ifndef VERIFICATION
#define VERIFICATION 1
#endif

typedef float matMN[_N_*_N_];
typedef float matMP[_N_*_N_];
typedef float matPN[_N_*_N_];

int N = _N_;
int M = _N_;
int P = _N_;
matMN a;
matMP b;

void MatrixMultiplication_openacc(matMN a, matMP b, float * c) __OSX_AVAILABLE_STARTING(1000, "message ");
//void MatrixMultiplication_openacc(matMN a, matMP b, float * c) __OSX_AVAILABLE_STARTING(__MAC_10_5, __IPHONE_2_0);
//void MatrixMultiplication_openacc(matMN a, matMP b, float * c) __OSX_AVAILABLE_STARTING(__MAC_10_5);
//void MatrixMultiplication_openacc(matMN a, matMP b, float * c) __asm("_" "matmul" /* nothing */);
//void MatrixMultiplication_openacc(matMN a, matMP b, float * c) __attribute__((__warn_unused_result__));

void
MatrixMultiplication_openacc(matMN a, matMP b, float * c)
{
  int i, j, k ;

#pragma acc kernels loop independent gang worker collapse(2) copyout(a), copyin(b[0:(M*P)],c[0:(P*N)])
    for (i=0; i<M; i++){
      for (j=0; j<N; j++)
        {
	  float sum = 0.0 ;
//#pragma acc loop seq
	  for (k=0; k<P; k++) {
	    sum += b[i*P+k]*c[k*N+j] ;
      }
	  a[i*N+j] = sum ;
        }
    }
}


void
MatrixMultiplication_sequential(float * a,float * b, float * c)
{
  int i, j, k ;
  int chunk = N/4;


  {
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
  float *c;
  float *a_CPU, *b_CPU, *c_CPU;
  int i,j;

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

#if VERIFICATION == 1
  MatrixMultiplication_sequential(a_CPU,b_CPU,c_CPU);
#endif
  MatrixMultiplication_openacc(a,b,c);

  free(a_CPU);
  free(b_CPU);
  free(c_CPU);
  free(a);
  free(b);
  free(c);

  return 0;
} 


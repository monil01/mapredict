#include <stdio.h>
#include <stdlib.h>

#define N 10
#define M 10

#ifndef TESTID
#define TESTID 6
#endif

int main(int argc, char** argv) {
  int size = N*M; 

  float *A = (float*) malloc(size * sizeof(float));
  float *B = (float*) malloc(size * sizeof(float));

  float C = 0;

  for (int i = 0; i < size; i++) {
    A[i] = (float) i;
    B[i] = (float) i * 100;
  }
  
  #if TESTID == 1
  // Bad Kernel 1. 
  // This does generate correct single work-item code,
  // but it also generates unwanted get_global_id and get_num_groups => fixed
  // Also strange warning from aoc
  #pragma acc kernels loop gang(1) worker(1) \
    copyin(A[0:size], B[0:size]) copy(C)
  for (int i = 0; i < size; i++) {
    C += A[i] + B[i];
  }
  #endif

  #if TESTID == 2
  // Bad Kernel 2. 
  // for (int z = ...   No problem
  // for (z = ...       Duplicate device memory allocation => fixed
  int z;
  //#pragma acc kernels copyin(A[0:size], B[0:size])
  #pragma acc kernels copyin(A[0:size], B[0:size]) copy(C)
  {
    #pragma acc loop gang(1) worker(1) reduction(+:C)
    for (z = 0; z < size; z++) {
      C += A[z] + B[z];
    }
  }
  #endif

  #if TESTID == 3
  // Bad Kernel 3. 
  // for (int z = ...   No problem
  // for (z = ...       Duplicate device memory allocation => fixed
  int z;
  //#pragma acc parallel num_gangs(1) num_workers(1) copyin(A[0:size], B[0:size])
  #pragma acc parallel num_gangs(1) num_workers(1) copyin(A[0:size], B[0:size]) copy(C)
  {
    #pragma acc loop gang worker reduction(+:C)
    for (z = 0; z < size; z++) {
      C += A[z] + B[z];
    }
  }
  #endif

  #if TESTID == 4
  // Bad Kernel 4. 
  // for (int n = ...   Reverts to ND-range kernel, incorrect output => fixed, but FPGA-specific collapse transformation is not applied => fixed.
  //   for (int m = ...   
  //
  // int n, m;
  // for (n = ...       Duplicate device memory allocation => fixed
  //   for (m = ...   
  //int n, m;
  #pragma acc kernels copyin(A[0:size], B[0:size]) copy(C)
  {
    #pragma acc loop gang(1) worker(1) collapse(2) reduction(+:C)
    for (int n = 0; n < N; n++) {
      for (int m = 0; m < M; m++) {
        int index = n*M + m;
        C += A[index] + B[index];
      }
    }
  }
  #endif

  #if TESTID == 5
  #define WINDOW
  // Bad Kernel 5. 
  // for (int n = ...   Undeclared symbol n,m => fixed.
  //   for (int m = ...   
  //
  // int n, m;
  // for (n = ...       Undeclared symbol n,m => fixed.
  //   for (m = ...   
  //  
  //int n, m;
  #pragma openarc transform window(A[0:size], A[0:size])
  #pragma acc kernels loop gang(1) worker(1) copyin(A[0:size]) copy(C) collapse(2)  
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {

      if (n*M + m - 1 >= 0 && n*M + m + 1 < size)
        C += A[n*M + m] + A[n*M + m - 1] + A[n*M + m + 1];
    }
  }
  #endif

  #if TESTID == 6
  #define WINDOW
  // Bad Kernel 6. 
  // java.lang.String cannot be cast to cetus.hir.Expression
  #pragma openarc transform window(A[0:size], A[0:size])
  #pragma acc parallel loop num_gangs(1) num_workers(1) \
    copyin(A[0:size]) copy(C)
  for (int z = 0; z < size; ++z) {

      if (z - 1 >= 0 && z + 1 < size)
        C += A[z] + A[z - 1] + A[z + 1];
    }
  }
  #endif

  #if TESTID == 7
  // Bad Kernel 7. 
  // Undeclared symbol C
  // Also, the shift register reduction is applied, but this isn't a 
  //   single work-item kernel
  #pragma acc data copyin(A[:0:size], B[0:size]) copy(C) 
  {
    #pragma acc kernels loop reduction(+:C)
    for (int z = 0; z < size; z++) {
      C += A[z] + B[z];
    }
  }
  #endif

  float C_ver = 0.0F;
  for (int i = 0; i < size; i++) {
    #ifdef WINDOW
    if (i > 0 && i < size - 1) 
      C_ver += 3*i;
    #else
    C_ver += i*101;
    #endif
  }


  printf("size:%d, error:%f\n", size, C_ver - C);

  return 0;
}


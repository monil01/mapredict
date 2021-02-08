#ifdef APP_SPARSELU

#ifndef SPARSELU_H
#define SPARSELU_H

#define EPSILON 1.0E-6


float * allocate_clean_block(int submatrix_size);
__device__ inline void lu0(float *diag, int submatrix_size);
__device__ inline void bdiv(float *diag, float *row, int submatrix_size);
__device__ inline void bmod(float *row, float *col, float *inner, int submatrix_size);
__device__ inline void fwd(float *diag, float *col, int submatrix_size);

__global__ void lu0_global(int kk, int nblocks, float *BENCH, int submatrix_size);
__global__ void fwd_global(int kk, int nblocks, float *BENCH, int submatrix_size);
__global__ void bdiv_global(int kk, int nblocks, float *BENCH, int submatrix_size);
__global__ void bmod_global(int kk, int nblocks, float *BENCH, int submatrix_size);


void sparselu_seq_call(float *BENCH, int matrix_size, int submatrix_size);
double sparselu_par_call(float *BENCH, int matrix_size, int submatrix_size);
double sparselu_global(float *BENCH, int matrix_size, int submatrix_size);

double run(struct user_parameters* params);

#endif

#endif

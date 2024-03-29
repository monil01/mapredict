
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
//#include <malloc.h>
#if OMP == 1
#include <omp.h>
#endif

#ifndef _N_
#define _N_ 8192
#endif

#define MUL(x,y) ((x)*(y)) 

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#ifndef TRANSPOSE_Bs
#define TRANSPOSE_Bs 0
#endif

#ifndef HOST_MEM_ALIGNMENT
#define HOST_MEM_ALIGNMENT 1
#endif

#if HOST_MEM_ALIGNMENT == 1
#define AOCL_ALIGNMENT 64
#endif

#ifndef DEBUG_PRINT
#define DEBUG_PRINT 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#ifdef _OPENARC_
#if BLOCK_SIZE == 4
#pragma openarc #define BLOCK_SIZE 4
#elif BLOCK_SIZE == 8
#pragma openarc #define BLOCK_SIZE 8
#elif BLOCK_SIZE == 16
#pragma openarc #define BLOCK_SIZE 16
#elif BLOCK_SIZE == 32
#pragma openarc #define BLOCK_SIZE 32
#elif BLOCK_SIZE == 64
#pragma openarc #define BLOCK_SIZE 64
#endif
#endif

float mul(float x, float y)
{
		return MUL(x,y);
}

double my_timer ()
{
		struct timeval time;

		gettimeofday (&time, 0);

		return time.tv_sec + time.tv_usec / 1000000.0;
}


int main(int argc, char **argv)
{
		int iter;
		int i, j;
		int num_iterations = 1;
		int bx, by, tx, ty;
		int wA, wB;
		float *A;
		float *B;
		float *GPU_C;
		float *CPU_C;
#if DEBUG_PRINT == 1
		float dSum = 0;
#endif
#if HOST_MEM_ALIGNMENT == 1
		void *p;
#endif

		double strt_time, done_time;
		printf("Matrix Multiplication\n");
		printf("width x height = %d x %d\n", _N_, _N_);
		printf("Iterations     = %d\n", num_iterations);
		wA = _N_;
		wB = _N_;

#if HOST_MEM_ALIGNMENT == 1
		posix_memalign(&p, AOCL_ALIGNMENT, _N_*_N_*sizeof(float));
		A = (float *)p;
		posix_memalign(&p, AOCL_ALIGNMENT, _N_*_N_*sizeof(float));
		B = (float *)p;
		posix_memalign(&p, AOCL_ALIGNMENT, _N_*_N_*sizeof(float));
		GPU_C = (float *)p;
		posix_memalign(&p, AOCL_ALIGNMENT, _N_*_N_*sizeof(float));
		CPU_C = (float *)p;
#else
		A = (float *)malloc(sizeof(float)*_N_*_N_);
		B = (float *)malloc(sizeof(float)*_N_*_N_);
		GPU_C = (float *)malloc(sizeof(float)*_N_*_N_);
		CPU_C = (float *)malloc(sizeof(float)*_N_*_N_);
#endif


		/* initialize matrix A and B */
		{
				for (i = 0; i < _N_; i++) {
						for (j = 0; j < _N_; j++) {
								if (i == j)
										A[i*_N_+j] = 1.0f;
								else
										A[i*_N_+j] = 0.0f;

								B[i*_N_+j] = i+j;

								GPU_C[i*_N_+j] = 0.0f;
						}
				}
		}

		printf("Starting with gpu run\n");
		strt_time = my_timer ();
		for (iter = 0; iter < num_iterations; iter++) {
#pragma openarc opencl num_simd_work_items(2)
#pragma acc kernels loop copyout(GPU_C[0:_N_*_N_]) copyin(A[0:_N_*_N_],B[0:_N_*_N_]) gang(_N_/BLOCK_SIZE)
				for(by = 0; by < (_N_/BLOCK_SIZE); by++) {
#pragma acc loop gang(_N_/BLOCK_SIZE)
						for(bx = 0; bx < (_N_/BLOCK_SIZE); bx++) {
								// Declaration of the shared memory array As used to
								// store the sub-matrix of A
								float As[BLOCK_SIZE][BLOCK_SIZE];

								// Declaration of the shared memory array Bs used to
								// store the sub-matrix of B
								float Bs[BLOCK_SIZE][BLOCK_SIZE];
#pragma acc loop worker(BLOCK_SIZE)
								for(ty = 0; ty < BLOCK_SIZE; ty++) {
#pragma acc loop worker(BLOCK_SIZE)
										for(tx = 0; tx < BLOCK_SIZE; tx++) {
												//Index of the first sub-matrix of A processed by the block
												int aBegin = wA * BLOCK_SIZE * by; 
												// Index of the last sub-matrix of A processed by the block
												int aEnd   = aBegin + wA - 1;

												// Step size used to iterate through the sub-matrices of A
												int aStep  = BLOCK_SIZE;

												// Index of the first sub-matrix of B processed by the block
												int bBegin = BLOCK_SIZE * bx; 

												// Step size used to iterate through the sub-matrices of B
												int bStep  = BLOCK_SIZE * wB; 

												// Csub is used to store the element of the block sub-matrix
												// that is computed by the thread
												float Csub = 0;

												// Loop over all the sub-matrices of A and B
												// required to compute the block sub-matrix
												for (int a = aBegin, b = bBegin;
																a <= aEnd;
																a += aStep, b += bStep) {   


														// Load the matrices from device memory
														// to shared memory; each thread loads
														// one element of each matrix
														As[ty][tx] = A[a + wA * ty + tx];
#if TRANSPOSE_Bs == 0
														Bs[ty][tx] = B[b + wB * ty + tx];
#else
														Bs[tx][ty] = B[b + wB * ty + tx];
#endif

														// Synchronize to make sure the matrices are loaded
//#pragma acc barrier
#pragma acc barrier(acc_mem_fence_local)

														// Multiply the two matrices together;
														// each thread computes one element
														// of the block sub-matrix
#pragma unroll

														for (int k = 0; k < BLOCK_SIZE; ++k) {
#if TRANSPOSE_Bs == 0
																Csub += As[ty][k] * Bs[k][tx];
#else
																Csub += As[ty][k] * Bs[tx][k];
#endif
														}

														// Synchronize to make sure that the preceding
														// computation is done before loading two new
														// sub-matrices of A and B in the next iteration
//#pragma acc barrier
#pragma acc barrier(acc_mem_fence_local)
												}

												// Write the block sub-matrix to device memory;
												// each thread writes one element
												int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
												GPU_C[c + wB * ty + tx] = Csub;
										}
								}
						}
				}
		}

		done_time = my_timer ();
		printf("Done with gpu run\n");
		printf ("Accelerator Elapsed time = %lf sec\n", done_time - strt_time);
#if VERIFICATION == 1
		printf("Starting with cpu run\n");
		strt_time = my_timer ();
		/* verification */
		for (iter = 0; iter < num_iterations; iter++) {
				int i;

#pragma omp parallel for shared(A,B,CPU_C) private(i)
				for (i = 0; i < _N_; i++) {
						int j;

						for (j = 0; j < _N_; j++) {
								int k;
								float sum = 0.0f;

								for (k = 0; k < _N_; k++) {
										sum += mul(A[i*_N_+k],B[k*_N_+j]);
								}

								CPU_C[i*_N_+j] = sum;
						}
				}
		}
		done_time = my_timer ();
		printf("Done with cpu run\n");
		printf ("Reference CPU time = %lf sec\n", done_time - strt_time);



		for (i = 0; i < _N_; i++) {
				for (j = 0; j < _N_; j++) {
						if (CPU_C[i*_N_+j] != GPU_C[i*_N_+j]) {
								printf("Verification: Failed\n");
								printf("CPU_C = %f\tGPU_C = %f\n", CPU_C[i*_N_+j], GPU_C[i*_N_+j]);
								return 1;
						}
				}
		}

		printf("Verification: Successful\n");
#endif
#if DEBUG_PRINT == 1
		for(i=0; i<_N_; i++) {
			dSum += GPU_C[i*_N_+i];
		}			
		printf("Diagonal Sum of GPU_C = %f\n", dSum);
#endif
	
		free(A);
		free(B);
		free(GPU_C);
		free(CPU_C);
		return 0;

}



#ifndef KASTORS_POISSON_H
#define KASTORS_POISSON_H
#include <assert.h>
#include "../../rt.h"
#include "../../helper/utils.h"

#ifdef APP_JACOBI

/******************************************************************************/
/* BEGIN: OPENARC MODIFIED CODE ***********************************************/
/******************************************************************************/

#define DEBUGG
double sweep(int nx, int ny, double dx, double dy, double *f,
	   int itold, int itnew, double *u, double *unew, int block_size);
double sweep_global(int nx, int ny, double dx, double dy, double *f,
	   int itold, int itnew, double *u, double *unew, int block_size);
void sweep_seq(int nx, int ny, double dx, double dy, double *f,
	   int itold, int itnew, double *u, double *unew);

__device__ inline void copy_block(int nx, int ny, int block_x, int block_y,
		double *u, double *unew, int block_size) {
#if DEBUG_TASK_PRINT_FORMAT
	if (threadIdx.x == 0)
		printf("JACOBI: cpblck\t%d\t%d\n", block_x, block_y);
#endif
	int i, j, start_i, start_j;
    start_i = block_x * block_size;
    start_j = block_y * block_size;
    for (i = start_i; i < start_i + block_size; i++) {
    	j = start_j + threadIdx.x;
        u[i*ny+j] = unew[i*ny+j];
    }
}

__device__ inline void compute_estimate(int block_x, int block_y, double *u,
                                    double *unew, double *f, double dx,
                                    double dy, int nx, int ny, int block_size) {
#if DEBUG_TASK_PRINT_FORMAT
	if (threadIdx.x == 0)
		printf("JACOBI: cmptblck\t%d\t%d\n", block_x, block_y);
#endif

    int i, j, start_i, start_j;
    start_i = block_x * block_size;
    start_j = block_y * block_size;
    for (i = start_i; i < start_i + block_size; i++) {
    	j = start_j + threadIdx.x;
		if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
			unew[i*ny+j] = f[i*ny+j];
		} else {
			unew[i*ny+j] = 0.25 * (u[(i-1)*ny+j] + u[i*ny+j+1]
								  + u[i*ny+j-1] + u[(i+1)*ny+j]
								  + f[i*ny+j] * dx * dy);
		}
    }
}

#endif


/******************************************************************************/
/* END: OPENARC MODIFIED CODE *************************************************/
/******************************************************************************/

#endif

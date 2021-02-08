/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include "../../rt.h"
#include "../timer.h"
#include "../main.h"

#ifdef APP_SPARSELU

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>
#include "sparselu.h"



/***********************************************************************
 * lu0:
 **********************************************************************/
__device__ inline void lu0(float *diag, int submatrix_size)
{
    int i, j, k;

//    for (k=0; k<submatrix_size; k++){
    	k = threadIdx.x;

    	for (i=k+1; i<submatrix_size; i++)
        {
            diag[i+submatrix_size*k] = diag[i+submatrix_size*k] / diag[k+submatrix_size*k];
            for (j=k+1; j<submatrix_size; j++)
                diag[i+submatrix_size*j] = diag[i+submatrix_size*j] - diag[i+submatrix_size*k] * diag[k+submatrix_size*j];
        }
//    }
}

/***********************************************************************
 * fwd:
 **********************************************************************/
__device__ inline void fwd(float *diag, float *col, int submatrix_size)
{
    int i, j, k;
    for (j=0; j<submatrix_size; j++)
    	k = threadIdx.x;
    	if (k >0 && k < submatrix_size){
            for (i=k+1; i<submatrix_size; i++)
                col[i+submatrix_size*j] = col[i+submatrix_size*j] - diag[i+submatrix_size*k]*col[k+submatrix_size*j];
    	}
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
__device__ inline void bdiv(float *diag, float *row, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
    	k = threadIdx.x;
    	if (k < submatrix_size){
            row[i+submatrix_size*k] = row[i+submatrix_size*k] / diag[k+submatrix_size*k];
            for (j=k+1; j<submatrix_size; j++)
                row[i+submatrix_size*j] = row[i+submatrix_size*j] - row[i+submatrix_size*k]*diag[k+submatrix_size*j];
        }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
__device__ inline void bmod(float *row, float *col, float *inner, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (j=0; j<submatrix_size; j++)
        	k = threadIdx.x;
        	if (k < submatrix_size){
                inner[i+submatrix_size*j] = inner[i+submatrix_size*j] - row[i+submatrix_size*k]*col[k+submatrix_size*j];
        	}
}

/******************************************************************************/
/* END: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/
__global__ void lu0_global(int kk, int nblocks, float *BENCH, int submatrix_size){
	const int submatrix_sqr = submatrix_size*submatrix_size;
	int diagBlockStart = kk*nblocks*submatrix_sqr+kk*submatrix_sqr;
	lu0(&(BENCH[diagBlockStart]), submatrix_size);

}

/***********************************************************************
 * fwd:
 **********************************************************************/
__global__ void fwd_global(int kk, int nblocks, float *BENCH, int submatrix_size)
{
	const int submatrix_sqr = submatrix_size*submatrix_size;
	int diagBlockStart = kk*nblocks*submatrix_sqr+kk*submatrix_sqr;

	int jj = kk+1+blockIdx.x; // fix offset for (jj=kk+1; jj<nblocks; jj++){

	int colBlockStart = kk*nblocks*submatrix_sqr+jj*submatrix_sqr;
	fwd(&BENCH[diagBlockStart], &BENCH[colBlockStart], submatrix_size);
}

__global__ void bdiv_global(int kk, int nblocks, float *BENCH, int submatrix_size)

{
	const int submatrix_sqr = submatrix_size*submatrix_size;
	int diagBlockStart = kk*nblocks*submatrix_sqr+kk*submatrix_sqr;

	int ii = kk+1+blockIdx.x; // fix offset for (ii=kk+1; ii<nblocks; ii++){
	int rowBlockStart = ii*nblocks*submatrix_sqr+kk*submatrix_sqr;

	bdiv(&BENCH[diagBlockStart], &BENCH[rowBlockStart], submatrix_size);

}
/***********************************************************************
 * bmod:
 **********************************************************************/
__global__ void bmod_global(int kk, int nblocks, float *BENCH, int submatrix_size)
{
	const int submatrix_sqr = submatrix_size*submatrix_size;

	int ii = kk+1+blockIdx.x; // fix offset for (ii=kk+1; ii<nblocks; ii++){
	int rowBlockStart = ii*nblocks*submatrix_sqr+kk*submatrix_sqr;

	int jj = kk+1+blockIdx.y; // fix offset for (jj=kk+1; jj<nblocks; jj++){
	int colBlockStart = kk*nblocks*submatrix_sqr+jj*submatrix_sqr;

	int subBlockStart = ii*nblocks*submatrix_sqr+jj*submatrix_sqr;

	bmod(&BENCH[rowBlockStart],&BENCH[colBlockStart],&BENCH[subBlockStart],submatrix_size);
}

__device__  void app_kernel(TASK* task, APP_CONTEXT* appContext,
	RT_CONTEXT* dynContext){

	if (task->kernel_type == KERNEL_TYPE_LU0){
		lu0(&(appContext->appData.BENCH[task->taskData.diagBlockStart]),
				appContext->appData.submatrix_size);
	}
//	else if (task->kernel_type == KERNEL_TYPE_FWD){
//		fwd(&(appContext->appData.BENCH[task->taskData.diagBlockStart]),
//				&(appContext->appData.BENCH[task->taskData.colBlockStart]),
//				appContext->appData.submatrix_size);
//	}
//	else if (task->kernel_type == KERNEL_TYPE_BDIV){
//		bdiv(&(appContext->appData.BENCH[task->taskData.diagBlockStart]),
//				&(appContext->appData.BENCH[task->taskData.rowBlockStart]),
//				appContext->appData.submatrix_size);
//	}
//	else if (task->kernel_type == KERNEL_TYPE_BMOD){
//		bmod(&(appContext->appData.BENCH[task->taskData.rowBlockStart]),
//				&(appContext->appData.BENCH[task->taskData.colBlockStart]),
//				&(appContext->appData.BENCH[task->taskData.subBlockStart]),
//				appContext->appData.submatrix_size);
//	}
}

#endif

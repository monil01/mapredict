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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include "sparselu-kernel.cuh"
#include "../timer.h"

/******************************************************************************/
/* BEGIN: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/
#include "../../rt.h"

#ifdef APP_SPARSELU

#define KERNEL_TYPE_LU0 0
#define KERNEL_TYPE_BDIV 1
#define KERNEL_TYPE_FWD 2
#define KERNEL_TYPE_BMOD 3


/******************************************************************************/
/* END: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/


double sparselu_global(float *BENCH, int matrix_size, int submatrix_size)
{
	int nblocks;
	nblocks = matrix_size/submatrix_size;

	float* BENCH_d;
    cudaMalloc(&BENCH_d,matrix_size*matrix_size*sizeof(float));

    cudaMemcpy(BENCH_d, BENCH,
    		matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice);

    START_TIMER

	for (int kk=0; kk<nblocks; kk++)
	{
		lu0_global<<<1,submatrix_size>>>(kk, nblocks, BENCH, submatrix_size);
	    cudaDeviceSynchronize();

		int sub = nblocks-kk-1;
		fwd_global<<<sub, submatrix_size>>>(kk,nblocks,BENCH,submatrix_size);
	    cudaDeviceSynchronize();

		bdiv_global<<<sub,submatrix_size>>>(kk,nblocks,BENCH,submatrix_size);
	    cudaDeviceSynchronize();

		bmod_global<<<sub*sub,submatrix_size>>>(kk,nblocks,BENCH,submatrix_size);
	    cudaDeviceSynchronize();
	}
    END_TIMER

    cudaMemcpy(BENCH_d, BENCH,
    		matrix_size*matrix_size*sizeof(float), cudaMemcpyDeviceToHost);

    return TIMER;
}

/******************************************************************************/
/* END: OPENARC GENERATED CODE ************************************************/
/******************************************************************************/
#endif

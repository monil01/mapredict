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
#include "sparselu.h"
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


double sparselu_par_call(float *BENCH, int matrix_size, int submatrix_size)
{
    int ii, jj, kk;
	int nblocks;
	int submatrix_sqr;
	nblocks = matrix_size/submatrix_size;
	submatrix_sqr = submatrix_size*submatrix_size;

/******************************************************************************/
/* BEGIN: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/
	APP_CONTEXT appContext_h;
	APP_CONTEXT appContext_d;
	RT_CONTEXT rtContext_d;

	// TODO: [OPEANARC] Compiler should inspect the tasks in the nested loops
	// to come up with nTasks and nEdges values.
	int nTasks = nblocks * //outermost
			(
					1 + // LU factorization of diagonal
					(nblocks+1)/2 + // trailing submatrix (increasing sum series)
					(nblocks+1)/2 + // panel (increasing sum series)
					(nblocks/3)*(nblocks+1)*((float)nblocks+0.5) // submatrix (increasing sum of squares series)
			);

	int nEdges = nblocks * //outermost
			(
					1*1 + // LU factorization of diagonal
					2*((nblocks+1)/2) + // trailing submatrix
					2*((nblocks+1)/2) + // panel
					3*(nblocks/3)*(nblocks+1)*((float)nblocks+0.5) // submatrix (sum of squares)
			);

	// Host related application initializations
	initAppContext_H(appContext_h, nTasks, nEdges);
	// TODO: this shouldn't be here
	appContext_h.appData.BENCH = (float*) malloc(matrix_size*matrix_size*sizeof(float));
	appContext_h.appData.matrix_size = matrix_size;
	appContext_h.appData.matrix_size = submatrix_size;


#pragma omp parallel private(kk,ii,jj) shared(BENCH)
#pragma omp single /* nowait */
    {
        /*#pragma omp task untied*/
        for (kk=0; kk<nblocks; kk++)
        {
			// Apply LU factorization to the panel ??
			int diagBlockStart = kk*nblocks*submatrix_sqr+kk*submatrix_sqr;

//#pragma omp task firstprivate(kk) shared(BENCH) depend(inout: BENCH[diagBlockStart:submatrix_sqr)
			// lu0(&(BENCH[diagBlockStart]), submatrix_size);
			int taskIndex=addTask(appContext_h, KERNEL_TYPE_LU0);
			appContext_h.tasks[taskIndex].taskData.diagBlockStart = diagBlockStart;

			// INOUT: For in-out dependencies, we do the in first, then out.
			processInDependency(appContext_h, taskIndex,
					3,(intptr_t)appContext_h.appData.BENCH,diagBlockStart,submatrix_sqr);
			processOutDependency(appContext_h, taskIndex,
					3,(intptr_t)appContext_h.appData.BENCH,diagBlockStart,submatrix_sqr);


			// update trailing submatrix (towards right)
            for (jj=kk+1; jj<nblocks; jj++){
				int colBlockStart = kk*nblocks*submatrix_sqr+jj*submatrix_sqr;
//#pragma omp task firstprivate(kk, jj) shared(BENCH) depend(in: BENCH[diagBlockStart:submatrix_size*submatrix_size])
				//depend(inout: BENCH[colBlockStart:submatrix_size*submatrix_size])
				// fwd(BENCH[diagBlockStart], BENCH[colBlockStart], submatrix_size);
				int taskIndex=addTask(appContext_h, KERNEL_TYPE_FWD);
				appContext_h.tasks[taskIndex].taskData.colBlockStart = colBlockStart;
				appContext_h.tasks[taskIndex].taskData.diagBlockStart = diagBlockStart;

				// INOUT: For in-out dependencies, we do the in first, then out.
				processInDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,colBlockStart,submatrix_sqr);
				processOutDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,colBlockStart,submatrix_sqr);

				// IN:
				processInDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,diagBlockStart,submatrix_sqr);
            }
			// update panel
            for (ii=kk+1; ii<nblocks; ii++){
				int rowBlockStart = ii*nblocks*submatrix_sqr+kk*submatrix_sqr;

#pragma omp task firstprivate(kk, ii) shared(BENCH) depend(in: BENCH[diagBlockStart:submatrix_size*submatrix_size]) depend(inout: BENCH[rowBlockStart:submatrix_size*submatrix_size])
				//bdiv (BENCH[diagBlockStart], BENCH[rowBlockStart], submatrix_size);
				int taskIndex=addTask(appContext_h, KERNEL_TYPE_FWD);
				appContext_h.tasks[taskIndex].taskData.diagBlockStart = diagBlockStart;
				appContext_h.tasks[taskIndex].taskData.rowBlockStart = rowBlockStart;

				// INOUT: For in-out dependencies, we do the in first, then out.
				processInDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,rowBlockStart,submatrix_sqr);
				processOutDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,rowBlockStart,submatrix_sqr);

				// IN:
				processInDependency(appContext_h, taskIndex,
						3,(intptr_t)appContext_h.appData.BENCH,diagBlockStart,submatrix_sqr);
            }
			// update submatrix
            for (ii=kk+1; ii<nblocks; ii++){
				int rowBlockStart = ii*nblocks*submatrix_sqr+kk*submatrix_sqr;
				for (jj=kk+1; jj<nblocks; jj++){
					int colBlockStart = kk*nblocks*submatrix_sqr+jj*submatrix_sqr;
					int subBlockStart = ii*nblocks*submatrix_sqr+jj*submatrix_sqr;
#pragma omp task firstprivate(kk, jj, ii) shared(BENCH) \
					depend(in: BENCH[rowBlockStart:submatrix_size*submatrix_size], \
							BENCH[colBlockStart:submatrix_size*submatrix_size]) \
					depend(inout: BENCH[subBlockStart:submatrix_size*submatrix_size])
					//bmod(BENCH[rowBlockStart], BENCH[colBlockStart], BENCH[subBlockStart], submatrix_size);
					int taskIndex=addTask(appContext_h, KERNEL_TYPE_FWD);
					appContext_h.tasks[taskIndex].taskData.rowBlockStart = rowBlockStart;
					appContext_h.tasks[taskIndex].taskData.colBlockStart = colBlockStart;
					appContext_h.tasks[taskIndex].taskData.subBlockStart = subBlockStart;

					// INOUT: For in-out dependencies, we do the in first, then out.
					processInDependency(appContext_h, taskIndex,
							3,(intptr_t)appContext_h.appData.BENCH,subBlockStart,submatrix_sqr);
					processOutDependency(appContext_h, taskIndex,
							3,(intptr_t)appContext_h.appData.BENCH,subBlockStart,submatrix_sqr);

					// IN:
					processInDependency(appContext_h, taskIndex,
							3,(intptr_t)appContext_h.appData.BENCH,rowBlockStart,submatrix_sqr);
					processInDependency(appContext_h, taskIndex,
							3,(intptr_t)appContext_h.appData.BENCH,colBlockStart,submatrix_sqr);

				}
			}
        }
    }

    // Convert vector based dependencyMatrix to flat array (dependencies_csr).
    buildCSR(appContext_h);

    // Parameters
    rtContext_d.nWorkers = N_WORKERS; // TODO: how and where to determine this? nworkers=number of active thread blocks
    appContext_h.blockSize = submatrix_size; // TODO: how and where to determine this? blockSize=number of threads within a thread block.
    appContext_h.subBlockSize = submatrix_size; // no sub blocks for now, same with above

    // Device related runtime initializations
    initAppContext_D(appContext_h,appContext_d);
    initRtContext_D(appContext_h,rtContext_d);

    // Now cudamalloc and memcpy appContext_h.appData contents to appContext_d.appData
    // TODO: how will the compiler find the size of the data sent? Assuming, it
    // would be based on iteration ranges.

    checkCudaErrors(cudaMalloc(&(appContext_d.appData.BENCH),matrix_size*matrix_size*sizeof(float)));
    checkCudaErrors(cudaMemcpy(appContext_d.appData.BENCH, appContext_h.appData.BENCH,
    		matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice));

    appContext_d.appData.matrix_size = matrix_size;
    appContext_d.appData.submatrix_size= submatrix_size;

    START_TIMER
    runtime<<<rtContext_d.nWorkers, appContext_h.blockSize>>>(appContext_d, rtContext_d);
    getLastCudaError("Kernel execution failed: ");
    checkCudaErrors(cudaDeviceSynchronize());
    END_TIMER

    analyzeRun(rtContext_d,appContext_d,appContext_h);

    cudaMemcpy(appContext_h.appData.BENCH, appContext_d.appData.BENCH,
    		matrix_size*matrix_size*sizeof(float), cudaMemcpyDeviceToHost);


    return TIMER;
}

/******************************************************************************/
/* END: OPENARC GENERATED CODE ************************************************/
/******************************************************************************/
#endif

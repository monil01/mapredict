#include "poisson.cuh"
#include "../../rt.h"
#include "../timer.h"

#ifdef APP_JACOBI

/* #pragma omp task/taskwait version of SWEEP. */
double sweep (int nx, int ny, double dx, double dy, double *f_,
            int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;
    int block_x, block_y;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);

/******************************************************************************/
/* BEGIN: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/
	APP_CONTEXT appContext_h;
	APP_CONTEXT appContext_d;
	RT_CONTEXT rtContext_d;

	// TODO: [OPEANARC] Compiler should inspect the tasks in the nested loops
	// to come up with nTasks and nEdges values.
	int nTasks = (itnew-itold) * // outermost loop (time iteration)
			max_blocks_x * max_blocks_y * // inner loops
			2; // there are two kernels

	int nEdges = (5 + 1) * // 1st kernel has 1 and 2nd 5 depend "in" clauses
			nTasks; // each task has those dependencies.

	// Host related application initializations
	initAppContext_H(appContext_h, nTasks, nEdges);
	appContext_h.appData.blockSize = block_size;
	appContext_h.appData.nx = nx;
	appContext_h.appData.ny = ny;
	appContext_h.appData.dx = dx;
	appContext_h.appData.dy = dy;
	appContext_h.appData.f = f_; // unnecessary assignment, but compiler needs
						//to pass the arguments to app data if they are in the
						// kernel scope
	appContext_h.appData.u = u_;
	appContext_h.appData.unew = unew_;

	int noDependencies = 0;

	// Populate dependency through loop inspection
	for (it = itold + 1; it <= itnew; it++) {
		for (block_x = 0; block_x < max_blocks_x; block_x++) {
			for (block_y = 0; block_y < max_blocks_y; block_y++) {
				// TODO: Task to TB mapping is TBD
				//dim3 blockIdx(block_x,block_y,0);
				int taskIndex=addTask(appContext_h, KERNEL_TYPE_COPY_BLOCK);
				appContext_h.tasks[taskIndex].taskData.block_x = block_x;
				appContext_h.tasks[taskIndex].taskData.block_y = block_y;

				if (noDependencies) // for debugging purposes
					continue;
				// First map "out" dependencies in taskmap to the current
				// task( and possibly others) to easily back refer
				// via an "in" dependency later.

				// We create a dependency identifier to store the info in the
				// OMP clause
				// Original OMP clause:
				// depend(out: u[block_x * block_size: block_size]
                //				[block_y * block_size: block_size])

				processOutDependency(appContext_h, taskIndex,
						5,
						(intptr_t)appContext_h.appData.u,
						block_x * block_size,
						block_size,
						block_y * block_size,
						block_size);

				// Then process "in" dependencies to find the parent(s) of the
				// current task.

				// Original OMP clause for the in dependency range:
				// depend(in: unew[block_x * block_size: block_size]
				//					[block_y * block_size: block_size])

				processInDependency(appContext_h, taskIndex,
						5,
						(intptr_t)appContext_h.appData.unew,
						block_x * block_size,
						block_size,
						block_y * block_size,
						block_size);
			}
		}

		for (block_x = 0; block_x < max_blocks_x; block_x++) {
			for (block_y = 0; block_y < max_blocks_y; block_y++) {
				int taskIndex=addTask(appContext_h, KERNEL_TYPE_COMPUTE_BLOCK);
				appContext_h.tasks[taskIndex].taskData.block_x = block_x;
				appContext_h.tasks[taskIndex].taskData.block_y = block_y;

				if (noDependencies) // for debugging purposes
					continue;

				// Again, first, create dep. identifiers for out dependencies.
                // OMP CLAUSE: (out: unew[block_x * block_size: block_size]
				//							[block_y * block_size: block_size])
				processOutDependency(appContext_h, taskIndex,
						5,
						(intptr_t)appContext_h.appData.unew,
						block_x * block_size,
						block_size,
						block_y * block_size,
						block_size);

				// Then, using in dependencies, find parent tasks. In the case
				// of multiple "in" clauses, we need to repeat the following
				// again for each.

				// 1st "in" dependency: (in: u[block_x * block_size: block_size]
				// 							[block_y * block_size: block_size])
				processInDependency(appContext_h, taskIndex,
						5,
						(intptr_t)appContext_h.appData.u,
						block_x * block_size,
						block_size,
						block_y * block_size,
						block_size);

				// 2nd "in" dependency: (in: u[(block_x - 1) * block_size: block_size]
				//								[block_y * block_size: block_size])
				if (block_x > 0){// TODO: these if checks may not be identified by the compiler, CHECK
					processInDependency(appContext_h, taskIndex,
							5,
							(intptr_t)appContext_h.appData.u,
							(block_x-1) * block_size,
							block_size,
							block_y * block_size,
							block_size);
				}

				// 3rd "in" dependency: (in: u[(block_x + 1) * block_size: block_size]
				//								[block_y * block_size: block_size])
				if (block_x < max_blocks_x-1){
					processInDependency(appContext_h, taskIndex,
							5,
							(intptr_t)appContext_h.appData.u,
							(block_x+1) * block_size,
							block_size,
							block_y * block_size,
							block_size);

				}

				// 4th "in" dependency: (in: u[block_x  * block_size: block_size]
				//								[(block_y-1) * block_size: block_size])
				if (block_y > 0){
					processInDependency(appContext_h, taskIndex,
							5,
							(intptr_t)appContext_h.appData.u,
							block_x * block_size,
							block_size,
							(block_y-1) * block_size,
							block_size);
				}

				// 5th "in" dependency: (in: u[block_x  * block_size: block_size]
				//								[(block_y+1) * block_size: block_size])
				if (block_y < max_blocks_y-1){
					processInDependency(appContext_h, taskIndex,
							5,
							(intptr_t)appContext_h.appData.u,
							block_x * block_size,
							block_size,
							(block_y+1) * block_size,
							block_size);
				}
			}
		}
	}

	// Convert vector based dependencyMatrix to flat array (dependencies_csr).
	buildCSR(appContext_h);

	// Parameters
	rtContext_d.nWorkers = N_WORKERS;// TODO: how and where to determine this? nworkers=number of active thread blocks
	appContext_h.blockSize = block_size; // TODO: how and where to determine this? blockSize=number of threads within a thread block.
	appContext_h.subBlockSize = block_size; // no sub blocks for now..

	// Device related runtime initializations
	initAppContext_D(appContext_h,appContext_d);
	initRtContext_D(appContext_h,rtContext_d);

	// Now cudamalloc and memcpy appContext_h.appData contents to appContext_d.appData
	// TODO: how will the compiler find the size of the data sent? Assuming, it
	// would be based on iteration ranges.

	size_t dataSize = nx*ny*sizeof(double);
	cudaMalloc(&(appContext_d.appData.f),dataSize);
	cudaMalloc(&(appContext_d.appData.u),dataSize);
	cudaMalloc(&(appContext_d.appData.unew),dataSize);

	checkCudaErrors(cudaMemcpy(appContext_d.appData.f, appContext_h.appData.f, dataSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(appContext_d.appData.u, appContext_h.appData.u, dataSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(appContext_d.appData.unew, appContext_h.appData.unew, dataSize, cudaMemcpyHostToDevice));

	START_TIMER
	runtime<<<rtContext_d.nWorkers, appContext_h.blockSize>>>(appContext_d, rtContext_d);
  getLastCudaError("Error during kernel launch: ");

  checkCudaErrors(cudaDeviceSynchronize());
	END_TIMER

	analyzeRun(rtContext_d, appContext_d, appContext_h);

	checkCudaErrors(cudaMemcpy(appContext_h.appData.f, appContext_d.appData.f, dataSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(appContext_h.appData.u, appContext_d.appData.u, dataSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(appContext_h.appData.unew, appContext_d.appData.unew, dataSize, cudaMemcpyDeviceToHost));

	return TIMER;
}

__device__  void app_kernel(TASK* task, APP_CONTEXT* appContext,
	RT_CONTEXT* dynContext){
	if (task->kernel_type == KERNEL_TYPE_COPY_BLOCK){
		copy_block(appContext->appData.nx, appContext->appData.ny,
				task->taskData.block_x, task->taskData.block_y, appContext->appData.u,
				appContext->appData.unew, appContext->appData.blockSize);

	}
	else if(task->kernel_type == KERNEL_TYPE_COMPUTE_BLOCK){
		compute_estimate(task->taskData.block_x, task->taskData.block_y, appContext->appData.u,
				appContext->appData.unew, appContext->appData.f,
				appContext->appData.dx, appContext->appData.dy,
				appContext->appData.nx, appContext->appData.ny, appContext->appData.blockSize);
	}
}
#endif



/******************************************************************************/
/* END: OPENARC GENERATED CODE ************************************************/
/******************************************************************************/

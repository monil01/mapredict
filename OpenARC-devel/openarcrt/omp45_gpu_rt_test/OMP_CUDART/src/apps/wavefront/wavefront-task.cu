#include "../main.h"
#include "../timer.h"
#include "../../rt.h"
#include "wavefront-kernel.cuh"

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)

double wavefront_task(APP_DATA appData, int n_tile_waves, int minDim, int maxDim, int tile_size)
{
	/******************************************************************************/
	/* BEGIN: OPENARC GENERATED CODE ******************************************/
	/******************************************************************************/
	APP_CONTEXT appContext_h;
	APP_CONTEXT appContext_d;
	RT_CONTEXT rtContext_d;

	// TODO: [OPEANARC] Compiler should pre-inspect the tasks in the nested loops
	// to come up with nTasks and nEdges values.

	int nTasks = appData.n_tile_rows*appData.n_tile_columns;
	int nEdges = 3*nTasks;

	// Host related application initializations
	initAppContext_H(appContext_h, nTasks, nEdges);
	appContext_h.appData = appData;


	// Populate dependency through loop inspection
	for (int waveId=0; waveId<n_tile_waves; waveId++)
	{
		int waveLength = waveId+1;
		if (waveId+1 > minDim){
			waveLength = min(minDim,n_tile_waves-waveId);
		}

		for (int bid = 0; bid< waveLength; bid++){
			// determine row and column of the tile
			int tileRow=waveId - bid;
			int tileColumn=bid;
			if (waveId > appContext_h.appData.n_tile_rows-1){
				tileRow=appContext_h.appData.n_tile_rows-1-bid;
				tileColumn=waveId-appContext_h.appData.n_tile_rows+1+bid;
			}
			int taskIndex=addTask(appContext_h, KERNEL_TYPE_WF);
			appContext_h.tasks[taskIndex].taskData.tileColumn = tileColumn;
			appContext_h.tasks[taskIndex].taskData.tileRow = tileRow;

			int tileBaseIndex = tile_size*appData.n_tile_columns*tileRow + tile_size*tileColumn;
					//(intptr_t)appContext_h.appData.data;

			processOutDependency(appContext_h, taskIndex,
				3,tileBaseIndex,tileRow,tileColumn);


			if (tileRow > 0){ // UP
				int neighborIndex = tileBaseIndex - tile_size*appData.n_tile_columns;
				processInDependency(appContext_h, taskIndex,
					3,neighborIndex,tileRow-1,tileColumn);
			}
			if (tileColumn > 0){ // LEFT
				int neighborIndex = tileBaseIndex - tile_size;
				processInDependency(appContext_h, taskIndex,
					3,neighborIndex,tileRow,tileColumn-1);
			}
		}
	}

    // Convert vector based dependencyMatrix to flat array (dependencies_csr).
    buildCSR(appContext_h);

    // Parameters
    rtContext_d.nWorkers = N_WORKERS;// TODO: how and where to determine this? nworkers=number of active thread blocks
    appContext_h.blockSize = appData.tile_width; // TODO: how and where to determine this? blockSize=number of threads within a thread block.
    appContext_h.subBlockSize = appData.tile_width; // no sub blocks for now, same with above

    // Device related runtime initializations
    initAppContext_D(appContext_h,appContext_d);
    initRtContext_D(appContext_h,rtContext_d);

    // Now cudamalloc and memcpy appContext_h.appData contents to appContext_d.appData
    // TODO: how will the compiler find the size of the data sent? Assuming, it
    // would be based on iteration ranges.

    long mallocSize = sizeof(float);
    mallocSize *= appData.data_width;
    mallocSize *= appData.data_height;
    checkCudaErrors(cudaMalloc(&(appContext_d.appData.data),mallocSize));

    checkCudaErrors(cudaMemcpy(appContext_d.appData.data, appContext_h.appData.data,
    		appData.data_width*appData.data_height*sizeof(float), cudaMemcpyHostToDevice));

#if defined(APP_SW) || defined(APP_DTW)
    cudaMalloc(&(appContext_d.appData.seq1),appContext_d.appData.data_width*sizeof(float));
    cudaMalloc(&(appContext_d.appData.seq2),appContext_d.appData.data_height*sizeof(float));
    cudaMemcpy(appContext_d.appData.seq1, appData.seq1,
    		appContext_d.appData.data_width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(appContext_d.appData.seq2, appData.seq2,
    		appContext_d.appData.data_height*sizeof(float), cudaMemcpyHostToDevice);
#endif

#if defined(APP_INT)
    size_t binSize = 255*sizeof(int);
    cudaMalloc(&(appContext_d.appData.bin),binSize);
    cudaMemcpy(appContext_d.appData.bin, appData.bin,binSize, cudaMemcpyHostToDevice);
#endif


    START_TIMER
    runtime<<<rtContext_d.nWorkers, appContext_h.blockSize>>>(appContext_d, rtContext_d);
    getLastCudaError("Error during kernel launch: ");
    checkCudaErrors(cudaDeviceSynchronize());
    END_TIMER

	analyzeRun(rtContext_d, appContext_d, appContext_h);


    checkCudaErrors(cudaMemcpy(appContext_h.appData.data, appContext_d.appData.data,
    		appData.data_width*appData.data_height*sizeof(float), cudaMemcpyDeviceToHost));

    return TIMER;
}

__device__   void  app_kernel(TASK* task, APP_CONTEXT* appContext,
	RT_CONTEXT* dynContext){
	//computeTile(task->taskData.tileRow, task->taskData.tileColumn, &appContext->appData);
	computeTile(task->taskData.tileRow, task->taskData.tileColumn,
			appContext->appData.tile_width, appContext->appData.tile_height,
			appContext->appData.n_tile_rows, appContext->appData.n_tile_columns,
			appContext->appData.data
#if defined(APP_SW) || defined(APP_DTW)
			//,appContext->appData.seq1
			//,appContext->appData.seq2
#elif defined(APP_INT)
			,appContext->appData.bin
#endif
	);
}

#endif

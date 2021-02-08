// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include <getopt.h>

#include "helper/utils.h"
#include "helper/helper_cuda.h"
#include "app.h"

// declaration, forward
TIME_MSEC run();

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
	TIME_MSEC start = get_current_msec();

	APP_CONTEXT appContext_h;
	APP_CONTEXT appContext_d;
	RT_CONTEXT rtContext_d;

	// Host related application initializations
	// TODO: determine nTasks and nEdges here.
	initAppContext_H(appContext_h, 0, 0);
	vector<vector<int>> dependencyMatrix;

	// Populate dependency matrix here
	for (int i = 0; i < 3; ++i) {
		addTask(appContext_h,0,0,0,0);
		//addDependency(appContext_h,i,j);
	}

	buildCSR(appContext_h);

	// Device related runtime initializations
	initAppContext_D(appContext_h,appContext_d);
	initRtContext_D(appContext_h,rtContext_d);

	runtime<<<appContext_h.blockSize, rtContext_d.nWorkers>>>(appContext_d, rtContext_d);

	// Wait scheduler and worker threads to finish
	checkCudaErrors(cudaDeviceSynchronize());

	TIME_MSEC end = get_current_msec();

#if DEBUG_TASK_PROCESSING_ORDER
	int* task_order_host = (int*) malloc(appContext.totalTaskCount*2*sizeof(int));
	checkCudaErrors(cudaMemcpy(task_order_host, dynContext.task_order,
			appContext.totalTaskCount*2*sizeof(int), cudaMemcpyDeviceToHost));

#endif

#if DEBUG_QUEUE_LOAD_TRACKING
	QUEUE_LOAD_RECORD* queueLoadRecordArray =
			(QUEUE_LOAD_RECORD*) malloc(sizeof(_QUEUE_LOAD_RECORD)*500);
	checkCudaErrors(cudaMemcpy(queueLoadRecordArray, dynContext.queueLoadRecordArray,
			sizeof(_QUEUE_LOAD_RECORD)*500, cudaMemcpyDeviceToHost));

	int queueLoadRecordArraySize;
	checkCudaErrors(cudaMemcpy(&queueLoadRecordArraySize,
			dynContext.queueLoadRecordArraySize,
			sizeof(int),
			cudaMemcpyDeviceToHost));
#endif

#if DEBUG_ON

	int* taskCounts_host = (int*) malloc(N_BLOCKS*sizeof(int));
	checkCudaErrors(cudaMemcpy(taskCounts_host,
			dynContext->processed_task_counts, N_BLOCKS*sizeof(int), cudaMemcpyDeviceToHost));

	clock_t* smComputeTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smComputeTime_host,
			dynContext->smComputeTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smWaitTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smWaitTime_host,
			dynContext->smWaitTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smQueueInsertTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smQueueInsertTime_host,
			dynContext->smQueueInsertTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smQueueRetrievalTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smQueueRetrievalTime_host,
			dynContext->smQueueRetrievalTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smDependencyResTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smDependencyResTime_host,
			dynContext->smDependencyResolutionTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smIQSignalingTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smIQSignalingTime_host,
			dynContext->smIQSignalingTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

	clock_t* smOQSignalingTime_host = (clock_t*) malloc(N_BLOCKS*sizeof(clock_t));
	checkCudaErrors(cudaMemcpy(smOQSignalingTime_host,
			dynContext->smOQSignalingTime, N_BLOCKS*sizeof(clock_t), cudaMemcpyDeviceToHost));

#endif

#if DEBUG_ON
	printf("Worker timings:\n");

	float totalComputeTime = 0;
	float totalWaitTime = 0;
	float totalQueueInsertTime = 0;
	float totalQueueRetrievalTime = 0;
	float totalDependencyResTime = 0;
	float totalIQSignalingTime = 0;
	float totalOQSignalingTime = 0;
	int totalTasksProcessed = 0;

	int device;
	cudaGetDevice(&device);

	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	float frequency = props.clockRate;

	printf("SM_ID:\tTC\tET   \tWT   \tQIT   \tQRT   \tDRT   \tIQS   \tOQS\n");

	for (int i=0; i<N_BLOCKS; i++)
	{
		printf("SM_%d:\t%d\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\n",
				i, taskCounts_host[i],
				smComputeTime_host[i]/frequency,
				smWaitTime_host[i]/frequency,
				smQueueInsertTime_host[i]/frequency,
				smQueueRetrievalTime_host[i]/frequency,
				smDependencyResTime_host[i]/frequency,
				smIQSignalingTime_host[i]/frequency,
				smOQSignalingTime_host[i]/frequency

		);

		totalComputeTime += (smComputeTime_host[i]/frequency);
		totalWaitTime += (smWaitTime_host[i]/frequency);
		totalQueueInsertTime += (smQueueInsertTime_host[i]/frequency);
		totalQueueRetrievalTime += (smQueueRetrievalTime_host[i]/frequency);
		totalDependencyResTime += (smDependencyResTime_host[i]/frequency);
		totalIQSignalingTime += (smIQSignalingTime_host[i]/frequency);
		totalOQSignalingTime += (smOQSignalingTime_host[i]/frequency);
		totalTasksProcessed += taskCounts_host[i];
	}

	printf("Avg:\t%d\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\n",
			totalTasksProcessed,
			totalComputeTime/N_BLOCKS,
			totalWaitTime/N_BLOCKS,
			totalQueueInsertTime/N_BLOCKS,
			totalQueueRetrievalTime/N_BLOCKS,
			totalDependencyResTime/N_BLOCKS,
			totalIQSignalingTime/N_BLOCKS,
			totalOQSignalingTime/N_BLOCKS
	);
#endif

#if DEBUG_TASK_PROCESSING_ORDER
	printf("\nTasks:\n");

	for (int i=0; i<appContext.totalTaskCount*2; i++)
	{
		int value = task_order_host[i];
		int sm_id = value/DEBUG_SM_ID_COEFFICIENT;
		int task_id = value-(sm_id*DEBUG_SM_ID_COEFFICIENT);
		if (DEBUG_TASK_PRINT_FORMAT == 0)
		{
			printf("(#%d:%d)", sm_id, task_id);

			if (i % 20 == 0 && i>0)
				printf("\n");
		}
		else if (DEBUG_TASK_PRINT_FORMAT == 1)
		{
			for (int k=0; k < sm_id;k++)
				printf("\t");

			printf("(#%d:%d)", sm_id, task_id);
			printf("\n");
		}
	}
	printf("\n");


#endif

#if DEBUG_QUEUE_LOAD_TRACKING
	printf("\nQueueLoad:\n");
	for (int j = 0; j < queueLoadRecordArraySize; ++j) {

		QUEUE_LOAD_RECORD record = queueLoadRecordArray[j];

		printf ("%2.3f\t", record.time/frequency);

		for(int i = 0; i<N_BLOCKS; i++)
		{
			printf("%d\t", record.load[i]);
		}
		printf("\n");
	}
#endif

	checkCudaErrors(cudaDeviceReset());

	return end-start;
}

//__global__ void launchDYN(APPCONTEXT* appContext, DYNCONTEXT* dynContext) {
//
//	cudaStream_t stream1;
//	cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
//	//scheduler<<<1, N_BLOCKS*SCHED_QUEUE_LENGTH,0,stream1>>>(dynContext);
//	cudaStreamDestroy(stream1);
//
//	cudaStream_t stream2;
//	cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
//	//worker_dyn<<<N_BLOCKS, THREADS_PER_BLOCK,0,stream2>>>(appContext, dynContext);
//	cudaStreamDestroy(stream2);
//
//	cudaDeviceSynchronize();
//}
__device__ void app_kernel(TASK* task, APP_CONTEXT* appContext,RT_CONTEXT* dynContext){
	if (task== NULL)
	{
		task[1].blockId_x = 3;
	}
}




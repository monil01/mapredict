#ifndef _RT_DEVICE_H_
#define _RT_DEVICE_H_


#include <stdio.h>
#include "rt.h"

__device__ inline float cosineSleep(int n) {
	double d = threadIdx.x;
	for (int i = 0; i < n; i++)
		d = std::cos(d);

	return d;
}

__global__ void runtime(APP_CONTEXT appContext, RT_CONTEXT rtContext) {

	const int tid = threadIdx.x;
	const int sm_id = blockIdx.x;
	//const int subBlockIndex = tid / appContext.subBlockSize;
	const int localTid = tid % appContext.subBlockSize;
	const int nWorkers = rtContext.nWorkers;

#if DEBUG_TASK_PROCESSING_ORDER
	*(rtContext.task_order_ptr) = 0;
#endif

#if TIMER_SWITCH
	__shared__ clock_t smExecuteTime;
	__shared__ clock_t smWaitTime;
	__shared__ clock_t smQueueRetrieval;
	__shared__ clock_t smIQSignalingTime;
	__shared__ clock_t smOQSignalingTime;
	rtContext.smComputeTime[sm_id] = 0;
	rtContext.smWaitTime[sm_id] = 0;

#endif

	__shared__ int totalTasksRetrieved;
	__shared__ int totalTasksProcessed;
	__shared__ int queuedTaskCount;
	__shared__ int totalQueueLoad;
	__shared__ int roundRobin;

	__shared__ TASK queueBuffer[QUEUE_BUFFER_LENGTH];
	__shared__ int currentQueueIndex;

	__shared__ int outputTaskIndicies[QUEUE_BUFFER_LENGTH];
	__shared__ int outputTasksCount;
	__shared__ int timeout;

	__shared__ int inputQueueSizeBuffer[MAX_WORKERS];

#if SCHED_POLICY == MQ || SCHED_POLICY == AL
	__shared__ int minLoadedSM;
	__shared__ int minLoad;
#endif

	// initialization and reset
	if (tid == 0) {
		timeout = 0;
		outputTasksCount = 0;
		currentQueueIndex = 0;
		totalTasksRetrieved = 0;
		totalTasksProcessed = 0;
		rtContext.roundRobin[0] = 0;
		roundRobin=sm_id;

#if TIMER_SWITCH
		smExecuteTime = 0;
		smWaitTime = 0;
		smQueueRetrieval = 0;
		smIQSignalingTime = 0;
		smOQSignalingTime = 0;
#endif
#if SCHED_POLICY == MQ || SCHED_POLICY == AL
		minLoadedSM = -1;
		minLoad = INT_MAX;
#endif
	}

	__syncthreads();

	do {
#if TIMER_SWITCH
		clock_t loopStart;
		if (tid == 0)
		{
			loopStart = clock64();
		}
#endif

		/*********************************/
		/***** RETRIEVE QUEUE SIZES ******/
		/*********************************/

		// buffer queue sizes of all workers into shared memory.
		if (tid < nWorkers) {
			inputQueueSizeBuffer[tid] = rtContext.inputQueueSize[tid];
		}
		__syncthreads();
#if TIMER_SWITCH
		smIQSignalingTime += (clock64()-loopStart);
#endif

		// Check the worker's own queue size, if it is >0 then we have task(s)
		// to process
		if (tid == 0) {
#if SCHED_POLICY == MQ || SCHED_POLICY == AL
			minLoadedSM = -1;
			minLoad = INT_MAX;
#endif
			queuedTaskCount = inputQueueSizeBuffer[sm_id];
			if (queuedTaskCount > 0) {
				if (queuedTaskCount > MAX_QUEUE_RETRIEVAL)
					queuedTaskCount = MAX_QUEUE_RETRIEVAL;
				totalTasksRetrieved += queuedTaskCount;
				//printf("SM:%d\tIncoming:%d\t\n",sm_id,queuedTaskCount);
			}
			totalQueueLoad=0;
		}
		__syncthreads();

		/*********************************/
		/****** TERMINATION CHECK ********/
		/*********************************/
		// This block is to determine total load to detect whether the execution
		// should end.

		if (tid < nWorkers) {
			const int workerLoad = inputQueueSizeBuffer[tid];
			if (workerLoad  > 0)
				atomicAdd(&totalQueueLoad, workerLoad);
#if SCHED_POLICY == MQ || SCHED_POLICY == AL
			atomicMin(&minLoad, workerLoad);
#endif
		}

		__syncthreads();

		if (tid < nWorkers) {
#if SCHED_POLICY == MQ || SCHED_POLICY == AL
			if (minLoad == inputQueueSizeBuffer[tid])
				minLoadedSM = tid;
#endif
			// Ending condition, if nobody has job, then the computation is over.
			if (totalQueueLoad == 0)
				queuedTaskCount = -1;
			//totalQueueLoad = 0;

			if (queuedTaskCount == -1)
				rtContext.inputQueueSize[tid] = -1;
		}

		__syncthreads();
		// Nobody has any job, exit....
		if (queuedTaskCount == -1)
			break;

		/*********************************/
		/******* IDLENESS CHECK *********/
		/*********************************/
		// Check if there are no ready tasks.
		if (queuedTaskCount == 0) {
			//Sleep little bit and skip iteration
			cosineSleep(POLL_CYCLES_WORKER);
			if (tid == 0)
				timeout++;
			continue;
		}

		/*********************************/
		/*TASK RETRIEVAL FROM LOCAL QUEUE*/
		/*********************************/
		// Two staged memory access:
		//(1) First get the task index
		int taskIndex;
		if (tid < QUEUE_BUFFER_LENGTH && tid < queuedTaskCount) {
			taskIndex = rtContext.queues[sm_id * QUEUE_LENGTH + currentQueueIndex+tid];
		}
		__syncthreads();
		//(1) Then, copy the task.
		if (tid < QUEUE_BUFFER_LENGTH && tid < queuedTaskCount) {
			queueBuffer[tid] = appContext.tasks[taskIndex]; // GLOBAL_MEM
#if DEBUG_TASK_PRINT_FORMAT
			printf("SM:%d <= [%d]^%d\tIQS:%d\n", sm_id, queueBuffer[tid].id,
					tid, queuedTaskCount);
#endif
		}
		__syncthreads();

		/****************************************/
		/**** ITERATE THROUGH BUFFERED TASKS ****/
		/****************************************/
		for (int i = 0; i < queuedTaskCount; ++i) {
			TASK* current_task_shmem = &(queueBuffer[i]);

#if DEBUG_ON
			if (task_shmem == NULL)
			{
				printf("problem:%d %d %d %d",subBlockIndex, localTid,currentQueueIndex,queuedTaskCount);
			}
#endif

			/****************************************/
			/** START PROCESSING CURRENT TASK) ***/
			/****************************************/

#if TIMER_SWITCH
			clock_t taskProcessingStart;
			if (tid == 0)
			{
				taskProcessingStart = clock64();
			}
#endif

			outputTasksCount = 0;
#if DEBUG_TASK_PRINT_FORMAT
			if (localTid == 0)
				printf("SM:%d ~~ [%d]^%d\tIQS:%d\n", sm_id,
						current_task_shmem->id, tid, queuedTaskCount);
#endif
			app_kernel(current_task_shmem, &appContext, &rtContext);
#if DEBUG_TASK_PROCESSING_ORDER

				// task update
				if (tid%SUB_BLOCK_SIZE == 0)
				{
					rtContext.task_order[atomicAdd(rtContext.task_order_ptr,1)] =
					(blockIdx.x*DEBUG_SM_ID_COEFFICIENT)+task_shmem->id;
				}
#endif

			__syncthreads();

			/****************************************/
			/******* HANDLE NEWLY FREED TASKS *******/
			/****************************************/

			// Resolve dependencies in parallel. Store ready tasks in outputTasks
			if (localTid < current_task_shmem->nChildren && current_task_shmem != NULL) {
				int start = current_task_shmem->childrenStartIndex;
				int childCSRIndex =
						appContext.dependencies_csr[start + localTid]; //GLOBAL ACCESS
				TASK* child_global = &(appContext.tasks[childCSRIndex]);
				int val = atomicSub(&((child_global->nDependingParents)), 1); // GLOBAL ACCESS + ATOMIC
				if (val == 1) { // That means the child has no dependencies left.
					const int newChildLocalIndex = atomicAdd(&outputTasksCount,
							1);
					outputTaskIndicies[newChildLocalIndex] = childCSRIndex; // local
				}
			}
			// also decrease current load, to decide better on the next schedule
			if (localTid == 0){
				atomicAdd(&inputQueueSizeBuffer[sm_id],-1);
			}

			__syncthreads();


			// New tasks are stored in outputTasksCount. Now schedule them.
			if (localTid < outputTasksCount) {
				//SCHEDULING POLICIES
				int target_sm;
#if SCHED_POLICY == RR
				//target_sm = (atomicAdd(rtContext.roundRobin,1) + 1) % nWorkers;
				target_sm = (atomicAdd(&roundRobin,1) + 1) % nWorkers;
#elif (SCHED_POLICY == LF)
//				target_sm = sm_id;
//				if (localTid > 0) {
//					target_sm=(target_sm+1)%nWorkers;
//				}
				target_sm = (sm_id+localTid+1) % nWorkers;
#elif (SCHED_POLICY == AL)
				//printf("iqsb[%d:%d]:%d\ttql:%d\tmlsm:%d\tml:%d\n", sm_id, localTid, inputQueueSizeBuffer[sm_id], totalQueueLoad, minLoadedSM, minLoad);
				if (inputQueueSizeBuffer[sm_id] <= ((float)totalQueueLoad / nWorkers)){
					inputQueueSizeBuffer[sm_id]+=1;
				}
				else
					target_sm = minLoadedSM;
#elif SCHED_POLICY == MQ
				target_sm=minLoadedSM;
#endif

				int newTaskIndex = atomicAdd(
						&(rtContext.queueEndIndex[target_sm]), 1) % QUEUE_LENGTH;
				rtContext.queues[target_sm * QUEUE_LENGTH + newTaskIndex] = outputTaskIndicies[localTid];
				//memcpy(&(rtContext.queues[target_sm * QUEUE_LENGTH + newTaskIndex]),&(appContext.tasks[outputTasks[localTid]]), sizeof(TASK));
				// now signal
				atomicAdd(&(rtContext.inputQueueSize[target_sm]), 1);

#if DEBUG_TASK_PRINT_FORMAT
				printf("SM:%d => [%d]^%d => [%d]@SM:%d \n", sm_id,
						current_task_shmem->id, localTid, outputTaskIndicies[localTid],
						target_sm);
#endif

			}//if (localTid < outputTasksCount)
			else if (localTid == 0) { // no output tasks.
#if DEBUG_TASK_PRINT_FORMAT
				printf("SM:%d => [%d]^%d =>  \n", sm_id, current_task_shmem->id,
						localTid);
#endif
			}

		}

		__syncthreads();

#if TIMER_SWITCH
		clock_t taskProcessingEnd;
		if (tid == 0)
		{
			taskProcessingEnd = clock64();
		}
#endif

//		__threadfence();

		if (tid == 0 && queuedTaskCount > 0) {
			currentQueueIndex = (currentQueueIndex + queuedTaskCount) % QUEUE_LENGTH;
			totalTasksProcessed += queuedTaskCount;

			// decreasing own iqs after inserting to other queues ensures that
			// premature ending will not happen.

			atomicSub(&(rtContext.inputQueueSize[sm_id * PADDING]), queuedTaskCount);
		}

		__syncthreads();

#if TIMEOUT
		if (timeout > 5) {
//			if (tid == 0 && sm_id == 0) {
//				printf("Timing out... \n");
//			}
			break;
		}
		timeout++;
#endif

#if TIMER_SWITCH
		if (tid == 0)
		{
			clock_t loopEnd = clock64();
			smExecuteTime += (taskProcessingEnd-taskProcessingStart);
			smWaitTime += (loopEnd - taskProcessingEnd);
			smQueueRetrieval += (taskProcessingStart - loopStart);
		}
#endif
	} while (queuedTaskCount > -1);
#if DEBUG_TASK_PRINT_FORMAT
	if (tid == 0) {
		printf("SM:%d == %d of %d processed.\n", sm_id, totalTasksProcessed,
				totalTasksRetrieved);
	}
#endif

#if TIMER_SWITCH
	if (tid == 0)
	{
		rtContext->smComputeTime[sm_id] = smExecuteTime;
		rtContext->smWaitTime[sm_id] = smWaitTime-smOQSignalingTime;
		rtContext->smQueueInsertTime[sm_id] = 0;
		rtContext->smQueueRetrievalTime[sm_id] = smQueueRetrieval - smIQSignalingTime;
		rtContext->smDependencyResolutionTime[sm_id] = 0;
		rtContext->smIQSignalingTime[sm_id] = smIQSignalingTime;
		rtContext->smOQSignalingTime[sm_id] = smOQSignalingTime;
		rtContext->processed_task_counts[sm_id] = totalTasksProcessed;
//		rtContext->processed_task_counts[sm_id] = queueRetrievalCount*1000000+mainLoopCount;
	}
#endif

	if (tid == 0)
		rtContext.doneTasksTotal[sm_id] = totalTasksProcessed;
}

#endif // #ifndef _TEMPLATE_KERNEL_H_

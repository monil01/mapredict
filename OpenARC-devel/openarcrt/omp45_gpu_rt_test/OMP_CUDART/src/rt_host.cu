#include <cuda.h>
#include "helper/helper_cuda.h"
#include "helper/utils.h"

#include "rt.h"

void initRtContext_D(APP_CONTEXT& appContext_H, RT_CONTEXT& rtContext)
{
	size_t queueSizeTask =safeSize(MAX_WORKERS*QUEUE_LENGTH*sizeof(TASK));
	size_t queueSizeInt =safeSize(MAX_WORKERS*QUEUE_LENGTH*sizeof(int));
	size_t nBlocksSize =safeSize(MAX_WORKERS*sizeof(int));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.inputQueueSize), nBlocksSize));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.queueEndIndex), nBlocksSize));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.queues), queueSizeInt));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.queueLoad), nBlocksSize));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.averageLoad), safeSize(sizeof(float))));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.minQueueLoad), safeSize(sizeof(int))));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.minQueueIndex), safeSize(sizeof(int))));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.roundRobin), safeSize(sizeof(int))));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.doneTasksTotal), nBlocksSize));
	checkCudaErrors(cudaMalloc((void**) &(rtContext.enqueuedTaskCount), safeSize(sizeof(int))));


	//checkCudaErrors(cudaMemset(dynContext.queues, -1, queueSize));
	checkCudaErrors(cudaMemset(rtContext.inputQueueSize, 0, nBlocksSize));
	checkCudaErrors(cudaMemset(rtContext.queueEndIndex, 0, nBlocksSize));
	checkCudaErrors(cudaMemset(rtContext.queueLoad, 0, nBlocksSize));


	//Distribute ready tasks equally to the workers
	int nReadyTasks = appContext_H.readyToExecuteTasks_h->size();
	int taskPerWorker = nReadyTasks/rtContext.nWorkers;
	int nAsssigned=0;
	for (int i = 0; i < rtContext.nWorkers && nAsssigned < nReadyTasks; ++i) {
		int nTasks= taskPerWorker+((nReadyTasks%rtContext.nWorkers>i)?1:0);
		if (nTasks>0){
			checkCudaErrors(cudaMemcpy(
					rtContext.queues+(i*QUEUE_LENGTH),
					&((*appContext_H.readyToExecuteTasks_h)[nAsssigned]),
					sizeof(int)*nTasks,
					cudaMemcpyHostToDevice));
			// insert first task to first worker
			checkCudaErrors(cudaMemcpy(rtContext.inputQueueSize+i, &nTasks, sizeof(int), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(rtContext.queueEndIndex+i, &nTasks, sizeof(int), cudaMemcpyHostToDevice));
			nAsssigned += nTasks;
		}
	}


#if DEBUG_ON
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smComputeTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smWaitTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smQueueInsertTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smQueueRetrievalTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smDependencyResolutionTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smIQSignalingTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.smOQSignalingTime), N_BLOCKS * sizeof(clock_t)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.processed_task_counts), N_BLOCKS * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**) &(dynContext.task_order, appContext->totalTaskCount*2*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.task_order_ptr, sizeof(int)));

	int zero = 0;
	checkCudaErrors(cudaMalloc((void**) &(dynContext.queueLoadRecordArray, sizeof(_QUEUE_LOAD_RECORD)*500));
	checkCudaErrors(cudaMalloc((void**) &(dynContext.queueLoadRecordArraySize, sizeof(int)));
	checkCudaErrors(cudaMemcpy(dynContext.queueLoadRecordArraySize, &zero, sizeof(int), cudaMemcpyHostToDevice));
#endif

}


// TODO: Total number of tasks is basically a sum of all the TBs in the kernels
// that are to be executed together.
// TODO: In case of dynamic task insertion, there should be an estimate
// count of TASKs.In this case, openarc should use that value instead.

void initAppContext_H(APP_CONTEXT& appContext_h, int nTasks, int nEdges){
	appContext_h.totalTaskCount = nTasks;
	appContext_h.totalDependencyCount = nEdges;
	appContext_h.tasks = (TASK*) malloc(sizeof(TASK)*nTasks);
	appContext_h.dependencies_csr = (int*) malloc(sizeof(int)*nEdges);
	appContext_h.currentTaskCount=0;

	// These are host related only.
	appContext_h.dependencyMatrix_h = new vector<vector<int>>;
	appContext_h.dependencyMatrix_h->resize(nTasks);
	appContext_h.readyToExecuteTasks_h = new vector<int>;
	appContext_h.taskMap = new map<vector<int>, int,FlatIntListMapComparator>;
}

void initAppContext_D(APP_CONTEXT& appContext_h, APP_CONTEXT& appContext_d){

	// Basic copy
	appContext_d.totalDependencyCount = appContext_h.totalDependencyCount;
	appContext_d.totalTaskCount= appContext_h.totalTaskCount;
	appContext_d.currentDependencyCount = appContext_h.currentDependencyCount;
	appContext_d.currentTaskCount = appContext_h.currentTaskCount;
	appContext_d.appData = appContext_h.appData; // App data is assumed to be device only
	appContext_d.blockSize = appContext_h.blockSize;
	appContext_d.subBlockSize = appContext_h.subBlockSize;

	// Allocate tasks (i.e. TBs)
	size_t safe_size =safeSize(appContext_d.totalTaskCount*sizeof(TASK));
	size_t exact_size = appContext_d.totalTaskCount*sizeof(TASK);
	checkCudaErrors(cudaMalloc((void**) &(appContext_d.tasks), safe_size));
	checkCudaErrors(cudaMemcpy(appContext_d.tasks,
			appContext_h.tasks,
			exact_size,
			cudaMemcpyHostToDevice));


	// Allocate edges (i.e. dependencies)
	safe_size =safeSize(appContext_d.totalDependencyCount*sizeof(int));
	exact_size =appContext_d.totalDependencyCount*sizeof(int);
	checkCudaErrors(cudaMalloc((void**) &(appContext_d.dependencies_csr), safe_size));
	checkCudaErrors(cudaMemcpy(appContext_d.dependencies_csr,
			appContext_h.dependencies_csr,
			exact_size,
			cudaMemcpyHostToDevice));
}

void analyzeRun(RT_CONTEXT& rtContext_d, APP_CONTEXT& appContext_d, APP_CONTEXT& appContext_h){
	RT_CONTEXT rtContext_h;

	rtContext_h.nWorkers = rtContext_d.nWorkers;
	size_t queueSizeInt =safeSize(rtContext_h.nWorkers*QUEUE_LENGTH*sizeof(int));
	size_t nBlocksSize =safeSize(rtContext_h.nWorkers*sizeof(int));

	rtContext_h.queues = (int*) malloc(queueSizeInt);
	rtContext_h.queueEndIndex = (int*) malloc(nBlocksSize);
	rtContext_h.doneTasksTotal = (int*) malloc(nBlocksSize);
	checkCudaErrors(cudaMemcpy(rtContext_h.queues,rtContext_d.queues,queueSizeInt, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rtContext_h.queueEndIndex,rtContext_d.queueEndIndex, nBlocksSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(rtContext_h.doneTasksTotal,rtContext_d.doneTasksTotal, nBlocksSize, cudaMemcpyDeviceToHost));

	int totalTasksProcessed1 = 0;
	int totalTasksProcessed2 = 0;
	int min=INT_MAX;
	int max=INT_MIN;
	float avg = 0;

	for (int i = 0; i < rtContext_h.nWorkers; ++i) {
		int doneTasks = rtContext_h.doneTasksTotal[i];
//		rtContext.queues+(i*QUEUE_LENGTH),
//					&((*appContext_H.readyToExecuteTasks_h)[nAsssigned]),
//					sizeof(int)*nTasks,
//					cudaMemcpyHostToDevice));
		//printf("doneTasksTotal[%d]\t%d\n",i,rtContext_h.doneTasksTotal[i]);
		printf("queueEndIndex[%d]\t%d\n",i,rtContext_h.queueEndIndex[i]);

		totalTasksProcessed1 += doneTasks;
		totalTasksProcessed2 += rtContext_h.queueEndIndex[i];

		avg = (avg*i + rtContext_h.queueEndIndex[i])/(i+1);
		if (min > rtContext_h.queueEndIndex[i])
		{
			min = rtContext_h.queueEndIndex[i];
		}

		if (max < rtContext_h.queueEndIndex[i]){
			max = rtContext_h.queueEndIndex[i];
		}

	}
	printf("totalSent\t%d\n",appContext_d.totalTaskCount);
	printf("totalProcessed1\t%d\n",totalTasksProcessed1);
	printf("totalProcessed2\t%d\n",totalTasksProcessed2);
	printf("minTaskLoad\t%d\n",min);
	printf("maxTaskLoad\t%d\n",max);
	printf("avgTaskLoad\t%d\n",totalTasksProcessed2/rtContext_h.nWorkers);
	printf("taskLoadDev\t%d\n",max-min);

}

void processOutDependency(APP_CONTEXT& appContext_h, int taskIndex, int nIdentifiers ...){
	vector<int> outDependencyIdentifiers;
	va_list arguments;
	va_start ( arguments, nIdentifiers);
	for ( int x = 0; x < nIdentifiers; x++ ){// Loop until all numbers are added
		int arg = va_arg ( arguments, int );
		outDependencyIdentifiers.push_back(arg);
	}
	va_end ( arguments );

	(*appContext_h.taskMap)[outDependencyIdentifiers] = taskIndex;
}


void processInDependency(APP_CONTEXT& appContext_h, int taskIndex, int nIdentifiers ...){
	vector<int> inDependencyIdentifiers;
	va_list arguments;
	va_start ( arguments, nIdentifiers);
	for ( int x = 0; x < nIdentifiers; x++ ){// Loop until all numbers are added
		int arg = va_arg ( arguments, int );
		inDependencyIdentifiers.push_back(arg);
	}
	va_end ( arguments );

	map<vector<int>, int,FlatIntListMapComparator>::iterator iter =
			(*appContext_h.taskMap).find(inDependencyIdentifiers);
	if (iter!=(*appContext_h.taskMap).end()){
		addDependency(appContext_h,taskIndex,(*appContext_h.taskMap)[inDependencyIdentifiers]);
	}
}
int addTask(APP_CONTEXT& appContext_h, char kernelType){
	TASK* task = &(appContext_h.tasks[appContext_h.currentTaskCount]);
	task->id=appContext_h.currentTaskCount;
	task->childrenStartIndex = 0;
	task->nChildren = 0;
	task->nDependingParents = 0;
	task->kernel_type = kernelType;

	return appContext_h.currentTaskCount++;
}

// TODO: this method assumes that the dependencies are static
void addDependency(APP_CONTEXT& appContext_h, int childTaskIndex, int parentTaskIndex){
	TASK* childTask = &(appContext_h.tasks[childTaskIndex]);
	TASK* parentTask = &(appContext_h.tasks[parentTaskIndex]);
	parentTask->nChildren++;
	childTask->nDependingParents++;
	(*appContext_h.dependencyMatrix_h)[parentTaskIndex].push_back(childTaskIndex);
#ifdef VERBOSE
	printf("ADD_DEP:\t[%d]=>[%d]\n",childTaskIndex, parentTaskIndex);
#endif
}

void buildCSR(APP_CONTEXT& appContext_h){
	int dependenciesIndex = 0;

#if DEBUG_TASK_PRINT_FORMAT_HOST
	printf("CSR INFO:\n");
#endif

	for (int i = 0; i < appContext_h.currentTaskCount; ++i) {
		TASK* parentTask = &appContext_h.tasks[i];
		parentTask->childrenStartIndex = dependenciesIndex;
		if (parentTask->nDependingParents == 0){
			appContext_h.readyToExecuteTasks_h->push_back(parentTask->id);
		}
		vector<int>* children = &(appContext_h.dependencyMatrix_h->at(i));
#if DEBUG_TASK_PRINT_FORMAT_HOST
	printf("[%d] => ");
#endif
		for (int j = 0; j < children->size(); ++j) {
			appContext_h.dependencies_csr[dependenciesIndex] = children->at(j);
#if DEBUG_TASK_PRINT_FORMAT_HOST
			printf("[%d] ",children->at(j));
#endif

			dependenciesIndex++;
		}
#if DEBUG_TASK_PRINT_FORMAT_HOST
		printf("\n");
#endif
	}

	printf("nTasksReal=%d\n",appContext_h.currentTaskCount);
	printf("nTasksEst=%d\n", appContext_h.totalTaskCount);
	printf("nDepsReal=%d\n",dependenciesIndex);
	printf("nDepsEst=%d\n",appContext_h.totalDependencyCount);
	appContext_h.currentDependencyCount = dependenciesIndex;
	appContext_h.totalTaskCount = appContext_h.currentTaskCount;
	appContext_h.totalDependencyCount = appContext_h.currentDependencyCount;
}

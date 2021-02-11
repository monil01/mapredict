#ifndef _RT_H
#define _RT_H

#include <vector>
#include <map>
#include <cstdint> // to use intptr_t
#include <cstdarg> // to use flat list dependency identifiers
#include "app.cuh"
#include "helper/helper_cuda.h"

using namespace std;


struct FlatIntListMapComparator {
	bool operator() (const vector<int> & lhs, const vector<int> & rhs) const {

		bool smaller = false;
		bool equal = true;

		for (int i = 0; i < lhs.size() && (!smaller && equal); ++i){
			smaller = lhs[i] < rhs[i];
			equal  = lhs[i] == rhs[i];
		}

		return smaller;
	}
};

typedef struct _TASK {
	// blockIdx is the block id that is intended to replace in-kernel blockIdX.x
	// App code should decide indexing based on the bid.

	uint id;

	uint childrenStartIndex;
	char kernel_type;
	char nChildren;
	int nDependingParents;

	// This pointer is optional and holds task specific data. May needed
	// when tasks are created dynamically.
	TASK_DATA taskData;

} TASK;

typedef struct _APPCONTEXT {
	/** COMMON (BOTH DEVICE AND HOST) VARIABLES **/
	int blockSize;//Threads per task or TB
	int subBlockSize;//Threads per task or TB
	TASK* tasks;
	int totalTaskCount;
	int* dependencies_csr;
	int totalDependencyCount;
	int currentTaskCount;
	int currentDependencyCount;

	/** HOST USE ONLY VARIABLES **/
	vector<vector<int> >* dependencyMatrix_h;
	vector<int>* readyToExecuteTasks_h;
	map<vector<int>, int,FlatIntListMapComparator>* taskMap;

	// This struct will hold all user data that will be used by the kernels in
	// the app. Application is responsible for the declaration of it in app.h
	APP_DATA appData;
} APP_CONTEXT;

typedef unsigned long long TIME_MSEC;

typedef struct _RTCONTEXT{
	// Internal
	int* queues; // N_BLOCKS times task queues each having QUEUE_LENGTH length
	int* queueEndIndex; // queue tail

	int nWorkers;

	int* inputQueueSize;

	//dist scheduling related
	int* queueLoad;
	float* averageLoad;
	int* minQueueLoad;
	int* minQueueIndex;

	int* roundRobin;
	int* doneTasksTotal;

	int *enqueuedTaskCount;

	// Statistic related
#if DEBUG_ON
	int* task_order;
	int* task_order_ptr;
	int* processed_task_counts;

	clock_t* smComputeTime;
	clock_t* smWaitTime;
	clock_t* smQueueInsertTime;
	clock_t* smQueueRetrievalTime;
	clock_t* smDependencyResolutionTime;
	clock_t* smIQSignalingTime;
	clock_t* smOQSignalingTime;

	int totalTaskCount;

	QUEUE_LOAD_RECORD* queueLoadRecordArray;
	int* queueLoadRecordArraySize;

#endif
} RT_CONTEXT;


// Host related functions
void initAppContext_H(APP_CONTEXT& appContext_h, int nTasks, int nEdges);
int addTask(APP_CONTEXT& appContext_h, char kernelType);
void addDependency(APP_CONTEXT& appContext_h, int srcTaskIndex, int dstTaskIndex);
void processOutDependency(APP_CONTEXT& appContext_h, int taskIndex, int nIdentifiers ...);
void processInDependency(APP_CONTEXT& appContext_h, int taskIndex, int nIdentifiers ...);
void buildCSR(APP_CONTEXT& appContext_h);

void analyzeRun(RT_CONTEXT& rtContext_d, APP_CONTEXT& appContext_d, APP_CONTEXT& appContext_h);

// Device related host functions
void initRtContext_D(APP_CONTEXT& appContext, RT_CONTEXT& rtContext);
void initAppContext_D(APP_CONTEXT& appContext_h, APP_CONTEXT& appContext_d);

// Device code
__global__ void runtime(APP_CONTEXT appContext, RT_CONTEXT rtContext);
__device__ void  app_kernel(TASK* task, APP_CONTEXT* appContext, RT_CONTEXT* dynContext);

// MODE not used currently
#define MODE_STATIC 0
#define MODE_DYNAMIC 1
#define MODE MODE_STATIC

#define RR 0
#define LF 1
#define AL 2
#define MQ 3

# define SCHED_POLICY LF

#define QUEUE_LENGTH 4096
#define QUEUE_BUFFER_LENGTH 16
#define MAX_WORKERS 64
#define N_WORKERS 56
#define POLL_CYCLES_WORKER 2

#define MAX_QUEUE_RETRIEVAL QUEUE_BUFFER_LENGTH
#define KHZ 928000.0 // for timing purposes, k40c

#define PADDING 1

#define TIMEOUT 0
#define TIMER_SWITCH 0


#define DEBUG_ON 0
#define DEBUG_TASK_PROCESSING_ORDER 0
#define DEBUG_QUEUE_LOAD_TRACKING 0
#define DEBUG_QUEUE_LOAD_TRACKING_CYCLES KHZ*5

#define HALT_DETECTION 0
#define HALT_DETECTION_CYCLES KHZ * 1500

#define DEBUG_SM_ID_COEFFICIENT 100000
#define DEBUG_TASK_PRINT_FORMAT 0
#define DEBUG_TASK_PRINT_FORMAT_HOST 0



#if DEBUG_ON
typedef struct _QUEUE_LOAD_RECORD{
	clock_t time;
	int load[N_BLOCKS];
} QUEUE_LOAD_RECORD;
#endif

#endif

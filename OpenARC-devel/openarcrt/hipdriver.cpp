#include "openacc.h"
#include "openaccrt_ext.h"

std::map<const void *, int> HipDriver::pinnedHostMemCounter;
std::vector<const void *> HipDriver::hostMemToUnpin;

HipDriver::HipDriver(acc_device_t devType, int devNum, std::set<std::string>kernelNames, HostConf_t *conf, int numDevices, const char *baseFileName) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HipDriver(%s, %d)\n", HI_get_device_type_string(devType), devNum);
    }
#endif
    dev = devType;
    device_num = devNum;
    num_devices = numDevices;
	fileNameBase = std::string(baseFileName);

    for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        kernelNameSet.insert(*it);
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HipDriver(%s, %d)\n", HI_get_device_type_string(devType), devNum);
    }
#endif
}

HI_error_t HipDriver::init(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::init()\n");
    }    
#endif
    hipError_t err;
    err = hipInit(0);
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
	err = hipSetDevice(device_num);
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
	err = hipGetDevice(&device_num);
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
	err = hipDeviceGet(&hipDevice, device_num);
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
	err = hipDeviceGetAttribute(&maxNumThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, device_num);
	err = hipDeviceGetAttribute(&maxBlockX, hipDeviceAttributeMaxBlockDimX, device_num);
	err = hipDeviceGetAttribute(&maxBlockY, hipDeviceAttributeMaxBlockDimY, device_num);
	err = hipDeviceGetAttribute(&maxBlockZ, hipDeviceAttributeMaxBlockDimZ, device_num);
	err = hipDeviceGetAttribute(&maxGridX, hipDeviceAttributeMaxGridDimX, device_num);
	err = hipDeviceGetAttribute(&maxGridY, hipDeviceAttributeMaxGridDimY, device_num);
	err = hipDeviceGetAttribute(&maxGridZ, hipDeviceAttributeMaxGridDimZ, device_num);
	err = hipDeviceGetAttribute(&compute_capability_major, hipDeviceAttributeComputeCapabilityMajor, device_num);
	err = hipDeviceGetAttribute(&compute_capability_minor, hipDeviceAttributeComputeCapabilityMinor, device_num);

	char name[256];
    HostConf_t * tconf = getHostConf(threadID);
    int thread_id = tconf->threadID;
#ifdef _OPENARC_PROFILE_
    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::init(): host Thread %d initializes device %d\n", thread_id, device_num);
#endif
    std::string codeName = fileNameBase + std::string(".hip");
    if( access( codeName.c_str(), F_OK ) == -1 ) {
        std::string command = std::string("hipcc --genco ") + fileNameBase + std::string(".hip.cpp -o ") + codeName;
        system(command.c_str());
    }
    err = hipModuleLoad(&hipModule, codeName.c_str());
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);

	hipStream_t s0, s1;
	hipEvent_t e0, e1;
#ifdef USE_BLOCKING_STREAMS
	err = hipStreamCreate(&s0);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Stream Create FAIL with error = %d\n", err);
    	exit(1);
    }
	err = hipStreamCreate(&s1);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Stream Create FAIL with error = %d\n", err);
    	exit(1);
    }
#else
	err = hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Stream Create FAIL with error = %d\n", err);
    	exit(1);
    }
	err = hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Stream Create FAIL with error = %d\n", err);
    	exit(1);
    }
#endif
    queueMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s0;
    queueMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s1;
#ifdef INIT_DEBUG
    fprintf(stderr, "[DEBUG] HIP Streams are created.\n");
    fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif

	err = hipEventCreate(&e0);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Event Create FAIL with error = %d\n", err);
    	exit(1);
    }
	err = hipEventCreate(&e1);
    if (err != hipSuccess) {
    	fprintf(stderr, "[ERROR in HipDriver::init()] Event Create FAIL with error = %d\n", err);
    	exit(1);
    }
    std::map<int, hipEvent_t> eventMap;
    eventMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e0;
    eventMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e1;
    threadQueueEventMap[thread_id] = eventMap;
    masterAddressTableMap[thread_id] = new addresstable_t();
    masterHandleTable[thread_id] = new addressmap_t();
    postponedFreeTableMap[thread_id] = new asyncfreetable_t();
    postponedTempFreeTableMap[thread_id] = new asynctempfreetable_t();
    postponedTempFreeTableMap2[thread_id] = new asynctempfreetable2_t();
    memPoolMap[thread_id] = new memPool_t();
    tempMallocSizeMap[thread_id] = new sizemap_t();

    createKernelArgMap();

    init_done = 1;

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::init()\n");
    }    
#endif
    return HI_success;
}

HI_error_t HipDriver::createKernelArgMap(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::createKernelArgMap()\n");
    }    
#endif
    hipError_t err;

    HostConf_t * tconf = getHostConf(threadID);
    std::map<std::string, hipFunction_t> kernelMap;
	std::map<std::string, kernelParams_t*> kernelArgs;

    for(std::set<std::string>::iterator it=kernelNameSet.begin(); it!=kernelNameSet.end(); ++it) {
        hipFunction_t hipFunc;
        const char *kernelName = (*it).c_str();
        //printf("[%s:%d][%s] kernel[%s]\n", __FILE__, __LINE__, __func__, kernelName);
        err = hipModuleGetFunction(&hipFunc, hipModule, kernelName);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Function Load FAIL on %s with error = %d\n", kernelName, err);
            exit(1);
        }
        kernelMap[*it] = hipFunc;

		kernelParams_t *kernelParams = new kernelParams_t;
		kernelParams->num_args = 0;
		kernelArgs.insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
    }

	tconf->kernelArgsMap[this] = kernelArgs;
    tconf->kernelsMap[this]=kernelMap;
    int thread_id = tconf->threadID;
    if( queueMap.count(0+thread_id*MAX_NUM_QUEUES_PER_THREAD) == 0 ) {
		hipStream_t s0, s1;
#ifdef USE_BLOCKING_STREAMS
		err = hipStreamCreate(&s0);
    	if (err != hipSuccess) {
    		fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Stream Create FAIL with error = %d\n", err);
    		exit(1);
    	}
		err = hipStreamCreate(&s1);
    	if (err != hipSuccess) {
    		fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Stream Create FAIL with error = %d\n", err);
    		exit(1);
    	}
#else
        err = hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Stream Create FAIL with error = %d\n", err);
            exit(1);
        }
#endif
        err = hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Stream Create FAIL with error = %d\n", err);
            exit(1);
        }
#endif
#endif
    	queueMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s0;
    	queueMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s1;
#ifdef INIT_DEBUG
    	fprintf(stderr, "[DEBUG] HIP Streams are created.\n");
    	fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
	}

    if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(thread_id*MAX_NUM_QUEUES_PER_THREAD) == 0) ) {
        hipEvent_t e0, e1;
        err = hipEventCreate(&e0);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Event Create FAIL with error = %d\n", err);
            exit(1);
        }
#endif
        std::map<int, hipEvent_t> eventMap;
        eventMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e0;
        err = hipEventCreate(&e1);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::createKernelArgMap()] Event Create FAIL with error = %d\n", err);
            exit(1);
        }
#endif
        eventMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e1;
        threadQueueEventMap[thread_id] = eventMap;
        masterAddressTableMap[thread_id] = new addresstable_t();
        masterHandleTable[thread_id] = new addressmap_t();
        postponedFreeTableMap[thread_id] = new asyncfreetable_t();
        postponedTempFreeTableMap[thread_id] = new asynctempfreetable_t();
        postponedTempFreeTableMap2[thread_id] = new asynctempfreetable2_t();
        memPoolMap[thread_id] = new memPool_t();
        tempMallocSizeMap[thread_id] = new sizemap_t();
#ifdef INIT_DEBUG
        fprintf(stderr, "[DEBUG] HIP events are created.\n");
        fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
	}

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::createKernelArgMap()\n");
    }    
#endif
    return HI_success;
}

HI_error_t HipDriver::HI_register_kernels(std::set<std::string>kernelNames, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_register_kernels()\n");
    }
#endif
#ifdef INIT_DEBUG
    fprintf(stderr, "[DEBUG] call HI_register_kernel().\n");
    fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipError_t err;
    for(std::set<std::string>::iterator it=kernelNames.begin(); it!=kernelNames.end(); ++it) {
		if( kernelNameSet.count(*it) == 0 ) {
        	// Create argument mapping for the kernel
        	const char *kernelName = (*it).c_str();
        	//fprintf(stderr, "[INFO in HipDriver::HI_register_kernels()] Function to load: %s\n", kernelName);
        	hipFunction_t hipFunc;
        	kernelParams_t *kernelParams = new kernelParams_t;
        	kernelParams->num_args = 0;
        	(tconf->kernelArgsMap[this]).insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
        	err = hipModuleGetFunction(&hipFunc, hipModule, kernelName);
        	if (err != hipSuccess) {
            	fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Function Load FAIL on %s with error = %d\n", kernelName, err);
            	exit(1);
        	}
        	(tconf->kernelsMap[this])[*it] = hipFunc;
		}
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_set_device_num);
#else
#ifdef _OPENMP
        #pragma omp critical(acc_set_device_num_critical)
#endif
#endif
    for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
        if( kernelNameSet.count(*it) == 0 ) {
            //Add a new kernel name.
            kernelNameSet.insert(*it);
        }    
    }    
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_set_device_num);
#endif
//[DEBUG on Jan. 7, 2021] Why we need this code here?
/*
    int thread_id = tconf->threadID;
    if( queueMap.count(0+thread_id*MAX_NUM_QUEUES_PER_THREAD) == 0 ) {
        hipStream_t s0, s1;
#ifdef USE_BLOCKING_STREAMS
        err = hipStreamCreate(&s0);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Stream Create FAIL with error = %d\n", err);
            exit(1);
        }
#endif
        err = hipStreamCreate(&s1);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Stream Create FAIL with error = %d (%s)\n", err);
            exit(1);
        }
#endif
#else
        err = hipStreamCreateWithFlags(&s0, hipStreamNonBlocking);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Stream Create FAIL with error = %d (%s)\n", err);
            exit(1);
        }
#endif
        err = hipStreamCreateWithFlags(&s1, hipStreamNonBlocking);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Stream Create FAIL with error = %d (%s)\n", err);
            exit(1);
        }
#endif
#endif
        queueMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s0;
        queueMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD] = s1;
#ifdef INIT_DEBUG
        fprintf(stderr, "[DEBUG] HIP Streams are created.\n");
        fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
    }

    if( (threadQueueEventMap.count(thread_id) == 0) || (threadQueueEventMap.at(thread_id).count(thread_id*MAX_NUM_QUEUES_PER_THREAD) == 0) ) {
        hipEvent_t e0, e1;
        err = hipEventCreate(&e0);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Event Create FAIL with error = %d (%s)\n", err);
            exit(1);
        }
#endif
        std::map<int, hipEvent_t> eventMap;
        eventMap[0+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e0;
        err = hipEventCreate(&e1);
#ifdef _OPENARC_PROFILE_
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_register_kernels()] Event Create FAIL with error = %d (%s)\n", err);
            exit(1);
        }
#endif
        eventMap[1+thread_id*MAX_NUM_QUEUES_PER_THREAD]= e1;
        threadQueueEventMap[thread_id] = eventMap;
        masterAddressTableMap[thread_id] = new addresstable_t();
        masterHandleTable[thread_id] = new addressmap_t();
        postponedFreeTableMap[thread_id] = new asyncfreetable_t();
        postponedTempFreeTableMap[thread_id] = new asynctempfreetable_t();
        postponedTempFreeTableMap2[thread_id] = new asynctempfreetable2_t();
        memPoolMap[thread_id] = new memPool_t();
        tempMallocSizeMap[thread_id] = new sizemap_t();
#ifdef INIT_DEBUG
        fprintf(stderr, "[DEBUG] HIP events are created.\n");
        fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
    }
*/

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_register_kernels()\n");
    }
#endif
    return HI_success;
}

int HipDriver::HI_get_num_devices(acc_device_t devType, int threadID) {
	int numDevices;
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_get_num_devices()\n");
    }    
#endif
    hipError_t err;
    err = hipInit(0);
    if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);

    if( devType == acc_device_gpu ) {
    	err = hipGetDeviceCount(&numDevices);
    	if (err != hipSuccess) printf("[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
    } else {
        numDevices = 0; 
    }  
    printf("[%s:%d][%s] count[%d]\n", __FILE__, __LINE__, __func__, numDevices);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_get_num_devices()\n");
    }    
#endif

    return numDevices;
}

HI_error_t HipDriver::destroy(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::destroy()\n");
    }
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::destroy()\n");
    }
#endif
    return HI_success;
}

// Pin host memory
HI_error_t HipDriver::HI_pin_host_memory(const void* hostPtr, size_t size, int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_pin_host_memory()\n");
    }    
#endif
    HI_error_t result = HI_success;
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {    
        const void *host = hostPtr;
        //If the hostPtr is already pinned
        if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end() && HipDriver::pinnedHostMemCounter[host] > 0) {
            HipDriver::pinnedHostMemCounter[host]++;
        } else {
            hipError_t hipResult = hipHostRegister((void*)host, size, hipHostRegisterPortable);
            if(hipResult == hipSuccess) {
                HipDriver::pinnedHostMemCounter[host] = 1; 
#ifdef _OPENARC_PROFILE_
                HostConf_t * tconf = getHostConf(threadID);
                tconf->IPMallocCnt++;
                tconf->IPMallocSize += size;
#endif
            } else  {
                result = HI_error;
            }    

        }    
    }    
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_pin_host_memory()\n");
    }    
#endif
    return result;
}

// Pin host memory if unpinned.
// If the memory is already pinned, it does not increase pinnedHostMemCounter.
// [CAUTION] this will work only if hostPtr refers to the base address of allocated
// memory.
HI_error_t HipDriver::pin_host_memory_if_unpinned(const void* hostPtr, size_t size, int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::pin_host_memory_if_unpinned()\n");
    }
#endif
    HI_error_t result = HI_success;
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
        const void * host = hostPtr;
        //If the hostPtr is already pinned
        if(HipDriver::pinnedHostMemCounter.find(host) == HipDriver::pinnedHostMemCounter.end() )  {
            hipError_t hipResult = hipHostRegister((void*)host, size, hipHostRegisterPortable);
            if(hipResult == hipSuccess) {
                HipDriver::pinnedHostMemCounter[host] = 1;
#ifdef _OPENARC_PROFILE_
                HostConf_t * tconf = getHostConf(threadID);
                tconf->IPMallocCnt++;
                tconf->IPMallocSize += size;
#endif
            } else  {
                result = HI_error;
            }
        }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::pin_host_memory_if_unpinned()\n");
    }
#endif
    return result;
}

void HipDriver::HI_unpin_host_memory(const void* hostPtr, int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_unpin_host_memory()\n");
    }
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
        const void * host = hostPtr;
        //Unpin if the hostPtr is already pinned
        if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end()) {
            if(HipDriver::pinnedHostMemCounter[host] > 1) {
                HipDriver::pinnedHostMemCounter[host]--;
            } else
            {
                hipError_t hipResult = hipHostUnregister((void*)host);
                if(hipResult == hipSuccess){
                    //HipDriver::pinnedHostMemCounter[host] = 0;
                    HipDriver::pinnedHostMemCounter.erase(host);
#ifdef _OPENARC_PROFILE_
                    HostConf_t * tconf = getHostConf(threadID);
                    tconf->IPFreeCnt++;
#endif
                } else {
                    fprintf(stderr, "[ERROR in HipDriver::HI_unpin_host_memory()] Cannot unpin host memory with error %d\n", hipResult);
                    exit(1);
                }
            }
        }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_unpin_host_memory()\n");
    }
#endif
}

void HipDriver::dec_pinned_host_memory_counter(const void* hostPtr)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::dec_pinned_host_memory_counter()\n");
    }
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
        const void * host = hostPtr;
        //If the hostPtr is already pinned
        if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end()) {
            HipDriver::pinnedHostMemCounter[host]--;
        } else {
            fprintf(stderr, "[ERROR in HipDriver::dec_pinned_host_memory_counter()] \n");
        }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::dec_pinned_host_memory_counter()\n");
    }
#endif
}

void HipDriver::inc_pinned_host_memory_counter(const void* hostPtr)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::inc_pinned_host_memory_counter()\n");
    }
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
        const void * host = hostPtr;
        //If the hostPtr is already pinned
        if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end()) {
            HipDriver::pinnedHostMemCounter[host]++;
        } else {
            fprintf(stderr, "[ERROR in HipDriver::inc_pinned_host_memory_counter()] \n");
            exit(1);
        }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::inc_pinned_host_memory_counter()\n");
    }
#endif
}

//Unpin host memories whose counters are less than 1.
//This also frees corresponding device memory.
void HipDriver::unpin_host_memory_all(int asyncID, int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::unpin_host_memory_all(%d)\n", asyncID);
    }
    HostConf_t * tconf = getHostConf(threadID);
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
        #pragma omp critical (victim_cache_critical)
#endif
#endif
        {
        addresstable_t::iterator it = HipDriver::auxAddressTable.find(asyncID);
        if(it != HipDriver::auxAddressTable.end()) {
            HipDriver::hostMemToUnpin.clear();
            for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
                const void * host = it2->first;
                //If the hostPtr is already pinned
                if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end()) {
                    if(HipDriver::pinnedHostMemCounter[host] < 1) {
                        hipError_t hipResult = hipHostUnregister((void*)host);
                        if(hipResult == hipSuccess){
                            HipDriver::pinnedHostMemCounter.erase(host);
                            HipDriver::hostMemToUnpin.push_back(it2->first);
#ifdef _OPENARC_PROFILE_
                            tconf->IPFreeCnt++;
#endif
                        } else {
                            fprintf(stderr, "[ERROR in HipDriver::unpin_host_memory_all(%d)] Cannot unpin host memory with error %d\n", asyncID, hipResult);
                            exit(1);
                        }
                        //Free corresponding device memory.
                        addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
                        hipFree(aet->basePtr);
#ifdef _OPENARC_PROFILE_
                        tconf->IDFreeCnt++;
                        //[FIXME] tconf->CIDMemorySize should be adjusted here.
#endif

                    }
                }
            }
            while( !HipDriver::hostMemToUnpin.empty() ) {
                (it->second)->erase(HipDriver::hostMemToUnpin.back());
                HipDriver::hostMemToUnpin.pop_back();
            }
        }
        }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::unpin_host_memory_all(%d)\n", asyncID);
    }
#endif
}

//Unpin host memories whose counters are less than 1.
void HipDriver::unpin_host_memory_all(int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::unpin_host_memory_all()\n");
    }
    HostConf_t * tconf = getHostConf(threadID);
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_pin_host_memory);
#else
#ifdef _OPENMP
    #pragma omp critical (pin_host_memory_critical)
#endif
#endif
    {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
        #pragma omp critical (victim_cache_critical)
#endif
#endif
        for( addresstable_t::iterator it = HipDriver::auxAddressTable.begin(); it != HipDriver::auxAddressTable.end(); ++it) {
            HipDriver::hostMemToUnpin.clear();
            for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
                const void * host = it2->first;
                //If the hostPtr is already pinned
                if(HipDriver::pinnedHostMemCounter.find(host) != HipDriver::pinnedHostMemCounter.end()) {
                    if(HipDriver::pinnedHostMemCounter[host] < 1) {
                        hipError_t hipResult = hipHostUnregister((void*)host);
                        if(hipResult == hipSuccess){
                            HipDriver::pinnedHostMemCounter.erase(host);
                            HipDriver::hostMemToUnpin.push_back(it2->first);
#ifdef _OPENARC_PROFILE_
                            tconf->IDFreeCnt++;
                            //[FIXME] tconf->CIDMemorySize should be adjusted here.
#endif
                        } else {
                            fprintf(stderr, "[ERROR in HipDriver::unpin_host_memory_all()] Cannot unpin host memory with error %d\n", hipResult);
                            exit(1);
                        }
                        //Free corresponding device memory.
                        addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
                        hipFree(aet->basePtr);
#ifdef _OPENARC_PROFILE_
                        tconf->IDFreeCnt++;
                        //[FIXME] tconf->CIDMemorySize should be adjusted here.
#endif
                    }
                }
            }
            while( !HipDriver::hostMemToUnpin.empty() ) {
                (it->second)->erase(HipDriver::hostMemToUnpin.back());
                HipDriver::hostMemToUnpin.pop_back();
            }
        }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_pin_host_memory);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::unpin_host_memory_all()\n");
    }
#endif
}

void HipDriver::release_freed_device_memory(int asyncID, int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::release_freed_device_memory(%d)\n", asyncID);
    }
    HostConf_t * tconf = getHostConf(threadID);
#endif
    {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
        #pragma omp critical (victim_cache_critical)
#endif
#endif
        {
        addresstable_t::iterator it = HipDriver::auxAddressTable.find(asyncID);
        if(it != HipDriver::auxAddressTable.end()) {
            for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
                //Free corresponding device memory.
                addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
                hipFree(aet->basePtr);
#ifdef _OPENARC_PROFILE_
                tconf->IDFreeCnt++;
                //[FIXME] tconf->CIDMemorySize should be adjusted here.
#endif
            }
        }
        }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::release_freed_device_memory(%d)\n", asyncID);
    }
#endif
}

void HipDriver::release_freed_device_memory(int threadID)
{
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::release_freed_device_memory()\n");
    }
    HostConf_t * tconf = getHostConf(threadID);
#endif
    {
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_victim_cache);
#else
#ifdef _OPENMP
        #pragma omp critical (victim_cache_critical)
#endif
#endif
        for( addresstable_t::iterator it = HipDriver::auxAddressTable.begin(); it != HipDriver::auxAddressTable.end(); ++it ) {
            for( addressmap_t::iterator it2 = (it->second)->begin(); it2 != (it->second)->end(); ++it2 ) {
                //Free corresponding device memory.
                addresstable_entity_t *aet = (addresstable_entity_t*) it2->second;
                hipFree(aet->basePtr);
#ifdef _OPENARC_PROFILE_
                tconf->IDFreeCnt++;
                //[FIXME] tconf->CIDMemorySize should be adjusted here.
#endif
            }
        }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_victim_cache);
#endif
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::release_freed_device_memory()\n");
    }
#endif
}

HI_error_t HipDriver::HI_malloc1D(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_malloc1D(hostPtr = %lx, asyncID = %d, size = %lu, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, count, threadID);
    }    
#endif
#ifdef INIT_DEBUG
    fprintf(stderr, "[DEBUG] call HI_malloc1D().\n");
    fprintf(stderr, "[DEBUG] Current host thread: %ld\n", syscall(__NR_gettid));
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf == NULL ) {
        fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] No host configuration exists for the current host thread (thread ID: %d); exit!\n", threadID);
        exit(1);
    }
    if( tconf->device->init_done == 0 ) {
        tconf->HI_init(DEVICE_NUM_UNDEFINED);
    }
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    HI_error_t result = HI_error;

    if(HI_get_device_address(hostPtr, devPtr, NULL, NULL, asyncID, tconf->threadID) == HI_success ) {
        if( unifiedMemSupported ) {
            result = HI_success;
        } else {
            fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] Duplicate device memory allocation for the same host data (%lx) by thread %d is not allowed; exit!\n",(long unsigned int)hostPtr, tconf->threadID);
            exit(1);
        }    
    } else {
    	hipError_t hipResult = hipSuccess;
#if VICTIM_CACHE_MODE <= 1
        memPool_t *memPool = memPoolMap[tconf->threadID];
        std::multimap<size_t, void *>::iterator it = memPool->find(count);
        if (it != memPool->end()) {
#ifdef _OPENARC_PROFILE_
            if( HI_openarcrt_verbosity > 2 ) {
                fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_malloc1D(%d, %lu) reuses memories in the memPool\n", asyncID, count);
            }
#endif
            *devPtr = it->second;
            memPool->erase(it);
            current_mempool_size -= count;
        } else {
            if( current_mempool_size > tconf->max_mempool_size ) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_malloc1D(%d, %lu) releases memories in the memPool\n", asyncID, count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *devPtr = it->second;
					hipResult = hipFree(*devPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
            }
    		hipResult = hipMalloc(devPtr, count);
#ifdef _OPENARC_PROFILE_
            tconf->IDMallocCnt++;
            tconf->IDMallocSize += count;
            tconf->CIDMemorySize += count;
            if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
                tconf->MIDMemorySize = tconf->CIDMemorySize;
            }
#endif
            if (hipResult != hipSuccess) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_malloc1D(%d, %lu) releases memories in the memPool\n", asyncID, count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *devPtr = it->second;
					hipResult = hipFree(*devPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
    			hipResult = hipMalloc(devPtr, count);
            }
        }
        if( hipResult == hipSuccess ) {
            //Pin host memory
            if( tconf->prepin_host_memory == 1 ) {
                if( HI_pin_host_memory(hostPtr, (size_t) count, tconf->threadID) == HI_error ) {
#ifdef _OPENMP
                    fprintf(stderr, "[ERROR in HipDriver::HI_pin_host_memory()] Cannot pin host memory by tid: %d\n", omp_get_thread_num());
                    exit(1);
#else
                    fprintf(stderr, "[ERROR in HipDriver::HI_pin_host_memory()] Cannot pin host memory by tid: %d\n", tconf->threadID);
                    exit(1);
#endif
                }
            }
            HI_set_device_address(hostPtr, *devPtr, (size_t) count, asyncID, tconf->threadID);
            result = HI_success;
        } else {
            fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] HIP memory alloc failed with error %d\n", hipResult);
            exit(1);
        }
#else
        ///////////////////////////
        // VICTIM_CACHE_MODE = 2 //
        ///////////////////////////
        if(HI_get_device_address_from_victim_cache(hostPtr, devPtr, NULL, NULL, asyncID, tconf->threadID) == HI_success ) {
            result = HI_success;
            if( tconf->prepin_host_memory == 1 ) {
                inc_pinned_host_memory_counter(hostPtr);
            }
            HI_remove_device_address_from_victim_cache(hostPtr, asyncID, tconf->threadID);
        } else {
#ifdef _OPENARC_PROFILE_
            tconf->IDMallocCnt++;
            tconf->IDMallocSize += count;
            tconf->CIDMemorySize += count;
            if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
                tconf->MIDMemorySize = tconf->CIDMemorySize;
            }
#endif
    		hipResult = hipMalloc(devPtr, count);
            if (hipResult != hipSuccess) {
                if( tconf->prepin_host_memory == 1 ) {
                    unpin_host_memory_all(asyncID);
                } else {
                    release_freed_device_memory(asyncID, tconf->threadID);
                    HI_reset_victim_cache(asyncID);
                }
                //Try to allocate device memory again.
    			hipResult = hipMalloc(devPtr, count);
                if (hipResult != hipSuccess) {
                    if( tconf->prepin_host_memory == 1 ) {
                        unpin_host_memory_all();
                    } else {
                        release_freed_device_memory(tconf->threadID);
                        HI_reset_victim_cache_all();
                    }
                    //Try to allocate device memory again.
    				hipResult = hipMalloc(devPtr, count);
                }
            }
            if( hipResult == hipSuccess ) {
                //Pin host memory
                if( tconf->prepin_host_memory == 1 ) {
                    result = HI_pin_host_memory(hostPtr, (size_t) count, tconf->threadID);
                    if( result != HI_success ) {
                        unpin_host_memory_all(asyncID);
                        result = HI_pin_host_memory(hostPtr, (size_t) count, tconf->threadID);
                    }
                    if( result != HI_success ) {
                        unpin_host_memory_all();
                        result = HI_pin_host_memory(hostPtr, (size_t) count, tconf->threadID);
                    }
                    if( result != HI_success ) {
#ifdef _OPENMP
                        fprintf(stderr, "[ERROR in HipDriver::HI_pin_host_memory()] Cannot pin host memory by tid: %d\n", omp_get_thread_num());
                        exit(1);
#else
                        fprintf(stderr, "[ERROR in HipDriver::HI_pin_host_memory()] Cannot pin host memory by tid: %d\n", tconf->threadID);
                        exit(1);
#endif
                    }
                }
            }
        }
        if( hipResult == hipSuccess ) {
            HI_set_device_address(hostPtr, *devPtr, (size_t) count, asyncID, tconf->threadID);
            result = HI_success;
        } else {
            fprintf(stderr, "[ERROR in HipDriver::HI_malloc1D()] HIP memory alloc failed with error %d\n", hipResult);
            exit(1);
        }
#endif
    }

#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        HI_print_device_address_mapping_summary(tconf->threadID);
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_malloc1D(hostPtr = %lx, asyncID = %d, size = %lu, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, count, threadID);
    }
#endif
    return result;
}

HI_error_t HipDriver::HI_free( const void *hostPtr, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_free(%d)\n", asyncID);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    HI_error_t result = HI_success;
    void *devPtr;
    size_t size;
    //Check if the mapping exists. Free only if a mapping is found
    if( HI_get_device_address(hostPtr, &devPtr, NULL, &size, asyncID, tconf->threadID) != HI_error) {
        //If this method is called for unified memory, memory deallocation
        //is skipped; unified memory will be deallocatedd only by 
        //HI_free_unified().
        if( hostPtr != devPtr ) {
#if VICTIM_CACHE_MODE == 1
            //We do not free the device memory; instead put it in the memory pool 
            //and remove host-pointer-to-device-pointer mapping.
            memPool_t *memPool = memPoolMap[tconf->threadID];
            memPool->insert(std::pair<size_t, void *>(size, devPtr));
            current_mempool_size += size;
            HI_remove_device_address(hostPtr, asyncID, tconf->threadID);
            // Unpin host memory
            HI_unpin_host_memory(hostPtr, tconf->threadID);
#endif
#if VICTIM_CACHE_MODE == 2
            ///////////////////////////
            // VICTIM_CACHE_MODE = 2 //
            ///////////////////////////
            HI_remove_device_address(hostPtr, asyncID, tconf->threadID);
            HI_set_device_address_in_victim_cache(hostPtr, devPtr, size, asyncID, tconf->threadID);
           // Decrease pinned host memory counter
            if( tconf->prepin_host_memory == 1 ) {
                dec_pinned_host_memory_counter(hostPtr);
            } else {
                HI_unpin_host_memory(hostPtr, tconf->threadID);
            }
#endif
#if VICTIM_CACHE_MODE == 0
            hipError_t hipResult = hipSuccess;
            hipResult = hipFree(devPtr);
            if( hipResult == hipSuccess ) {
                HI_remove_device_address(hostPtr, asyncID, tconf->threadID);
                // Unpin host memory
                if( tconf->prepin_host_memory == 1 ) {
                    HI_unpin_host_memory(hostPtr, tconf->threadID);
                }
            } else {
                fprintf(stderr, "[ERROR in HipDriver::HI_free()] HIP memory free failed with error %d\n", hipResult);
                exit(1);
                result = HI_error;
            }
#ifdef _OPENARC_PROFILE_
            tconf->IDFreeCnt++;
            tconf->CIDMemorySize -= size;
#endif
#endif
        }
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_free(%d)\n", asyncID);
    }
#endif
    return result;
}

#if VICTIM_CACHE_MODE == 0

//malloc used for allocating temporary data.
//If the method is called for a pointer to existing memory, the existing memory
//will be freed before allocating new memory.
void HipDriver::HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempMalloc1D()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia ||
        devType == acc_device_radeon || devType == acc_device_current) {
        if( tempMallocSet.count(*tempPtr) > 0 ) {
            tempMallocSet.erase(*tempPtr);
            hipError_t hipResult = hipFree(*tempPtr);
            if(hipResult != hipSuccess) {
                fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D()] failed to free on HIP with error %d\n", hipResult);
                exit(1);
            }
#ifdef _OPENARC_PROFILE_
            tconf->IDFreeCnt++;
#endif
            if( tempMallocSizeMap.count(tconf->threadID) > 0 ) {
                sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
                if( tempMallocSize->count(*tempPtr) > 0 ) {
#ifdef _OPENARC_PROFILE_
                    tconf->CIDMemorySize -= (*tempMallocSize)[*tempPtr];
#endif
                    tempMallocSize->erase((const void *)*tempPtr);
                }
            }
        }
        hipError_t hipResult = hipMalloc(tempPtr, (size_t) count);
        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D()] failed to malloc on HIP with error %d\n", hipResult);
            exit(1);
        }
        tempMallocSet.insert(*tempPtr);
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
        (*tempMallocSize)[(const void *)*tempPtr] = count;
#ifdef _OPENARC_PROFILE_
        tconf->IDMallocCnt++;
        tconf->IDMallocSize += count;
        tconf->CIDMemorySize += count;
        if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
            tconf->MIDMemorySize = tconf->CIDMemorySize;
        }
#endif
    } else {
        if( tempMallocSet.count(*tempPtr) > 0 ) {
            tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->IHFreeCnt++;
#endif
        }
        *tempPtr = malloc(count);
        tempMallocSet.insert(*tempPtr);
#ifdef _OPENARC_PROFILE_
        tconf->IHMallocCnt++;
        tconf->IHMallocSize += count;
        tconf->CIHMemorySize += count;
        if( tconf->MIHMemorySize < tconf->CIHMemorySize ) {
            tconf->MIHMemorySize = tconf->CIHMemorySize;
        }
#endif
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempMalloc1D()\n");
    }
#endif
}

//malloc used for allocating temporary data.
void HipDriver::HI_tempMalloc1D_async( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempMalloc1D_async()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia ||
        devType == acc_device_radeon || devType == acc_device_current) {
        hipError_t hipResult = hipMalloc(tempPtr, (size_t) count);
        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D_async()] failed to malloc on HIP with error %d\n", hipResult);
            exit(1);
        }
        tempMallocSet.insert(*tempPtr);
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
        (*tempMallocSize)[(const void *)*tempPtr] = count;
#ifdef _OPENARC_PROFILE_
        tconf->IDMallocCnt++;
        tconf->IDMallocSize += count;
        tconf->CIDMemorySize += count;
        if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
            tconf->MIDMemorySize = tconf->CIDMemorySize;
        }
#endif
    } else {
        *tempPtr = malloc(count);
        tempMallocSet.insert(*tempPtr);
#ifdef _OPENARC_PROFILE_
        tconf->IHMallocCnt++;
        tconf->IHMallocSize += count;
        tconf->CIHMemorySize += count;
        if( tconf->MIHMemorySize < tconf->CIHMemorySize ) {
            tconf->MIHMemorySize = tconf->CIHMemorySize;
        }
#endif
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempMalloc1D_async()\n");
    }
#endif
}

//Used for de-allocating temporary data.
void HipDriver::HI_tempFree( void** tempPtr, acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia
    || devType == acc_device_radeon || devType == acc_device_current ) {
        if( *tempPtr != 0 ) {
            tempMallocSet.erase(*tempPtr);
            hipError_t hipResult = hipFree(*tempPtr);
            if(hipResult != hipSuccess) {
                fprintf(stderr, "[ERROR in HipDriver::HI_tempFree()] failed to free on HIP with error %d\n", hipResult);
                exit(1);
            }
#ifdef _OPENARC_PROFILE_
            tconf->IDFreeCnt++;
#endif
            if( tempMallocSizeMap.count(tconf->threadID) > 0 ) {
                sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
                if( tempMallocSize->count(*tempPtr) > 0 ) {
#ifdef _OPENARC_PROFILE_
                    tconf->CIDMemorySize -= (*tempMallocSize)[*tempPtr];
#endif
                    tempMallocSize->erase((const void *)*tempPtr);
                }
            }
        }
    } else {
        if( *tempPtr != 0 ) {
            tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
            if( tconf->prepin_host_memory == 1 ) {
                // Unpin host memory if already pinned.
                HI_unpin_host_memory(*tempPtr, tconf->threadID);
            }
#ifdef _OPENARC_PROFILE_
            tconf->IHFreeCnt++;
#endif
        }
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
    *tempPtr = 0;
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
#endif
}

#else

///////////////////////////
// VICTIM_CACHE_MODE > 0 //
///////////////////////////

//malloc used for allocating temporary data.
void HipDriver::HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempMalloc1D()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia ||
        devType == acc_device_radeon || devType == acc_device_current) {
        hipError_t hipResult = hipSuccess;
        memPool_t *memPool = memPoolMap[tconf->threadID];
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
        std::multimap<size_t, void *>::iterator it = memPool->find(count);
        if (it != memPool->end()) {
#ifdef _OPENARC_PROFILE_
            if( HI_openarcrt_verbosity > 2 ) {
                fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_tempMalloc1D(%lu) reuses memories in the memPool\n", count);
            }
#endif
            *tempPtr = it->second;
            memPool->erase(it);
            current_mempool_size -= count;
            (*tempMallocSize)[(const void *)*tempPtr] = count;
        } else {
            if( current_mempool_size > tconf->max_mempool_size ) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_malloc1D(%lu) releases memories in the memPool\n", count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *tempPtr = it->second;
                    hipResult = hipFree(*tempPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_ 
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
            }
            hipResult = hipMalloc(tempPtr, count);
            if (hipResult != hipSuccess) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_malloc1D(%lu) releases memories in the memPool\n", count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *tempPtr = it->second;
                    hipResult = hipFree(*tempPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_ 
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
                hipResult = hipMalloc(tempPtr, count);
            }
            if(hipResult == hipSuccess) {
                //New temporary device memory is allocated.
                (*tempMallocSize)[(const void *)*tempPtr] = count;
#ifdef _OPENARC_PROFILE_
                tconf->IDMallocCnt++;
                tconf->IDMallocSize += count;
                tconf->CIDMemorySize += count;
                if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
                    tconf->MIDMemorySize = tconf->CIDMemorySize;
                }
#endif
            } else {
                fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D()] failed to malloc on HIP with error %d\n", hipResult);
                exit(1);
            }
        }
    } else {
        if( tempMallocSet.count(*tempPtr) > 0 ) {
            tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->IHFreeCnt++;
#endif
        }
        *tempPtr = malloc(count);
        tempMallocSet.insert(*tempPtr);
#ifdef _OPENARC_PROFILE_
        tconf->IHMallocCnt++;
        tconf->IHMallocSize += count;
        tconf->CIHMemorySize += count;
        if( tconf->MIHMemorySize < tconf->CIHMemorySize ) {
            tconf->MIHMemorySize = tconf->CIHMemorySize;
        }
#endif
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempMalloc1D()\n");
    }
#endif
}

//malloc used for allocating temporary data.
void HipDriver::HI_tempMalloc1D_async( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempMalloc1D_async()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia ||
        devType == acc_device_radeon || devType == acc_device_current) {
        hipError_t hipResult = hipSuccess;
        memPool_t *memPool = memPoolMap[tconf->threadID];
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
        std::multimap<size_t, void *>::iterator it = memPool->find(count);
        if (it != memPool->end()) {
#ifdef _OPENARC_PROFILE_
            if( HI_openarcrt_verbosity > 2 ) {
                fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_tempMalloc1D_async(%lu) reuses memories in the memPool\n", count);
            }
#endif
            *tempPtr = it->second;
            memPool->erase(it);
            current_mempool_size -= count;
            (*tempMallocSize)[(const void *)*tempPtr] = count;
        } else {
            if( current_mempool_size > tconf->max_mempool_size ) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_tempMalloc1D_async(%lu) releases memories in the memPool\n", count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *tempPtr = it->second;
                    hipResult = hipFree(*tempPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_ 
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D_async()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
            }
            hipResult = hipMalloc(tempPtr, count);
            if (hipResult != hipSuccess) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tHipDriver::HI_tempMalloc1D_async(%lu) releases memories in the memPool\n", count);
                }
#endif
                for (it = memPool->begin(); it != memPool->end(); ++it) {
                    *tempPtr = it->second;
                    hipResult = hipFree(*tempPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_ 
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
                    if(hipResult != hipSuccess) {
                        fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D_async()] failed to free on HIP with error %d\n", hipResult);
                    }
                }
                memPool->clear();
                hipResult = hipMalloc(tempPtr, count);
            }
            if(hipResult == hipSuccess) {
                //New temporary device memory is allocated.
                (*tempMallocSize)[(const void *)*tempPtr] = count;
#ifdef _OPENARC_PROFILE_
                tconf->IDMallocCnt++;
                tconf->IDMallocSize += count;
                tconf->CIDMemorySize += count;
                if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
                    tconf->MIDMemorySize = tconf->CIDMemorySize;
                }
#endif
            } else {
                fprintf(stderr, "[ERROR in HipDriver::HI_tempMalloc1D_async()] failed to malloc on HIP with error %d\n", hipResult);
                exit(1);
            }
        }
    } else {
        if( tempMallocSet.count(*tempPtr) > 0 ) {
            tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
#ifdef _OPENARC_PROFILE_
            tconf->IHFreeCnt++;
#endif
        }
        *tempPtr = malloc(count);
        tempMallocSet.insert(*tempPtr);
#ifdef _OPENARC_PROFILE_
        tconf->IHMallocCnt++;
        tconf->IHMallocSize += count;
        tconf->CIHMemorySize += count;
        if( tconf->MIHMemorySize < tconf->CIHMemorySize ) {
            tconf->MIHMemorySize = tconf->CIHMemorySize;
        }
#endif
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempMalloc1D_async()\n");
    }
#endif
}

//Used for de-allocating temporary data.
void HipDriver::HI_tempFree( void** tempPtr, acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_tempMalloc);
#else
#ifdef _OPENMP
    #pragma omp critical (tempMalloc_critical)
#endif
#endif
    {
    if( devType == acc_device_gpu || devType == acc_device_nvidia
    || devType == acc_device_radeon || devType == acc_device_current ) {
        if( *tempPtr != 0 ) {
            //We do not free the device memory; instead put it in the memory pool 
            memPool_t *memPool = memPoolMap[tconf->threadID];
            sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
            if( tempMallocSize->count((const void *)*tempPtr) == 0 ) {
                if( tempMallocSet.count(*tempPtr) > 0 ) {
                    tempMallocSet.erase(*tempPtr);
                    free(*tempPtr);
                    if( tconf->prepin_host_memory == 1 ) {
                        // Unpin host memory if already pinned.
                        HI_unpin_host_memory(*tempPtr, tconf->threadID);
                    }
                } else {
                    fprintf(stderr, "[OPENARCRT-WARNING in HipDriver::HI_tempFree(devType = %d, thread ID = %d)] no tempMallocSize mapping found for tempPtr (%lx)\n", devType, threadID, (long unsigned int)(*tempPtr));
                }
            } else {
                size_t size = tempMallocSize->at((const void *)*tempPtr);
                memPool->insert(std::pair<size_t, void *>(size, *tempPtr));
                current_mempool_size += size;
                tempMallocSize->erase((const void *)*tempPtr);
            }
        }
    } else {
        if( *tempPtr != 0 ) {
            tempMallocSet.erase(*tempPtr);
            free(*tempPtr);
            if( tconf->prepin_host_memory == 1 ) {
                // Unpin host memory if already pinned.
                HI_unpin_host_memory(*tempPtr, tconf->threadID);
            }
#ifdef _OPENARC_PROFILE_
            tconf->IHFreeCnt++;
#endif
        }
    }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_tempMalloc);
#endif
    *tempPtr = 0;
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
#endif
}

#endif


//////////////////////
// Kernel Execution //
//////////////////////


HI_error_t HipDriver::HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_memcpy(%lu)\n", count);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);

    hipError_t err = hipSuccess;
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
	if( dst != src ) {
    	switch( kind ) {
        	case HI_MemcpyHostToHost:       memcpy(dst, src, count);                break;
    		case HI_MemcpyHostToDevice:     err = hipMemcpyHtoD(dst, (void*) src, count);   break;
    		case HI_MemcpyDeviceToHost:     err = hipMemcpyDtoH(dst, (void*) src, count);   break;
    		case HI_MemcpyDeviceToDevice:   err = hipMemcpyDtoD(dst, (void*) src, count);   break;
    	}
	}
#ifdef _OPENARC_PROFILE_
    if( dst != src ) {
        if( kind == HI_MemcpyHostToDevice ) {
            tconf->H2DMemTrCnt++;
            tconf->H2DMemTrSize += count;
        } else if( kind == HI_MemcpyDeviceToHost ) {
            tconf->D2HMemTrCnt++;
            tconf->D2HMemTrSize += count;
        } else if( kind == HI_MemcpyDeviceToDevice ) {
            tconf->D2DMemTrCnt++;
            tconf->D2DMemTrSize += count;
        } else {
            tconf->H2HMemTrCnt++;
            tconf->H2HMemTrSize += count;
        }
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( err == hipSuccess ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_memcpy(%lu)\n", count);
    }
#endif
        return HI_success;
    } else {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HipDriver::HI_memcpy()] Memcpy failed with error %d in tid %d\n", err, omp_get_thread_num());
        exit(1);
#else
        fprintf(stderr, "[ERROR in HipDriver::HI_memcpy()] Memcpy failed with error %d in tid %d\n", err, 0);
        exit(1);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_memcpy(%lu)\n", count);
    }
#endif
        return HI_error;
    }
}

HI_error_t HipDriver::HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_memcpy_async(%d, %lu)\n", async, count);
    }
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
	 HostConf_t * tconf = getHostConf(threadID);
    HI_wait_for_events(async, num_waits, waits, tconf->threadID);

    hipError_t hipResult = hipSuccess;
    hipStream_t stream = getQueue(async, tconf->threadID);
    hipEvent_t event = getEvent(async, tconf->threadID);
    const void * baseHostPtr = 0;

    if( dst != src ) {
        switch( kind ) {
        case HI_MemcpyHostToHost: {
            memcpy(dst, src, count);
            break;
        }
        case HI_MemcpyHostToDevice: {
            baseHostPtr = HI_get_base_address_of_host_memory(src, async, tconf->threadID);
            if( baseHostPtr == 0 ) {
                baseHostPtr = src;
            }
            pin_host_memory_if_unpinned(baseHostPtr, count);
            hipResult = hipMemcpyHtoDAsync( dst, (void *)src, count, stream);
            break;
        }
        case HI_MemcpyDeviceToHost: {
            baseHostPtr = HI_get_base_address_of_host_memory((const void *)dst, async, tconf->threadID);
            if( baseHostPtr == 0 ) {
                baseHostPtr = (const void *)dst;
            }
            pin_host_memory_if_unpinned(baseHostPtr, count);
            hipResult = hipMemcpyDtoHAsync(dst, (void *)src, count, stream);
            break;
        }
        case HI_MemcpyDeviceToDevice: {
            hipResult = hipMemcpyDtoDAsync(dst, (void *)src, count, stream);
            break;
        }
        }
    }

    hipEventRecord(event, stream);
#ifdef _OPENARC_PROFILE_
    if( dst != src ) {
        if( kind == HI_MemcpyHostToDevice ) {
            tconf->H2DMemTrCnt++;
            tconf->H2DMemTrSize += count;
        } else if( kind == HI_MemcpyDeviceToHost ) {
            tconf->D2HMemTrCnt++;
            tconf->D2HMemTrSize += count;
        } else if( kind == HI_MemcpyDeviceToDevice ) {
            tconf->D2DMemTrCnt++;
            tconf->D2DMemTrSize += count;
        } else {
            tconf->H2HMemTrCnt++;
            tconf->H2HMemTrSize += count;
        }
    }
    tconf->totalMemTrTime += HI_get_localtime() - ltime;
#endif
    if( hipResult == hipSuccess ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_memcpy_async(%d, %lu)\n", async, count);
    }
#endif
        return HI_success;
    } else {
#ifdef _OPENMP
        fprintf(stderr, "[ERROR in HipDriver::HI_memcpy_async()] Memcpy failed with error %d in tid %d with asyncId %d\n", hipResult, omp_get_thread_num(), async);
        exit(1);
#else
        fprintf(stderr, "[ERROR in HipDriver::HI_memcpy_async()] Memcpy failed with error %d in tid %d with asyncId %d\n", hipResult, 0, async);
        exit(1);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_memcpy_async(%d, %lu)\n", async, count);
    }
#endif
        return HI_error;
    }
    return HI_success;
}

HI_error_t HipDriver::HI_register_kernel_numargs(std::string kernel_name, int num_args, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_register_kernel_numargs()\n");
    }
#endif
    HostConf_t *tconf = getHostConf(threadID);
	kernelParams_t *kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
	if( kernelParams->num_args == 0 ) {
		if( num_args > 0 ) {
			kernelParams->num_args = num_args;
    		kernelParams->kernelParams = (void**)malloc(sizeof(void*) * num_args);
		} else {
        	fprintf(stderr, "[ERROR in HipDriver::HI_register_kernel_numargs(%s, %d)] num_args should be greater than zero.\n",kernel_name.c_str(), num_args);
			exit(1);
		}
	}
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_register_kernel_numargs()\n");
    }
#endif
    return HI_success;
}

HI_error_t HipDriver::HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_register_kernel_arg()\n");
    }
#endif
    HostConf_t *tconf = getHostConf(threadID);
	kernelParams_t * kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
	if( kernelParams->num_args > arg_index ) {
		*(kernelParams->kernelParams + arg_index) = arg_value;
	} else {
		fprintf(stderr, "[ERROR in HipDriver::HI_register_kernel_arg()] Kernel %s is registered to have %d arguments, but the current argument index %d is out of the bound.\n",kernel_name.c_str(), kernelParams->num_args, arg_index);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_register_kernel_arg()\n");
    }
#endif
    return HI_success;
}

HI_error_t HipDriver::HI_kernel_call(std::string kernel_name, size_t gridSize[3], size_t blockSize[3], int async, int num_waits, int *waits, int threadID) {
    const char* c_kernel_name = kernel_name.c_str();
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_kernel_call(%s, %d)\n",c_kernel_name, async);
    }
#endif
    if( (gridSize[0] > maxGridX) || (gridSize[1] > maxGridY) || (gridSize[2] > maxGridZ) ) {
        fprintf(stderr, "[ERROR in HipDriver::HI_kernel_call()] Kernel [%s] Launch FAIL due to too large Grid configuration (%lu, %lu, %lu); exit!\n", c_kernel_name, gridSize[2], gridSize[1], gridSize[0]);
        exit(1);
    }
    if( (blockSize[0] > maxBlockX) || (blockSize[1] > maxBlockY) || (blockSize[2] > maxBlockZ) || (blockSize[0]*blockSize[1]*blockSize[2] > maxNumThreadsPerBlock) ) {
        fprintf(stderr, "[ERROR in HipDriver::HI_kernel_call()] Kernel [%s] Launch FAIL due to too large threadBlock configuration (%lu, %lu, %lu); exit!\n",c_kernel_name, blockSize[2], blockSize[1], blockSize[0]);
        exit(1);
    }
    HostConf_t *tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    HI_wait_for_events(async, num_waits, waits, tconf->threadID);

    hipError_t err;
    hipStream_t stream = getQueue(async, tconf->threadID);
    hipEvent_t event = getEvent(async, tconf->threadID);
    if(async != DEFAULT_QUEUE+tconf->asyncID_offset) {
    	err = hipModuleLaunchKernel(tconf->kernelsMap.at(this).at(kernel_name), gridSize[0], gridSize[1], gridSize[2], blockSize[0], blockSize[1], blockSize[2], 0, stream, (tconf->kernelArgsMap.at(this).at(kernel_name))->kernelParams, NULL);

        hipEventRecord(event, stream);

    } else {
    	err = hipModuleLaunchKernel(tconf->kernelsMap.at(this).at(kernel_name), gridSize[0], gridSize[1], gridSize[2], blockSize[0], blockSize[1], blockSize[2], 0, 0, (tconf->kernelArgsMap.at(this).at(kernel_name))->kernelParams, NULL);
    }
    if (err != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_kernel_call(%s)] Kernel Launch FAIL with error %d\n",c_kernel_name, err);
    	fprintf(stderr, "\tgrid configuration: %lu %lu %lu\n", gridSize[2], gridSize[1], gridSize[0]);
    	fprintf(stderr, "\tthread block configuration: %lu %lu %lu\n", blockSize[2], blockSize[1], blockSize[0]);
        exit(1);
        return HI_error;
    }

#ifdef _OPENARC_PROFILE_
    if(tconf->KernelCNTMap.count(kernel_name) == 0) {
        tconf->KernelCNTMap[kernel_name] = 0;
    }
    tconf->KernelCNTMap[kernel_name] += 1;
    if(tconf->KernelTimingMap.count(kernel_name) == 0) {
        tconf->KernelTimingMap[kernel_name] = 0.0;
    }
    err = hipStreamSynchronize(stream);
    err = hipStreamSynchronize(0);
    tconf->KernelTimingMap[kernel_name] += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_kernel_call(%s, %d)\n",c_kernel_name, async);
    }
#endif

    return HI_success;
}


HI_error_t HipDriver::HI_synchronize( int forcedSync, int threadID ) {
    //err = hipDeviceSynchronize();
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_synchronize(%d)\n", forcedSync);
    }
#endif
    if( (forcedSync != 0) || (unifiedMemSupported == 1) ) {
        HostConf_t * tconf = getHostConf(threadID);
        hipStream_t stream = getQueue(DEFAULT_QUEUE+tconf->asyncID_offset, tconf->threadID);
    	hipError_t err;
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_synchronize()] DEFAULT_QUEUE Context Synchronization FAIL with error %d \n", err);
            exit(1);
            return HI_error;
        }
        if( stream != 0 ) {
            err = hipStreamSynchronize(0);
            if (err != hipSuccess) {
                fprintf(stderr, "[ERROR in HipDriver::HI_synchronize()] Defalut Context Synchronization FAIL with error %d \n", err);
                exit(1);
                return HI_error;
            }
        }
    }

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_synchronize(%d)\n", forcedSync);
    }
#endif
    return HI_success;
}


void HipDriver::HI_set_async(int asyncId, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_set_async(%d)\n", asyncId);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    int thread_id = tconf->threadID;
#ifdef _THREAD_SAFETY
        pthread_mutex_lock(&mutex_set_async);
#else
#ifdef _OPENMP
    #pragma omp critical (HI_set_async_critical)
#endif
#endif
    {
        asyncId += 2;
        std::map<int, hipStream_t>::iterator it= queueMap.find(asyncId);

        if(it == queueMap.end()) {
            hipStream_t str;
#ifdef USE_BLOCKING_STREAMS
            hipStreamCreateWithFlags(&str, hipStreamDefault);
#else
            hipStreamCreateWithFlags(&str, hipStreamNonBlocking);
#endif
            queueMap[asyncId] = str;
        }

        std::map<int, std::map<int, hipEvent_t> >::iterator threadIt;
        threadIt = threadQueueEventMap.find(thread_id);

        //threadQueueEventMap is empty for this thread
        if(threadIt == threadQueueEventMap.end()) {
            std::map<int, hipEvent_t> newMap;
            hipEvent_t ev;
            hipEventCreateWithFlags(&ev, hipEventDefault);
            newMap[asyncId] = ev;
            threadQueueEventMap[thread_id] = newMap;
        } else {
            //threadQueueEventMap does not have an entry for this stream
            //std::map<int, hipEvent_t> evMap = threadIt->second;
            if(threadIt->second.find(asyncId) == threadIt->second.end()) {
                hipEvent_t ev;
                hipEventCreateWithFlags(&ev, hipEventDefault);
                threadIt->second[asyncId] = ev;
                //threadIt->second = evMap;
            }
        }
    }
#ifdef _THREAD_SAFETY
        pthread_mutex_unlock(&mutex_set_async);
#endif
#ifndef USE_BLOCKING_STREAMS
    if( unifiedMemSupported == 0 ) {
        //We need explicit synchronization here for the default queue 
        //since HI_synchronize() does not explicitly synchronize if 
        //unified memory is not used.
        hipStream_t stream = getQueue(DEFAULT_QUEUE+tconf->asyncID_offset, tconf->threadID);
        hipError_t err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_set_async()] DEFAULT_QUEUE Context Synchronization FAIL with error %d \n", err);
            exit(1);
        }
        if( stream != 0 ) {
            err = hipStreamSynchronize(0);
            if (err != hipSuccess) {
                fprintf(stderr, "[ERROR in HipDriver::HI_set_async()] Default Context Synchronization FAIL with error %d \n", err);
                exit(1);
            }
        }
    }
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_set_async(%d)\n", asyncId-2);
    }
#endif
}

void HipDriver::HI_set_context(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_set_context()\n");
    }    
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_set_context()\n");
    }    
#endif
}

void HipDriver::HI_wait(int arg, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait(%d)\n", arg);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent(arg, tconf->threadID);

    hipError_t hipResult = hipEventSynchronize(event);

    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_wait()] failed wait on HIP queue %d with error %d\n", arg, hipResult);
        exit(1);
    }

    HI_postponed_free(arg, tconf->threadID);
    HI_postponed_tempFree(arg, tconf->acc_device_type_var, tconf->threadID);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait(%d)\n", arg);
    }
#endif
}

void HipDriver::HI_wait_ifpresent(int arg, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_ifpresent(%d)\n", arg);
    }    
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent_ifpresent(arg, tconf->threadID);
    if( event != NULL ) {

        hipError_t hipResult = hipEventSynchronize(event);

        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_wait_ifpresent()] failed wait on HIP queue %d with error %d\n", arg, hipResult);
            exit(1);
        }    

        HI_postponed_free(arg, tconf->threadID);
        HI_postponed_tempFree(arg, tconf->acc_device_type_var, tconf->threadID);
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_ifpresent(%d)\n", arg);
    }
#endif
}

//[DEBUG] Below implementation is inefficient.
void HipDriver::HI_wait_async(int arg, int async, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_async(%d, %d)\n", arg, async);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent(arg, tconf->threadID);
    hipEvent_t event2 = getEvent(async, tconf->threadID);

    hipError_t hipResult = hipEventSynchronize(event);

    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_wait_async()] failed wait on HIP queue %d with error %d\n", arg, hipResult);
        exit(1);
    }

    HI_postponed_free(arg, tconf->threadID);
    HI_postponed_tempFree(arg, tconf->acc_device_type_var, tconf->threadID);

    hipResult = hipEventSynchronize(event2);

    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_wait_async()] failed wait on HIP queue %d with error %d\n", async, hipResult);
        exit(1);
    }

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_async(%d, %d)\n", arg, async);
    }
#endif
}

void HipDriver::HI_wait_async_ifpresent(int arg, int async, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_async_ifpresent(%d, %d)\n", arg, async);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent_ifpresent(arg, tconf->threadID);
    hipEvent_t event2 = getEvent_ifpresent(async, tconf->threadID);
    if( (event != NULL) && (event2 != NULL) ) {

        hipError_t hipResult = hipEventSynchronize(event);

        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_wait_async_ifpresent()] failed wait on HIP queue %d with error %d\n", arg, hipResult);
            exit(1);
        }

        HI_postponed_free(arg, tconf->threadID);
        HI_postponed_tempFree(arg, tconf->acc_device_type_var, tconf->threadID);

        hipResult = hipEventSynchronize(event2);

        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_wait_async_ifpresent()] failed wait on HIP queue %d with error %d\n", async, hipResult);
            exit(1);
        }
    }

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_async_ifpresent(%d, %d)\n", arg, async);
    }
#endif
}

void HipDriver::HI_wait_all(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_all()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    eventmap_hip_t *eventMap = &threadQueueEventMap.at(tconf->threadID);
    hipError_t hipResult;

    for(eventmap_hip_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        hipResult = hipEventSynchronize(it->second);
        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_wait_all()] failed wait on HIP queue %d with error %d\n", it->first, hipResult);
            exit(1);
        }
        HI_postponed_free(it->first-2, tconf->threadID);
        HI_postponed_tempFree(it->first-2, tconf->acc_device_type_var, tconf->threadID);
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_all()\n");
    }
#endif
}

//[DEBUG] Below implementation is inefficient.
void HipDriver::HI_wait_all_async(int async, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_all_async(%d)\n", async);
    }    
#endif
    HostConf_t * tconf = getHostConf(threadID);
    eventmap_hip_t *eventMap = &threadQueueEventMap.at(tconf->threadID);
    hipError_t hipResult;

    for(eventmap_hip_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        hipResult = hipEventSynchronize(it->second);
        if(hipResult != hipSuccess) {
            fprintf(stderr, "[ERROR in HipDriver::HI_wait_all_async()] failed wait on HIP queue %d with error %d\n", it->first, hipResult);
            exit(1);
        }    
        HI_postponed_free(it->first-2, tconf->threadID);
        HI_postponed_tempFree(it->first-2, tconf->acc_device_type_var, tconf->threadID);
    }    

    hipEvent_t event2 = getEvent(async, tconf->threadID);
    hipResult = hipEventSynchronize(event2);

    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_wait_all_async()] failed wait on HIP queue %d with error %d\n", async, hipResult);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_all_async(%d)\n", async);
    }
#endif
}

int HipDriver::HI_async_test(int asyncId, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_async_test(%d)\n", asyncId);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent(asyncId, tconf->threadID);

    hipError_t hipResult = hipEventQuery(event);

    if(hipResult != hipSuccess) {
        //fprintf(stderr, "in HipDriver::HI_async_test()] stream %d code %d\n", asyncId, hipResult);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_async_test(%d)\n", asyncId);
    }
#endif
        return 0;
    }

    HI_postponed_free(asyncId, tconf->threadID);
    HI_postponed_tempFree(asyncId, tconf->acc_device_type_var, tconf->threadID);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_async_test(%d)\n", asyncId);
    }
#endif
    return 1;
}

int HipDriver::HI_async_test_ifpresent(int asyncId, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_async_test_ifpresent(%d)\n", asyncId);
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    hipEvent_t event = getEvent_ifpresent(asyncId, tconf->threadID);
    if( event != NULL ) {

        hipError_t hipResult = hipEventQuery(event);

        if(hipResult != hipSuccess) {
            //fprintf(stderr, "in HipDriver::HI_async_test_ifpresent()] stream %d code %d\n", asyncId, hipResult);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_async_test_ifpresent(%d)\n", asyncId);
    }
#endif
            return 0;
        }

        HI_postponed_free(asyncId, tconf->threadID);
        HI_postponed_tempFree(asyncId, tconf->acc_device_type_var, tconf->threadID);
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_async_test_ifpresent(%d)\n", asyncId);
    }
#endif
    return 1;
}

int HipDriver::HI_async_test_all(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_async_test_all()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
    eventmap_hip_t *eventMap = &threadQueueEventMap.at(tconf->threadID);
    hipError_t hipResult;

    std::set<int> queuesChecked;

    for(eventmap_hip_t::iterator it = eventMap->begin(); it != eventMap->end(); ++it) {
        hipResult = hipEventQuery(it->second);
        if(hipResult != hipSuccess) {
            return 0;
        }
        queuesChecked.insert(it->first);
    }

    //release the waiting frees
    std::set<int>::iterator it;
    for (it=queuesChecked.begin(); it!=queuesChecked.end(); ++it) {
        HI_postponed_free(*it, tconf->threadID);
        HI_postponed_tempFree(*it, tconf->acc_device_type_var, tconf->threadID);
    }

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_async_test_all()\n");
    }
#endif
    return 1;
}

void HipDriver::HI_wait_for_events(int async, int num_waits, int* waits, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_wait_for_events()\n");
    }
#endif
    hipError_t hipResult = hipSuccess;
    HostConf_t * tconf = getHostConf(threadID);
    if (num_waits > 0 && async != DEFAULT_QUEUE+tconf->asyncID_offset) {
        hipStream_t stream = getQueue(async, tconf->threadID);
        for (int i = 0; i < num_waits; i++) {
            if (waits[i] == async) continue;
            hipEvent_t event = getEvent(waits[i], tconf->threadID);
            hipResult = hipStreamWaitEvent(stream, event, 0);
            if(hipResult != hipSuccess) {
                fprintf(stderr, "[ERROR in HipDriver::HI_wait_for_events()] failed to call hipStreamWaitEvent %d\n", hipResult);
                exit(1);
            }
        }
    }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_wait_for_events()\n");
    }
#endif
}

void HipDriver::HI_malloc(void **devPtr, size_t size, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_malloc()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
    hipError_t hipResult = hipMalloc(devPtr, (size_t) size);
    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_malloc()] failed to malloc on HIP with error %d\n", hipResult);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
    tconf->IDMallocCnt++;
    tconf->IDMallocSize += size;
    tconf->CIDMemorySize += size;
    if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
        tconf->MIDMemorySize = tconf->CIDMemorySize;
    }
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_malloc()\n");
    }
#endif
}

void HipDriver::HI_free(void *devPtr, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter HipDriver::HI_free()\n");
    }
#endif
    HostConf_t * tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

    hipError_t hipResult = hipSuccess;
    void *devPtr2;
    size_t memSize = 0;
//    if( (HI_get_device_address(devPtr, &devPtr2, DEFAULT_QUEUE+tconf->asyncID_offset, tconf->threadID) == HI_error) ||
    if( (HI_get_device_address(devPtr, &devPtr2, NULL, &memSize, DEFAULT_QUEUE+tconf->asyncID_offset, tconf->threadID) == HI_error) ||
        (devPtr != devPtr2) ) {
        //Free device memory if it is not on unified memory.
        hipResult = hipFree(devPtr);
#ifdef _OPENARC_PROFILE_
        tconf->IDFreeCnt++;
        tconf->CIDMemorySize -= memSize;
#endif
    }

    if(hipResult != hipSuccess) {
        fprintf(stderr, "[ERROR in HipDriver::HI_free()] failed to free on HIP with error %d\n", hipResult);
        exit(1);
    }
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit HipDriver::HI_free()\n");
    }
#endif
}

//Resume

HI_error_t HipDriver::HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy2D_asyncS(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

// Experimental API to support unified memory //
HI_error_t HipDriver::HI_malloc1D_unified(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_free_unified( const void *hostPtr, int asyncID, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_memcpy_const_async(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

HI_error_t HipDriver::HI_present_or_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

    return HI_success;
}

void HipDriver::HI_waitS1(int arg, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void HipDriver::HI_waitS2(int arg, int threadID) {
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}


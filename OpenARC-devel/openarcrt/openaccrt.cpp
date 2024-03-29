#include <stdlib.h>
#include <limits.h>
#include "openacc.h"
#include "openaccrt_ext.h"
////////////////////////////////////////
// Functions used for resilience test //
////////////////////////////////////////
#include "resilience.cpp"

#ifdef _THREAD_SAFETY
pthread_mutex_t mutex_HI_init;
pthread_mutex_t mutex_HI_hostinit;
pthread_mutex_t mutex_HI_kernelnames;
pthread_mutex_t mutex_pin_host_memory;
pthread_mutex_t mutex_victim_cache;
pthread_mutex_t mutex_tempMalloc;
pthread_mutex_t mutex_set_async;
pthread_mutex_t mutex_set_device_num;
pthread_mutex_t mutex_clContext;
pthread_mutex_t mutex_presentTable;
#endif

static const char *omp_num_threads_env = "OMP_NUM_THREADS";
static const char *acc_device_type_env = "ACC_DEVICE_TYPE";
static const char *acc_device_num_env = "ACC_DEVICE_NUM";
static const char *omp_device_num_env = "OMP_DEFAULT_DEVICE";
static const char *outputType = "OPENARC_ARCH";
static const char *openarcrt_verbosity_env = "OPENARCRT_VERBOSITY";
static const char *openarcrt_max_mempool_size_env = "OPENARCRT_MAXMEMPOOLSIZE";
static const char *openarcrt_unifiedmemory_env = "OPENARCRT_UNIFIEDMEM";
static const char *openarcrt_prepinhostmemory_env = "OPENARCRT_PREPINHOSTMEM";
static const char *openarcrt_memoryalignment_env = "OPENARCRT_MEMORYALIGNMENT";
static const char *NVIDIA = "NVIDIA";
static const char *RADEON = "RADEON";
static const char *XEONPHI = "XEONPHI";
static const char *ALTERA = "ALTERA";

devmap_t HostConf::devMap;
//HI_kernelnames contain all the kernels used by any host thread, which does not distinguish target device types.
std::set<std::string> HostConf::HI_kernelnames;

int HI_hostinit_done = 0;
int HI_openarcrt_verbosity = 0;
int HI_openarcrt_memoryalignment = 1;
long HI_max_mempool_size = 0;
int HI_use_unifiedmemory = 0;
int HI_prepin_host_memory = 1;
int HI_num_hostthreads = 1;

//Return a local time in seconds.
double HI_get_localtime () {
    struct timeval time;
    gettimeofday(&time, 0);
    return time.tv_sec + time.tv_usec / 1000000.0;
}

//std::vector<HostConf_t *> hostConfList;
std::map<int, HostConf_t *> hostConfMap;

//DEBUG: for now, resilience test is enabled by default
#define _OPENARC_RESILIENCE_


////////////////////////
// Runtime init/reset //
////////////////////////
//[FIXME] if default device type is different from the one passed to acc_init(),
//the default device type should be updated to the passed value.
//==> tconf->acc_device_type_var is updated in acc_init().
//This function assumes that threadID argument has a valid value (>=0).
void HI_hostinit(int threadID) {
	int openarcrt_verbosity = 0;
    char * envVar;
    envVar = getenv(openarcrt_verbosity_env);
	if( envVar != NULL ) {
		openarcrt_verbosity = atoi(envVar);
		if( openarcrt_verbosity > 0 ) {
			HI_openarcrt_verbosity = openarcrt_verbosity;
		}
	}
    envVar = getenv(openarcrt_memoryalignment_env);
	if( envVar != NULL ) {
		HI_openarcrt_memoryalignment = atoi(envVar);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_hostinit(thread ID = %d)\n", threadID);
	}
#endif
    if( HI_hostinit_done == 0 ) {
#ifdef _THREAD_SAFETY
		pthread_mutex_init(&mutex_HI_init, NULL);
		pthread_mutex_init(&mutex_HI_hostinit, NULL);
		pthread_mutex_init(&mutex_HI_kernelnames, NULL);
		pthread_mutex_init(&mutex_pin_host_memory, NULL);
		pthread_mutex_init(&mutex_victim_cache, NULL);
		pthread_mutex_init(&mutex_tempMalloc, NULL);
		pthread_mutex_init(&mutex_set_async, NULL);
		pthread_mutex_init(&mutex_set_device_num, NULL);
		pthread_mutex_init(&mutex_clContext, NULL);
#endif
    	envVar = getenv(openarcrt_max_mempool_size_env);
		if( envVar != NULL ) {
			HI_max_mempool_size = strtol(envVar, NULL, 10);
		} else {
			HI_max_mempool_size = 0;
		}
    	envVar = getenv(openarcrt_unifiedmemory_env);
		if( envVar != NULL ) {
			HI_use_unifiedmemory = atoi(envVar);
		} else {
			HI_use_unifiedmemory = 0;
		}
    	envVar = getenv(openarcrt_prepinhostmemory_env);
		if( envVar != NULL ) {
			HI_prepin_host_memory = atoi(envVar);
		} else {
			//Default behavior is changed to no-prepinning.
			HI_prepin_host_memory = 0;
		}
    	HI_hostinit_done = 1;
		if( HI_prepin_host_memory == 1 ) {
#ifdef _OPENARC_PROFILE_
			fprintf(stderr, "[OPENARCRT-INFO] Host memory will be prepinned for fast memory transfers; to disable this, set environment variable %s to 0 (excessive prepinning may slow or crash the program.)\n", openarcrt_prepinhostmemory_env);
#endif
		}
	}
    if( hostConfMap.count(threadID) == 0 ) {
		HostConf_t * tconf = new HostConf_t;
		tconf->threadID=threadID;
		tconf->setDefaultDevNum();
		tconf->setDefaultDevice();
		tconf->createHostTables();
		//tconf->initKernelNames();
		if( HI_max_mempool_size > 0 ) {
			tconf->max_mempool_size = HI_max_mempool_size;
		}
		tconf->use_unifiedmemory = HI_use_unifiedmemory;
		tconf->prepin_host_memory = HI_prepin_host_memory;
		//tconf->HI_init_done=1;
		tconf->asyncID_offset=threadID*MAX_NUM_QUEUES_PER_THREAD;
#ifdef _THREAD_SAFETY
		pthread_mutex_lock(&mutex_HI_hostinit);
#else
#ifdef _OPENMP
        #pragma omp critical (HI_hostinit_critical)
#endif
#endif
        {
            hostConfMap[threadID] = tconf;
        }
		//HI_num_hostthreads = hostConfMap.size();
#ifdef _THREAD_SAFETY
		pthread_mutex_unlock(&mutex_HI_hostinit);
#endif
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_hostinit(thread ID = %d)\n", threadID);
	}
#endif
}

//Get the initial host configuration, or create it if nox existing.
//This function is called only in acc_init() function.
HostConf_t * getInitHostConf(int threadID) {
    HostConf_t * tconf = NULL;
	if( threadID == NO_THREAD_ID ) {
    	threadID = get_thread_id();
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters getInitHostConf()\n", threadID);
	}
#endif
    if( hostConfMap.count(threadID) > 0 ) {
        tconf = hostConfMap.at(threadID);
    }

    if( tconf == NULL ) {
        HI_hostinit(threadID);
        tconf = hostConfMap.at(threadID);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in getInitHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", threadID);
            exit(1);
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits getInitHostConf()\n", threadID);
	}
#endif
    return tconf;
}

//Similar to getInitHostConf(), but this also invokes HI_init() if not done.
//This function is called in most of existing OpenARC runtime APIs.
HostConf_t * getHostConf(int threadID) {
    HostConf_t * tconf = NULL;
	if( threadID == NO_THREAD_ID ) {
    	threadID = get_thread_id();
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters getHostConf()\n", threadID);
	}
#endif
    if( hostConfMap.count(threadID) > 0 ) {
        tconf = hostConfMap.at(threadID);
    }

    if( tconf == NULL ) {
        HI_hostinit(threadID);
        tconf = hostConfMap.at(threadID);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in getHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", threadID);
            exit(1);
        }
    }
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1; //This should execute first.
		tconf->HI_init(DEVICE_NUM_UNDEFINED);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits getHostConf()\n", threadID);
	}
#endif
    return tconf;
}

//Similar to getHostConf(), but this also sets device number.
//This function is called only in acc_set_device_num() function.
HostConf_t * setNGetHostConf(int devNum, int threadID) {
    HostConf_t * tconf = NULL;
	if( threadID == NO_THREAD_ID ) {
    	threadID = get_thread_id();
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d enters setNGetHostConf(devNum = %d)\n", threadID, devNum);
	}
#endif
    if( hostConfMap.count(threadID) > 0 ) {
        tconf = hostConfMap.at(threadID);
    }

    if( tconf == NULL ) {
        HI_hostinit(threadID);
        tconf = hostConfMap.at(threadID);
        if( tconf == NULL ) {
            fprintf(stderr, "[ERROR in setNGetHostConf] No host configuration exists for the current host thread (thread ID: %d);\n", threadID);
            exit(1);
        }
    }
	if( tconf->HI_init_done == 0 ) {
		tconf->HI_init_done = 1; //This should execute first.
		tconf->HI_init(devNum);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\t\tHost thread %d exits setNGetHostConf(devNum = %d)\n", threadID, devNum);
	}
#endif
    return tconf;
}

//Function to convert input string to uppercases.
static char *convertToUpper(char *str) {
    char *newstr, *p;
    p = newstr = strdup(str);
    //while((*p++=toupper(*p)));
    while((*p=toupper(*p))) {p++;} //Changed to unambiguous way
    return newstr;
}


void HostConf::setDefaultDevice() {
    char * envVar;
    char * envVarU;
    envVar = getenv(acc_device_type_env);
    if( envVar == NULL ) {
		user_set_device_type_var = acc_device_default;
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 3
        acc_device_type_var = acc_device_altera;
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 2
        acc_device_type_var = acc_device_xeonphi;
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 6
        acc_device_type_var = acc_device_default;
#else
        acc_device_type_var = acc_device_gpu;
#endif
    } else {
        envVarU = convertToUpper(envVar);
        if( strcmp(envVarU, NVIDIA) == 0 ) {
			user_set_device_type_var = acc_device_nvidia;
            acc_device_type_var = acc_device_gpu;
        } else if( strcmp(envVarU, RADEON) == 0 ) {
			user_set_device_type_var = acc_device_radeon;
            acc_device_type_var = acc_device_gpu;
        } else if( strcmp(envVarU, "ACC_DEVICE_DEFAULT") == 0 ) {
			user_set_device_type_var = acc_device_default;
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 3
        	acc_device_type_var = acc_device_altera;
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 2
        	acc_device_type_var = acc_device_xeonphi;
#else
        	acc_device_type_var = acc_device_gpu;
#endif
        } else if( strcmp(envVarU, ALTERA) == 0 ) {
			user_set_device_type_var = acc_device_altera;
            acc_device_type_var = acc_device_altera;
        } else if( strcmp(envVarU, XEONPHI) == 0 ) {
			user_set_device_type_var = acc_device_xeonphi;
            acc_device_type_var = acc_device_xeonphi;
        } else if( strcmp(envVarU, "ACC_DEVICE_NONE") == 0 ) {
			user_set_device_type_var = acc_device_none;
            acc_device_type_var = acc_device_none;
        } else if( strcmp(envVarU, "ACC_DEVICE_HOST") == 0 ) {
			user_set_device_type_var = acc_device_host;
            acc_device_type_var = acc_device_host;
        } else if( strcmp(envVarU, "ACC_DEVICE_NOT_HOST") == 0 ) {
			user_set_device_type_var = acc_device_not_host;
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 3
        	acc_device_type_var = acc_device_altera;
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 2
        	acc_device_type_var = acc_device_xeonphi;
#else
        	acc_device_type_var = acc_device_gpu;
#endif
        } else {
			user_set_device_type_var = acc_device_none;
            acc_device_type_var = acc_device_none;
        }
        free(envVarU);
    }
}

void HostConf::setDefaultDevNum() {
    int dev;
    char * envVar;
    //Set device number.
    acc_device_t devtype = acc_device_type_var;
    envVar = getenv(acc_device_num_env);
    if( envVar == NULL ) {
    	envVar = getenv(omp_device_num_env);
	}
    if( envVar == NULL ) {
        //default device number (0) will be used.
        dev = 0;
    } else {
        dev = atoi(envVar);
        if( dev < 0 ) {
            dev = 0;
        }
    }
#ifdef _PMAS_
    //dev = get_thread_id();
	dev = threadID;
#endif
    //acc_set_device_num(dev, devtype);
    if( (devtype == acc_device_nvidia) || (devtype == acc_device_not_host) ||
            (devtype == acc_device_default) || (devtype == acc_device_radeon) || 
            (devtype == acc_device_gpu) || (devtype == acc_device_xeonphi) ||
			(devtype == acc_device_altera) || (devtype == acc_device_altera_emulator) ) {
        acc_device_num_var = dev;
    } else if( devtype == acc_device_host ) {
        acc_device_num_var = dev;
    } else {
        fprintf(stderr, "[ERROR in setDefaultDevNum()] Not supported device type %d; exit!\n", devtype);
        exit(1);
    }
}

void HostConf::setTranslationType()
{
    //int dev;
    char * envVar;
    //Set target device type. 
    //acc_device_t devtype = acc_device_type_var;
    envVar = getenv(outputType);
    if( envVar == NULL ) {
        //default device number (0) will be used.
        genOCL = 0;
    } else {
        genOCL = atoi(envVar);
        if( genOCL < 0 ) {
            genOCL = 0;
        }
        if( genOCL == 5 ) {
            genOCL = 0;
        }
        if( genOCL == 6 ) {
            genOCL = 0;
        }
    }

}

void HostConf::createHostTables() {
    prtcntmaptable = new countermap_t;
    hostmemstatusmaptable = new memstatusmap_t;
    devicememstatusmaptable = new memstatusmap_t;
}


//This function initializes OpenARC runtimes for devices, but 
//not the actual device initialization, which will be done by
//acc_set_device_num() function later.
//If DEVICE_NUM_UNDEFINED is passed as an argument to this function,
//acc_set_device_num() is directly called in this function.
//(DEVICE_NUM_UNDEFINED will be passed when this function is
//called in getHostConf() or acc_init() function.)
//(DEVICE_NUM_UNDERFINED is also passed when HI_malloc() is
//called in a device driver, but it will not be actually called
//since an enclosing wrapper function always calls getHostConf()
//before calling the driver-level HI_malloc() function.
void HostConf::HI_init(int devNum) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_init(devType = %d, devNum = %d)\n", acc_device_type_var, devNum);
	}
#endif
    if( HI_hostinit_done == 0 ) {
        HI_hostinit(threadID);
    }
#ifdef _OPENARC_PROFILE_
	double ltime1 = HI_get_localtime();
    //printf("====> Profiling is enabled!!\n");
    H2DMemTrCnt = 0;
    H2HMemTrCnt = 0;
    D2HMemTrCnt = 0;
    D2DMemTrCnt = 0;
    HMallocCnt = 0;
    IHMallocCnt = 0;
    IPMallocCnt = 0;
    DMallocCnt = 0;
    IDMallocCnt = 0;
    HFreeCnt = 0;
    IHFreeCnt = 0;
    IPFreeCnt = 0;
    DFreeCnt = 0;
    IDFreeCnt = 0;
	KernelSyncCnt = 0;
	PresentTableCnt = 0;
	IPresentTableCnt = 0;
	WaitCnt = 0;
	RegKernelArgCnt = 0;
	BTaskCnt = 0;
    H2DMemTrSize = 0;
    H2HMemTrSize = 0;
    D2HMemTrSize = 0;
    D2DMemTrSize = 0;
    HMallocSize = 0;
    IHMallocSize = 0;
    IPMallocSize = 0;
    DMallocSize = 0;
    IDMallocSize = 0;
	CHMemorySize = 0;
	MHMemorySize = 0;
	CIHMemorySize = 0;
	MIHMemorySize = 0;
	CDMemorySize = 0;
	MDMemorySize = 0;
	CIDMemorySize = 0;
	MIDMemorySize = 0;
    totalWaitTime = 0.0;
    totalResultCompTime = 0.0;
    totalMemTrTime = 0.0;
    totalMallocTime = 0.0;
    totalFreeTime = 0.0;
    totalACCTime = ltime1;
    totalInitTime = ltime1;
    totalShutdownTime = 0.0;
    totalKernelSyncTime = 0.0;
    totalPresentTableTime = 0.0;
    totalRegKernelArgTime = 0.0;
	KernelCNTMap.clear();
	KernelTimingMap.clear();
    for (std::set<std::string>::iterator it = kernelnames.begin() ; it != kernelnames.end(); ++it) {
        //const char *kernelName = (*it).c_str();
        //fprintf(stderr, "[HI_init()] Kernel name = %s\n", kernelName);
		KernelCNTMap[*it] = 0;
		KernelTimingMap[*it] = 0.0;
    }  
#endif
    setTranslationType();
    if( acc_device_type_var == acc_device_host ) {
		acc_num_devices = 1;
/*
        devnummap_t numDevMap;
        Accelerator *dev = NULL;
        numDevMap[0] = dev;
#ifdef _THREAD_SAFETY
		pthread_mutex_lock(&mutex_HI_init);
#else
#ifdef _OPENMP
        #pragma omp critical (HI_init_critical)
#endif
#endif
		{ //starts critical section.
			//The current implementation does not support host as a target device.
			HostConf::devMap[acc_device_type_var] = numDevMap;
		} //ends critical section.
#ifdef _THREAD_SAFETY
		pthread_mutex_unlock(&mutex_HI_init);
#endif
*/
        isOnAccDevice = 1;
        HI_init_done = 1;
    } else if( acc_device_type_var != acc_device_none ) {
		//printf("init start with dev %d\n", acc_device_type_var);
        devnummap_t numDevMap;
		acc_device_t ttDevType = acc_device_type_var;
#if defined(OPENARC_ARCH) && OPENARC_ARCH == 6
		ttDevType = acc_device_default;
#endif
        int numDevices;
#ifdef _THREAD_SAFETY
		pthread_mutex_lock(&mutex_HI_init);
#else
#ifdef _OPENMP
        #pragma omp critical (HI_init_critical)
#endif
#endif
		{ //starts critical section.
			if( HostConf::devMap.count(ttDevType) > 0 ) {
				numDevices = HostConf::devMap.at(ttDevType).size();
			} else {
				numDevices = 0;
			}
			if( numDevices == 0 ) {
        		if(genOCL) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH > 0 && OPENARC_ARCH != 5 && OPENARC_ARCH != 6
            		numDevices = OpenCLDriver::HI_get_num_devices(ttDevType);
#else
					fprintf(stderr, "[OPENARCRT-ERROR]To generate OpenCL program, the environment variable OPENARC_ARCH should be a positive integer between 1 and 4.\n");
					exit(1);
#endif
        		}	else {
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
            	numDevices = CudaDriver::HI_get_num_devices(ttDevType);
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 5
            	numDevices = HipDriver::HI_get_num_devices(ttDevType);
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 6
            	numDevices = BrisbaneDriver::HI_get_num_devices_init(ttDevType);
#endif
        		}
				acc_num_devices = numDevices;
				//printf("Num dev %d\n", numDevices);
				//fprintf(stderr, "Init dev num %d\n", acc_device_num_var);
				//[FIXME] initializing multiple devices may not work since only the last
				//device context will be visible to the current host thread; even if 
				//device type/number is changed, 
				//it will not be changed (only the last device will be executed.)
				//Easy fix is to call Accelerator::init() whenever device type/number is 
				//changed or new host thread joins. However, this may create too 
				//many device contexts if device type/number is frequently changed.
				//Better way is to call Accelerator::init() only if context does not 
				//exist; otherwise, attach host thread to the the context for the new 
				//device type/number.
        		for(int i=0 ; i < numDevices; i++) {
            		Accelerator *dev;
            		if(genOCL) {
#if defined(OPENARC_ARCH) && OPENARC_ARCH > 0 && OPENARC_ARCH != 5 && OPENARC_ARCH != 6
                		dev = new OpenCLDriver_t(ttDevType, i, kernelnames, this, numDevices, baseFileName.c_str());
#else
						fprintf(stderr, "[OPENARCRT-ERROR]To generate OpenCL program, the environment variable OPENARC_ARCH should be a positive integer.\n");
						exit(1);
#endif
            		} else {
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
                		dev = new CudaDriver_t(ttDevType, i, kernelnames, this, numDevices, baseFileName.c_str());
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 5
                		dev = new HipDriver_t(ttDevType, i, kernelnames, this, numDevices, baseFileName.c_str());
#elif defined(OPENARC_ARCH) && OPENARC_ARCH == 6
                		dev = new BrisbaneDriver_t(ttDevType, i, kernelnames, this, numDevices, baseFileName.c_str());
#endif
            		}
            		//printf("Dev created %d\n", i);
            		//(*dev).init(); //Init will be called in acc_set_device_num().
            		numDevMap[i] = dev;
        		}
        		//insert all devices of this type into the map
        		HostConf::devMap[ttDevType] = numDevMap;
			}
		} //ends critical section.
#ifdef _THREAD_SAFETY
		pthread_mutex_unlock(&mutex_HI_init);
#endif
		if( devNum == DEVICE_NUM_UNDEFINED ) {
			//[DEBUG] if devNum != DEVICE_NUM_UNDEFINED, device number 
			//will be set by a separate acc_set_device_num() call.
        	setDefaultDevNum();
        	acc_set_device_num(acc_device_num_var, user_set_device_type_var, threadID);
		}
		//printf("init done for type %d\n", ttDevType);
        isOnAccDevice = 1;
        HI_init_done = 1;
    } else if( acc_device_type_var == acc_device_none ) {
        isOnAccDevice = 0;
        HI_init_done = 1;
        acc_num_devices = 1;
    }
#ifdef _OPENARC_RESILIENCE_
    HI_set_srand();
#endif

    //createHostTables();
#ifdef _OPENARC_PROFILE_
    totalInitTime = HI_get_localtime() - ltime1;
#endif

#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_init(devType = %d, devNum = %d)\n", acc_device_type_var, devNum);
	}
#endif
}

void HostConf::HI_reset() {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_reset()\n");
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif

	//Wait until all previous device activities are done.
	device->HI_wait_all(threadID);

    /*
    delete addressmaptable;
    delete countermaptable;
	#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
    delete asyncmaptable;
    #endif
    delete asynchostphostmaptable;
    delete asynchostsizemaptable;
    delete asyncfreemaptable;
    */
    //createHostTables();
    //delete prtcntmaptable;
    //delete hostmemstatusmaptable;
    //delete devicememstatusmaptable;

	//device->masterAddressTable.clear();
	for( addresstablemap_t::iterator it = device->masterAddressTableMap.begin(); it != device->masterAddressTableMap.end(); it++) {
		addresstable_t * tmap = it->second;
		for( addresstable_t::iterator it2 = tmap->begin(); it2 != tmap->end(); it2++) {
		(it2->second)->clear();
	}
		tmap->clear();
	} 
	for( addresstable_t::iterator it = device->masterHandleTable.begin(); it != device->masterHandleTable.end(); it++) {
		(it->second)->clear();
	} 
	for( memPoolmap_t::iterator it = device->memPoolMap.begin(); it != device->memPoolMap.end(); it++) {
		(it->second)->clear();
	} 
	//device->postponedFreeTable.clear();
	for( asyncfreetablemap_t::iterator it = device->postponedFreeTableMap.begin(); it != device->postponedFreeTableMap.end(); it++) {
		(it->second)->clear();
	} 
	
	//device->destroy(threadID);
	//device->init_done = 0;
	//Reset HostConf variables.
/*
	HI_init_done = 0;
	HI_kernels_registered = 0;
	acc_device_type_var = acc_device_none;
	user_set_device_type_var = acc_device_none;
	acc_device_num_var = 0;
	acc_num_devices = 0;
	isOnAccDevice = 0;
	max_mempool_size = _MAX_MEMPOOL_SIZE;
	//use_unifiedmemory = 1;
	//prepin_host_memory = 1;
	//asyncID_offset = 0;
	//threadID = 0;
	HI_kernelnames.clear();
*/

#ifdef _OPENARC_PROFILE_
    //totalFreeTime += HI_get_localtime() - ltime;
    totalShutdownTime = HI_get_localtime() - ltime;
    totalACCTime = HI_get_localtime() - totalACCTime;
    int thread_id = threadID;
	double tAvgTime;
    printf("\n/************************************/\n");
    printf("/* Profile Output for host thread %d */\n", thread_id);
    printf("/************************************/\n");
    printf("Number of Host-to-Device Memory Transfer Calls: %ld\n", H2DMemTrCnt);
    printf("Number of Device-to-Host Memory Transfer Calls: %ld\n", D2HMemTrCnt);
    printf("Number of Host-to-Host Memory Transfer Calls: %ld\n", H2HMemTrCnt);
    printf("Number of Device-to-Device Memory Transfer Calls: %ld\n", D2DMemTrCnt);
    printf("Number of External Device Memory Allocation Calls: %ld\n", DMallocCnt);
    printf("Number of Internal Device Memory Allocation Calls: %ld\n", IDMallocCnt);
    printf("Number of External Host Memory Allocation Calls by OpenARC runtime: %ld\n", HMallocCnt);
    printf("Number of Internal Host Memory Allocation Calls by OpenARC runtime: %ld\n", IHMallocCnt);
    printf("Number of Internal Pinned Memory Allocation Calls by OpenARC runtime: %ld\n", IPMallocCnt);
    printf("Number of External Device Memory Free Calls by OpenARC runtime: %ld\n", DFreeCnt);
    printf("Number of Internal Device Memory Free Calls by OpenARC runtime: %ld\n", IDFreeCnt);
    printf("Number of External Host Memory Free Calls by OpenARC runtime: %ld\n", HFreeCnt);
    printf("Number of Internal Host Memory Free Calls by OpenARC runtime: %ld\n", IHFreeCnt);
    printf("Number of Internal Pinned Memory Free Calls by OpenARC runtime: %ld\n", IPFreeCnt);
    printf("Number of Host-Kernel Synchronization Calls by OpenARC runtime: %ld\n", KernelSyncCnt);
    printf("Number of External Present Table Lookups by OpenARC runtime: %ld\n", PresentTableCnt);
	if( HI_openarcrt_verbosity > 1 ) {
		IPresentTableCnt = (device->presentTableCntMap.find(thread_id))->second;
    	printf("Number of Internal Present Table Lookups by OpenARC runtime: %ld\n", IPresentTableCnt);
	}
    printf("Number of Wait Calls: %ld\n", WaitCnt);
    printf("Number of Kernel Argument Register Calls: %ld\n", RegKernelArgCnt);
    printf("Number of Brisbane Task Submissions: %ld\n", BTaskCnt);
    printf("Size of Data Transferred From Host to Device: %lu\n", H2DMemTrSize);
    printf("Size of Data Transferred From Host to Host: %lu\n", H2HMemTrSize);
    printf("Size of Data Transferred From Device to Host: %lu\n", D2HMemTrSize);
    printf("Size of Data Transferred From Device to Device: %lu\n", D2DMemTrSize);
    printf("Size of Device Memory Externally Requested by OpenARC runtime : %lu\n", DMallocSize);
    printf("Size of Device Memory Internally Requested by OpenARC runtime : %lu\n", IDMallocSize);
    printf("Size of Host Memory Externally Requested by OpenARC runtime : %lu\n", HMallocSize);
    printf("Size of Host Memory Internally Requested by OpenARC runtime : %lu\n", IHMallocSize);
    printf("Size of Pinned Memory Internally Requested by OpenARC runtime : %lu\n", IPMallocSize);
    printf("Size of Maximum Device Memory Usage Externally by OpenARC runtime : %lu\n", MDMemorySize);
    printf("Size of Maximum Device Memory Usage Internally by OpenARC runtime : %lu\n", MIDMemorySize);
	//[DEBUG on Dec. 13, 2020] MHMemorySize and MIHMemorySize are not correctly tracked for now.
    //printf("Size of Maximum Host Memory Usage Externally by OpenARC runtime : %lu\n", MHMemorySize);
    //printf("Size of Maximum Host Memory Usage Internally by OpenARC runtime : %lu\n", MIHMemorySize);
    printf("Total Memory Transfer Time: %lf sec\n", totalMemTrTime);
    printf("Total Memory Allocation Time: %lf sec\n", totalMallocTime);
    printf("Total Memory Free Time: %lf sec\n", totalFreeTime);
    printf("Total ACC Init Time: %lf sec\n", totalInitTime);
    printf("Total ACC Shutdown Time: %lf sec\n", totalShutdownTime);
	if( KernelSyncCnt > 0 ) {
		tAvgTime = totalKernelSyncTime/((double)KernelSyncCnt);
	} else {
		tAvgTime = totalKernelSyncTime;
	}
    printf("Total Host-Kernel Synchronization Time: %lf sec (%lf sec per call)\n", totalKernelSyncTime, tAvgTime);
	if( PresentTableCnt > 0 ) {
		tAvgTime = totalPresentTableTime/((double)PresentTableCnt);
	} else {
		tAvgTime = totalPresentTableTime;
	}
    printf("Total External Present Table Lookup Time: %lf sec (%lf sec per call)\n", totalPresentTableTime, tAvgTime);
	if( WaitCnt > 0 ) {
		tAvgTime = totalWaitTime/((double)WaitCnt);
	} else {
		tAvgTime = totalWaitTime;
	}
    printf("Total Wait Time: %lf sec (%lf sec per call)\n", totalWaitTime, tAvgTime);
	if( RegKernelArgCnt > 0 ) {
		tAvgTime = totalRegKernelArgTime/((double)RegKernelArgCnt);
	} else {
		tAvgTime = totalRegKernelArgTime;
	}
    printf("Total Kernel Arg Register Time: %lf sec (%lf sec per call)\n", totalRegKernelArgTime, tAvgTime);
	long nKernelCalls = 0;
	for(std::map<std::string, long>::iterator it=KernelCNTMap.begin(); it!=KernelCNTMap.end(); ++it) {
		printf("Number of Kernel Calls (%s): %ld\n", (it->first).c_str(), it->second);
		nKernelCalls += it->second;
	}
	printf("Total Number of All Kernel Calls: %ld\n", nKernelCalls);
	double totalKernelTimes = 0.0;
	for(std::map<std::string, double>::iterator it=KernelTimingMap.begin(); it!=KernelTimingMap.end(); ++it) {
		int tCnt = KernelCNTMap.at(it->first);
		if( tCnt > 0 ) {
			tAvgTime = (it->second)/((double)tCnt);
		} else {
			tAvgTime = (it->second);
		}
		printf("Total Execution Time of a Kernel (%s): %lf (%lf per call)\n", (it->first).c_str(), it->second, tAvgTime);
		totalKernelTimes += it->second;
	}
	printf("Total Execution Time of All Kernels: %lf\n", totalKernelTimes);
    printf("Total Time of Other Overhead: %lf sec\n", (totalACCTime - totalKernelTimes - totalWaitTime - totalResultCompTime - totalMemTrTime - totalMallocTime - totalFreeTime - totalInitTime - totalShutdownTime - totalKernelSyncTime - totalPresentTableTime - totalRegKernelArgTime));
    printf("Total Host-Device Execution Time: %lf sec\n", totalACCTime);
    if( totalResultCompTime != 0.0 ) {
        printf("Total Result-Comp Time for Kernel Verification: %lf sec\n", totalResultCompTime);
    }
    H2DMemTrCnt = 0;
    H2HMemTrCnt = 0;
    D2HMemTrCnt = 0;
    D2DMemTrCnt = 0;
    HMallocCnt = 0;
    IHMallocCnt = 0;
    DMallocCnt = 0;
    IDMallocCnt = 0;
    HFreeCnt = 0;
    IHFreeCnt = 0;
    DFreeCnt = 0;
    IDFreeCnt = 0;
	KernelSyncCnt = 0;
	PresentTableCnt = 0;
	IPresentTableCnt = 0;
	WaitCnt = 0;
    H2DMemTrSize = 0;
    H2HMemTrSize = 0;
    D2HMemTrSize = 0;
    D2DMemTrSize = 0;
	HMallocSize = 0;
	IHMallocSize = 0;
	DMallocSize = 0;
	IDMallocSize = 0;
	CHMemorySize = 0;
	MHMemorySize = 0;
	CIHMemorySize = 0;
	MIHMemorySize = 0;
	CDMemorySize = 0;
	MDMemorySize = 0;
	CIDMemorySize = 0;
	MIDMemorySize = 0;
    totalWaitTime = 0.0;
    totalResultCompTime = 0.0;
    totalMemTrTime = 0.0;
    totalMallocTime = 0.0;
    totalFreeTime = 0.0;
    totalACCTime = HI_get_localtime();
    totalInitTime = 0.0;
    totalShutdownTime = 0.0;
	totalKernelSyncTime = 0.0;
	totalPresentTableTime = 0.0;
	totalWaitTime = 0.0;
	KernelCNTMap.clear();
	KernelTimingMap.clear();
#endif
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_reset()\n");
	}
#endif
}

//////////////////////
// Kernel Execution //
//////////////////////
HI_error_t HI_register_kernel_numargs(std::string kernel_name, int num_args, int threadID)
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_register_kernel_numargs(%s, thread ID = %d)\n", kernel_name.c_str(), threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* tconf = getHostConf(threadID);
    return_status = tconf->device->HI_register_kernel_numargs(kernel_name, num_args, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_register_kernel_numargs(thread ID = %d)\n", threadID);
	}
	tconf->RegKernelArgCnt++;
	tconf->totalRegKernelArgTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type, int threadID)
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_register_kernel_arg(arg_index = %d, arg_value = %lx, thread ID = %d)\n", arg_index, *(unsigned long *)arg_value, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* tconf = getHostConf(threadID);
    return_status = tconf->device->HI_register_kernel_arg(kernel_name, arg_index, arg_size, arg_value, arg_type, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_register_kernel_arg(arg_index = %d, arg_value = %lx, thread ID = %d)\n", arg_index, *(unsigned long *)arg_value, threadID);
	}
	tconf->RegKernelArgCnt++;
	tconf->totalRegKernelArgTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_kernel_call(std::string kernel_name, size_t gridSize[3], size_t blockSize[3], int async, int num_waits, int *waits, int threadID) {
	HI_error_t return_status;
	const char *kernelName = kernel_name.c_str();
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_kernel_call(async = %d): %s, thread ID = %d\n", async, kernelName, threadID);
		fprintf(stderr, "                \t\tGang configuration: %lu, %lu, %lu\n", gridSize[2], gridSize[1], gridSize[0]);
		fprintf(stderr, "                \t\tWorker configuration: %lu, %lu, %lu\n", blockSize[2], blockSize[1], blockSize[0]);
	}
#endif
	//if( (gridSize[0] == 0) && (gridSize[1] == 0) && (gridSize[2] == 0) ) {
	if( gridSize[0] == 0 ) {
    	//fprintf(stderr, "[WARNING in HI_kernel_call()] the kernel, %s, is called with 0 gangs; skip executing this kernel.\n", kernel_name);
    	std::cerr << "[WARNING in HI_kernel_call()] the kernel, " << kernel_name << " is called with 0 gangs; skip executing this kernel." << std::endl;
        return HI_success;
	}
    HostConf_t* tconf = getHostConf(threadID);
	int *waitslist = NULL;
	if( num_waits > 0 ) {
		waitslist = (int *)malloc(num_waits*sizeof(int));
		for( int i=0; i<num_waits; i++ ) {
			waitslist[i] = waits[i]+tconf->asyncID_offset;
		}
	}
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_kernel_call()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_kernel_call(kernel_name, gridSize, blockSize, async+tconf->asyncID_offset, num_waits, waitslist, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_kernel_call(async = %d): %s, thread ID = %d\n", async, kernelName, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_synchronize( int forcedSync, int threadID )
{
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_synchronize(sync = %d, thread ID = %d)\n", forcedSync, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t* tconf = getHostConf(threadID);
    return_status = tconf->device->HI_synchronize(forcedSync, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_synchronize(sync = %d, thread ID = %d)\n", forcedSync, threadID);
	}
	tconf->KernelSyncCnt++;
	tconf->totalKernelSyncTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

/////////////////////////////
//Device Memory Allocation //
/////////////////////////////
HI_error_t HI_malloc1D( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc1D(hostPtr = %lx, asyncID = %d, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, threadID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_malloc1D()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_malloc1D()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_malloc1D(hostPtr, devPtr, count, asyncID+tconf->asyncID_offset, flags, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += count;
    tconf->CDMemorySize += count;
    if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    	tconf->MDMemorySize = tconf->CDMemorySize;
    }   
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc1D(hostPtr = %lx, asyncID = %d, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc1D_unified( const void *hostPtr, void** devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc1D_unified(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_malloc1D_unified()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_malloc1D_unified()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_malloc1D_unified()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_malloc1D_unified(hostPtr, devPtr, count, asyncID+tconf->asyncID_offset, flags, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += count;
    tconf->CDMemorySize += count;
    if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    	tconf->MDMemorySize = tconf->CDMemorySize;
    }   
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc1D_unified(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc2D(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( (widthInBytes == 0) || (height == 0) ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_malloc2D()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_malloc2D()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_malloc2D( hostPtr, devPtr,pitch, widthInBytes, height, asyncID+tconf->asyncID_offset, flags, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += widthInBytes*height;
    tconf->CDMemorySize += widthInBytes*height;
    if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    	tconf->MDMemorySize = tconf->CDMemorySize;
    }   
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc2D(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_malloc3D(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( (widthInBytes == 0) || (height == 0) || (depth == 0) ) {
    	fprintf(stderr, "[ERROR in HI_malloc2D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_malloc3D()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_malloc3D()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_malloc3D()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_malloc3D( hostPtr, devPtr, pitch, widthInBytes, height, depth, asyncID+tconf->asyncID_offset, flags, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DMallocCnt++;
	tconf->DMallocSize += widthInBytes*height*depth;
    tconf->CDMemorySize += widthInBytes*height*depth;
    if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    	tconf->MDMemorySize = tconf->CDMemorySize;
    }   
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_malloc3D(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_free( const void *hostPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_free()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	size_t memSize = 0;
	void *devPtr;
	tconf->device->HI_get_device_address(hostPtr, &devPtr, NULL, &memSize, asyncID+tconf->asyncID_offset, tconf->threadID);
#endif
    return_status = tconf->device->HI_free(hostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DFreeCnt++;
	if( memSize > 0 ) {
		tconf->CDMemorySize -= memSize;
	}
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

HI_error_t HI_free_unified( const void *hostPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free_unified(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free_unified()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_free_unified()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	size_t memSize = 0;
	void *devPtr;
	tconf->device->HI_get_device_address(hostPtr, &devPtr, NULL, &memSize, asyncID+tconf->asyncID_offset, tconf->threadID);
#endif
    return_status = tconf->device->HI_free_unified(hostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DFreeCnt++;
	if( memSize > 0 ) {
		tconf->CDMemorySize -= memSize;
	} else {
		fprintf(stderr, "[OPENARCRT-INFO]\tHI_free_unified(asyncID = %d, thread ID = %d) has non-positive size, %lu\n",asyncID, threadID, memSize);
	}
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free_unified(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

//Unlike HI_free(), this method does not do actual memory deallocation;
//instead, it tells that following synchronization calls (acc_wait,
//acc_async_test, etc.) deallocate the device memory for the variable.
HI_error_t HI_free_async( const void *hostPtr, int asyncID, int threadID ) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_free_async(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_free_async()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_free_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
	size_t memSize = 0;
	void *devPtr;
	tconf->device->HI_get_device_address(hostPtr, &devPtr, NULL, &memSize, asyncID+tconf->asyncID_offset, tconf->threadID);
#endif
    return_status = tconf->device->HI_free_async(hostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	tconf->DFreeCnt++;
	if( memSize > 0 ) {
		tconf->CDMemorySize -= memSize;
	} else {
		fprintf(stderr, "[OPENARCRT-INFO]\tHI_free_async(asyncID = %d, thread ID = %d) has non-positive size, %lu\n",asyncID, threadID, memSize);
	}
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_free_async(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
	return return_status;
}

//malloc used for allocating temporary data.
//If the method is called for a pointer to existing memory, the existing memory
//will be freed before allocating new memory.
void HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempMalloc1D(count = %ld, thread ID = %d)\n", count, threadID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_tempMalloc1D()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_tempMalloc1D()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_tempMalloc1D( tempPtr, count, devType, flags, tconf->threadID);
#ifdef _OPENARC_PROFILE_
    if(  devType == acc_device_gpu || devType == acc_device_nvidia ||
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
		tconf->DMallocCnt++;
		tconf->DMallocSize += count;
    	tconf->CDMemorySize += count;
    	if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    		tconf->MDMemorySize = tconf->CDMemorySize;
    	}   
	} else {
		tconf->HMallocCnt++;
		tconf->HMallocSize += count;
    	tconf->CHMemorySize += count;
    	if( tconf->MHMemorySize < tconf->CHMemorySize ) {
    		tconf->MHMemorySize = tconf->CHMemorySize;
    	}   
	}
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempMalloc1D(count = %ld, thread ID = %d)\n", count, threadID);
	}
#endif
}

//malloc used for asynchronously allocating temporary data.
void HI_tempMalloc1D_async( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempMalloc1D_async(count = %ld, asyncID = %d, thread ID = %d)\n", count, asyncID, threadID);
	}
#endif
	if( count == 0 ) {
    	fprintf(stderr, "[ERROR in HI_tempMalloc1D_async()] allocate 0 byte is not allowed; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_tempMalloc1D_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_tempMalloc1D_async( tempPtr, count, devType, flags, asyncID, tconf->threadID);
#ifdef _OPENARC_PROFILE_
    if(  devType == acc_device_gpu || devType == acc_device_nvidia ||
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
		tconf->DMallocCnt++;
		tconf->DMallocSize += count;
    	tconf->CDMemorySize += count;
    	if( tconf->MDMemorySize < tconf->CDMemorySize ) {
    		tconf->MDMemorySize = tconf->CDMemorySize;
    	}   
	} else {
		tconf->HMallocCnt++;
		tconf->HMallocSize += count;
    	tconf->CHMemorySize += count;
    	if( tconf->MHMemorySize < tconf->CHMemorySize ) {
    		tconf->MHMemorySize = tconf->CHMemorySize;
    	}   
	}
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempMalloc1D_async(count = %ld, asyncID = %d, thread ID = %d)\n", count, asyncID, threadID);
	}
#endif
}

//Used for de-allocating temporary data.
void HI_tempFree( void** tempPtr, acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempFree(devType = %d, thread ID = %d)\n", devType, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_tempFree()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
    if(  devType == acc_device_gpu || devType == acc_device_nvidia ||
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
		tconf->DFreeCnt++;
        if( tconf->device->tempMallocSizeMap.count(tconf->threadID) > 0 ) {
        	sizemap_t *tempMallocSize = tconf->device->tempMallocSizeMap[tconf->threadID];
        	if( tempMallocSize->count(*tempPtr) > 0 ) {
        		tconf->CDMemorySize -= (*tempMallocSize)[*tempPtr];
        	} else {
				fprintf(stderr, "[OPENARCRT-INFO]\tHI_tempFree(devType = %d, thread ID = %d) does not know data size: case 1\n",devType, threadID);
			}
        } else {
			fprintf(stderr, "[OPENARCRT-INFO]\tHI_tempFree(devType = %d, thread ID = %d) does not know data size: case 2\n",devType, threadID);
		}
	} else {
		tconf->HFreeCnt++;
	}
#endif
    tconf->device->HI_tempFree( tempPtr, devType, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempFree(devType = %d, thread ID = %d)\n", devType, threadID);
	}
#endif
}

//Unlike HI_tempFree(), this method does not do actual memory deallocation;
//instead, it tells that following synchronization calls (acc_wait,
//acc_async_test, etc.) deallocate the memory.

void HI_tempFree_async( void** tempPtr, acc_device_t devType, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_tempFree_async(devType = %d, asyncID = %d, thread ID = %d)\n", devType, asyncID, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_tempFree_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
#ifdef _OPENARC_PROFILE_
    if(  devType == acc_device_gpu || devType == acc_device_nvidia ||
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
		tconf->DFreeCnt++;
        if( tconf->device->tempMallocSizeMap.count(tconf->threadID) > 0 ) {
        	sizemap_t *tempMallocSize = tconf->device->tempMallocSizeMap[tconf->threadID];
        	if( tempMallocSize->count(*tempPtr) > 0 ) {
        		tconf->CDMemorySize -= (*tempMallocSize)[*tempPtr];
        	} else {
				fprintf(stderr, "[OPENARCRT-INFO]\tHI_tempFree_async(devType = %d, asyncID = %d, thread ID = %d) does not know data size: case 1\n",devType, asyncID, threadID);
			}
        } else {
			fprintf(stderr, "[OPENARCRT-INFO]\tHI_tempFree_async(devType = %d, asyncID = %d, thread ID = %d) does not know data size: case 2\n",devType, asyncID, threadID);
		}
	} else {
		tconf->HFreeCnt++;
	}
#endif
    tconf->device->HI_tempFree_async( tempPtr, devType, asyncID, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_tempFree_async(devType = %d, asyncID = %d, thread ID = %d)\n", devType, asyncID, threadID);
	}
#endif
}

/////////////////////////////////////////////////
//Memory transfers between a host and a device //
/////////////////////////////////////////////////
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0
enum cudaMemcpyKind toCudaMemcpyKind( HI_MemcpyKind_t kind ) {
    switch( kind ) {
    case HI_MemcpyHostToHost: {
        return cudaMemcpyHostToHost;
    }
    case HI_MemcpyHostToDevice: {
        return cudaMemcpyHostToDevice;
    }
    case HI_MemcpyDeviceToHost: {
        return cudaMemcpyDeviceToHost;
    }
    case HI_MemcpyDeviceToDevice: {
        return cudaMemcpyDeviceToDevice;
    }
    }
    return cudaMemcpyHostToHost;
}
#endif

char const *HI_getMemcpyTypeString( HI_MemcpyKind_t kind ) {
	char const *str;
	if(kind == HI_MemcpyHostToHost) {
		str = "Host-to-Host";
	} else if(kind == HI_MemcpyHostToDevice) {
		str = "Host-to-Device";
	} else if(kind == HI_MemcpyDeviceToHost) {
		str = "Device-to-Host";
	} else {
		str = "Device-to-Device";
	}
	return str;
}

// Copy count bytes from the memory area pointed by src to the memory area
// pointed by dst, where kind is one of HI_MemcpyHostToHost, HI_MemcpyHostToDevice,
// HI_MemcpyDeviceToHost, or HI_MemcpyDeviceToDevice.
//     - trType is one of the following:
//         0: normal memcopy; for CUDA, this simply wraps cudaMemcpy().
//         1: use cudaMemcpyToSymbol or cudaMemcpyFromSymbol
HI_error_t HI_memcpy(void *dst, const void *src, size_t count,
                           HI_MemcpyKind_t kind, int trType, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy()] NULL src pointer; exit!\n");
        exit(1);
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy( dst, src, count, kind, trType, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_unified(void *dst, const void *src, size_t count,
                           HI_MemcpyKind_t kind, int trType, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_unified(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_unified()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
		//If src pointer is NULL, skip memory copy operation.
		return HI_success;
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy_unified()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy_unified()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy_unified( dst, src, count, kind, trType, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_unified(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_async(void *dst, const void *src, size_t count,
                                 HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_async(async = %d, count = %ld, thread ID = %d)\n", async, count, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_async()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_async()] NULL src pointer; exit!\n");
        exit(1);
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		int *waitslist = NULL;
		if( num_waits > 0 ) {
			waitslist = (int *)malloc(num_waits*sizeof(int));
			for( int i=0; i<num_waits; i++ ) {
				waitslist[i] = waits[i]+tconf->asyncID_offset;
			}
		}
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy_async(dst, src, count, kind, trType, async+tconf->asyncID_offset, num_waits, waitslist, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_async(async = %d, count = %ld, thread ID = %d)\n", async, count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_asyncS(void *dst, const void *src, size_t count,
                                 HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_asyncS(count = %ld, async = %d, thread ID = %d)\n", count, async, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] NULL src pointer; exit!\n");
        exit(1);
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
		int *waitslist = NULL;
		if( num_waits > 0 ) {
			waitslist = (int *)malloc(num_waits*sizeof(int));
			for( int i=0; i<num_waits; i++ ) {
				waitslist[i] = waits[i]+tconf->asyncID_offset;
			}
		}
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy_asyncS()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy_asyncS(dst, src, count, kind, trType, async+tconf->asyncID_offset, num_waits, waitslist, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_asyncS(count = %ld, async = %d, thread ID = %d)\n", count, async, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

void HI_waitS1(int async, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_waitS1(async = %d, thread ID = %d)\n", async, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_waitS1()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_waitS1(async+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_waitS1(async = %d, thread ID = %d)\n", async, threadID);
	}
	tconf->WaitCnt++;
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif
}

void  HI_waitS2(int async, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_waitS2(async = %d, thread ID = %d)\n", async, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_waitS2()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_waitS2(async+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_waitS2(async = %d, thread ID = %d)\n", async, threadID);
	}
	tconf->totalWaitTime += (HI_get_localtime() - ltime);
#endif

}

HI_error_t HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                             size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy2D(size = %ld, thread ID = %d)\n", widthInBytes*height, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D()] NULL src pointer; exit!\n");
        exit(1);
	}
	if( widthInBytes*height > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy2D()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy2D()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy2D(dst, dpitch, src, spitch, widthInBytes, height, kind, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy2D(size = %ld, thread ID = %d)\n", widthInBytes*height, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy2D_async(void *dst, size_t dpitch, const void *src,
                                   size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy2D_async(size = %ld, async = %d, thread ID = %d)\n", widthInBytes*height, async, threadID);
		fprintf(stderr, "                \tdst = %lx\tsrc = %lx\n", (unsigned long)dst, (unsigned long)src);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( dst == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] NULL dst pointer; exit!\n");
        exit(1);
	} else if( src == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] NULL src pointer; exit!\n");
        exit(1);
	}
	if( widthInBytes*height > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		int *waitslist = NULL;
		if( num_waits > 0 ) {
			waitslist = (int *)malloc(num_waits*sizeof(int));
			for( int i=0; i<num_waits; i++ ) {
				waitslist[i] = waits[i]+tconf->asyncID_offset;
			}
		}
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy2D_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy2D_async(dst, dpitch, src, spitch, widthInBytes, height, kind, async+tconf->asyncID_offset, num_waits, waitslist, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy2D_async(size = %ld, async = %d, thread ID = %d)\n", widthInBytes*height, async, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

//extern HI_error_t HI_memcpy3D(void *dst, size_t dpitch, const void *src, size_t spitch,
//	size_t widthInBytes, size_t height, size_t depth, HI_MemcpyKind_t kind);
//extern HI_error_t HI_memcpy3D_async(void *dst, size_t dpitch, const void *src,
//	size_t spitch, size_t widthInBytes, size_t height, size_t depth,
//	HI_MemcpyKind_t kind, int async);

////////////////////////////
//Internal mapping tables //
////////////////////////////
HI_error_t HI_get_device_address(const void * hostPtr, void **devPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_device_address(hostPtr = %lx, asyncID = %d, thread ID = %d)\n",(unsigned long)hostPtr, asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_get_device_address(hostPtr, devPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_device_address(hostPtr = %lx, devPtr = %lx, asyncID = %d, thread ID = %d)\n",(unsigned long)hostPtr, (unsigned long)(*devPtr), asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_get_device_address(const void * hostPtr, void **devPtrBase, size_t *offset, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_get_device_address(hostPtr, devPtrBase, offset, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_get_device_address(const void * hostPtr, void **devPtrBase, size_t *offset, size_t *size, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_get_device_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_get_device_address(hostPtr, devPtrBase, offset, size, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_set_device_address(const void * hostPtr, void *devPtr, size_t size, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_device_address(asyncID = %d , thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_set_device_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_set_device_address(hostPtr, devPtr, size, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_device_address(asyncID = %d , thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_remove_device_address(const void * hostPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_remove_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_remove_device_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_remove_device_address(hostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_remove_device_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}


HI_error_t HI_get_host_address(const void * devPtr, void **hostPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_host_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_get_host_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_get_host_address(devPtr, hostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_host_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

HI_error_t HI_get_temphost_address(const void * hostPtr, void **temphostPtr, int asyncID, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_get_temphost_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_get_temphost_address()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_get_temphost_address(hostPtr, temphostPtr, asyncID+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_get_temphost_address(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
#endif
	return return_status;
}

int HI_getninc_prtcounter(const void * hostPtr, void **devPtr, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_getninc_prtcounter(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);

    int result;
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType, tconf->threadID);
    countermap_t * prtcounter = tconf->prtcntmaptable;
	void * hostPtrBase = (void *)hostPtr;
	void *devPtrBase;
	size_t offset;
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_getninc_prtcounter()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    if( tconf->device->HI_get_device_address(hostPtr, &devPtrBase, &offset, asyncID+tconf->asyncID_offset, tconf->threadID) == HI_success ) {
		if( offset > 0 ) {
			hostPtrBase = (void *)((size_t)hostPtrBase - offset);
			*devPtr = (void *) ((size_t)devPtrBase + offset);
		} else {
			*devPtr = devPtrBase;
		}
        if( prtcounter->count(hostPtrBase) > 0 ) {
            result = prtcounter->at(hostPtrBase);
            if( result <= 0 ) result = 1;
            (*prtcounter)[hostPtrBase] = result + 1;
        } else {
            result = 1;
            (*prtcounter)[hostPtrBase] = 2;
        }
    } else {
        *devPtr = 0;
		if( prtcounter->count(hostPtrBase) > 0 ) {
        	(*prtcounter)[hostPtrBase] = 0;
		}
        result = 0;
    }
#ifdef _OPENARC_PROFILE_
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_getninc_prtcounter(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
    return result;
}

int HI_decnget_prtcounter(const void * hostPtr, void **devPtr, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_decnget_prtcounter(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);

    int result;
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType, tconf->threadID);
    countermap_t * prtcounter = tconf->prtcntmaptable;
	void * hostPtrBase = (void *)hostPtr;
	void *devPtrBase;
	size_t offset;
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_decnget_prtcounter()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    if( tconf->device->HI_get_device_address(hostPtr, &devPtrBase, &offset, asyncID+tconf->asyncID_offset, tconf->threadID) == HI_success) {
		if( offset > 0 ) {
			hostPtrBase = (void *)((size_t)hostPtrBase - offset);
			*devPtr = (void *) ((size_t)devPtrBase + offset);
		} else {
			*devPtr = devPtrBase;
		}
        if( prtcounter->count(hostPtrBase) > 0 ) {
            result = prtcounter->at(hostPtrBase);
            result = result -1;
            if( result < 0 ) result = 0;
        } else {
            result = 0;
        }
        (*prtcounter)[hostPtrBase] = result;
    } else {
        *devPtr = 0;
		if( prtcounter->count(hostPtrBase) > 0 ) {
        	(*prtcounter)[hostPtrBase] = 0;
		}
        result = -1; //error!!
    }
#ifdef _OPENARC_PROFILE_
	tconf->PresentTableCnt++;
	tconf->totalPresentTableTime += (HI_get_localtime() - ltime);
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_decnget_prtcounter(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
    return result;
}

//extern size_t HI_get_pitch(const void *hostPtr);

/////////////////////////////////////////////////////////////////////////
//async integer argument => internal handler (ex: CUDA stream) mapping //
/////////////////////////////////////////////////////////////////////////


void HostConf::initKernelNames(int kernels, std::string kernelNames[]) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HostConf::initKernelNames(num of kernels = %d)\n", kernels);
	}
#endif
    for(int i= 0 ; i< kernels; i++) {
        kernelnames.insert(kernelNames[i]);
#ifdef _THREAD_SAFETY
		pthread_mutex_lock(&mutex_HI_kernelnames);
#else
#ifdef _OPENMP
        #pragma omp critical (HI_kernelnames_critical)
#endif
#endif
		{
			HostConf::HI_kernelnames.insert(kernelNames[i]);
		}
#ifdef _THREAD_SAFETY
		pthread_mutex_unlock(&mutex_HI_kernelnames);
#endif
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 2 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HostConf::initKernelNames(num of kernels = %d)\n", kernels);
	}
#endif
}

//Non-master threads initialize kernelnames using the static kernelname set.
//Not used for now.
void HostConf::initKernelNames() {
    for (std::set<std::string>::iterator it = HostConf::HI_kernelnames.begin() ; it != HostConf::HI_kernelnames.end(); ++it) {
        kernelnames.insert(*it);
    }   
}

void HostConf::addKernelNames(int kernels, std::string kernelNames[]) {
    for(int i= 0 ; i< kernels; i++) {
		std::string tName = kernelNames[i];
		if( kernelnames.count(tName) == 0 ) {
        	kernelnames.insert(tName);
#ifdef _THREAD_SAFETY
		pthread_mutex_lock(&mutex_HI_kernelnames);
#else
#ifdef _OPENMP
        #pragma omp critical (HI_kernelnames_critical)
#endif
#endif
			{
				HostConf::HI_kernelnames.insert(kernelNames[i]);
			}
#ifdef _THREAD_SAFETY
		pthread_mutex_unlock(&mutex_HI_kernelnames);
#endif
#ifdef _OPENARC_PROFILE_
			KernelCNTMap[tName] = 0;
			KernelTimingMap[tName] = 0.0;
#endif
		}
    }
}


//Compiler will insert this before the first read access of the variable.
void HI_check_read(const void * hostPtr, acc_device_t dtype, const char * varName, const char *refName, int loopIndex, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_check_read(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu) || (dtype == acc_device_xeonphi) || (dtype == acc_device_altera) || (dtype == acc_device_altera_emulator) ) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( devicestatus == HI_stale ) {
            //printf("[DEBUG-ERROR] variable %32s should be copied from host to device for %64s.\n", varName, refName);
            std::cout <<"[DEBUG-ERROR] variable " << varName << " should be copied from host to device for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<"." <<std::endl;
        }
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( hoststatus == HI_stale ) {
            //printf("[DEBUG-ERROR] variable %32s should be copied from device to host for %64s.\n", varName, refName);
            std::cout <<"[DEBUG-ERROR] variable " << varName << " should be copied from device to host for " << refName <<"." <<std::endl;
        }
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_check_read(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
}

//Compiler will insert this before the first write access of the variable.
void HI_check_write(const void * hostPtr, acc_device_t dtype, const char * varName, const char *refName, int loopIndex, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_check_write(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType, tconf->threadID);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu) || (dtype == acc_device_xeonphi) || (dtype == acc_device_altera) || (dtype == acc_device_altera_emulator) ) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( devicestatus == HI_stale ) {
            //printf("[DEBUG-WARNING] variable %32s should be copied from host to device for %64s unless it is completely overwritten.\n", varName, refName);
            std::cout <<"[DEBUG-WARNING] variable " << varName << " should be copied from host to device for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<", unless it is completely overwritten before it is read." <<std::endl;
        }
        (*hostmemstatusmap)[hostPtr] = HI_stale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( hoststatus == HI_stale ) {
            //printf("[DEBUG-WARNING] variable %32s should be copied from device to host for %64s unless it is completely overwritten.\n", varName, refName);
            std::cout <<"[DEBUG-WARNING] variable " << varName << " should be copied from device to host for " << refName;
            if( loopIndex != INT_MIN ) {
                std::cout <<" (enclosing loop index = " << loopIndex <<")";
            }
            std::cout <<", unless it is completely overwritten before it is read." <<std::endl;
        }
        (*devicememstatusmap)[hostPtr] = HI_stale;
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_check_write(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
}

//Compiler will insert this after each memory transfer call for the variable
//or after GPU memory is freed.
void HI_set_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, const char * varName, const char *refName, int loopIndex, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_status(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);

    //acc_device_t devType = acc_get_device_type();
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType, tconf->threadID);
    memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
    memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
    //Initialize status maps if not existing (HI_init_status).
    if( hostmemstatusmap->count(hostPtr) == 0 ) {
        (*hostmemstatusmap)[hostPtr] = HI_notstale;
        (*devicememstatusmap)[hostPtr] = HI_notstale;
    }
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu) || (dtype == acc_device_xeonphi) || (dtype == acc_device_altera) || (dtype == acc_device_altera_emulator) ) {
        HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( status == HI_notstale ) {
            if( devicestatus == HI_notstale ) {
                //printf("[DEBUG-INFO] copying variable %32s from host to device for %64s seems to be redundant.\n", varName, refName);
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from host to device for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" seems to be redundant." <<std::endl;
            } else if( devicestatus == HI_maystale ) {
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from host to device for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" can be redundant if it is completely overwritten by the device befere it is read." <<std::endl;
            }
        }
        (*devicememstatusmap)[hostPtr] = status;
    } else {
        HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        if( status == HI_notstale ) {
            if( hoststatus == HI_notstale ) {
                //printf("[DEBUG-INFO] copying variable %32s from device to host for %64s seems to be redundant.\n", varName, refName);
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from device to host for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" seems to be redundant." <<std::endl;
            } else if( hoststatus == HI_maystale ) {
                std::cout <<"[DEBUG-INFO] copying variable " << varName << " from device to host for " << refName;
                if( loopIndex != INT_MIN ) {
                    std::cout <<" (enclosing loop index = " << loopIndex <<")";
                }
                std::cout <<" can be redundant if it is completely overwritten by the host befere it is read." <<std::endl;
            }
        }
        (*hostmemstatusmap)[hostPtr] = status;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_status(%s, %s, thread ID = %d)\n", varName, refName, threadID);
	}
#endif
}

//Compiler will insert this right after a kernel call if the compiler analyzes
//that either the following CPU region does not access the variable
//(CPU status = notstale) or the variable seems to be not upward-exposed in the
//following CPU region (CPU status = maystale).
//This method is also inserted right after a kernel call for each reduction variable
//(GPU status = stale) or if GPU variable is deallocated (GPU status = stale).
//This method is similar to HI_set_status(), but this does not check any error
//or redundancy.
void HI_reset_status(const void * hostPtr, acc_device_t dtype, HI_memstatus_t status, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_reset_status(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);

    //acc_device_t devType = acc_get_device_type(tconf->threadID);
    acc_device_t devType = tconf->acc_device_type_var;
    int devNum = acc_get_device_num(devType, tconf->threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_reset()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    if( dtype == acc_device_nvidia || (dtype == acc_device_radeon) || (dtype == acc_device_gpu) || (dtype == acc_device_xeonphi) || (dtype == acc_device_altera) || (dtype == acc_device_altera_emulator) ) {
        memstatusmap_t * devicememstatusmap = tconf->devicememstatusmaptable;
        //HI_memstatus_t devicestatus = (*devicememstatusmap)[hostPtr];
        if( status == HI_stale ) {
            //Set the status to stale if GPU variable is freed.
            addresstable_t::iterator it = tconf->device->masterAddressTableMap[tconf->threadID]->find(asyncID+tconf->asyncID_offset);
            std::map<const void *,void*>::iterator it2 =	(it->second)->find(hostPtr);
            if(it2 == (it->second)->end() ) {
                (*devicememstatusmap)[hostPtr] = status;
            }
            /*
            addressmap_t * addressmap = tconf->addressmaptable;
            if( addressmap->count(hostPtr) == 0 ) {
            	(*devicememstatusmap)[hostPtr] = status;
            }
            */
            else if (asyncID != DEFAULT_QUEUE) {
                /*
                asyncfreemap_t * asyncfreemap = tconf->asyncfreemaptable;
                pointerset_t * freeset;
                if( asyncfreemap->count(asyncID+tconf->asyncID_offset) > 0 ) {
                	freeset = asyncfreemap->at(asyncID+tconf->asyncID_offset);
                	//GPU variable will be freed asyncronously.
                	if( (freeset != 0) && (freeset->count(hostPtr) > 0) ) {
                		(*devicememstatusmap)[hostPtr] = status;
                	}
                }
                */
				asyncfreetable_t *postponedFreeTable = tconf->device->postponedFreeTableMap[tconf->threadID];
                asyncfreetable_t::iterator hostPtrIter = postponedFreeTable->find(asyncID+tconf->asyncID_offset);

                while(hostPtrIter != postponedFreeTable->end()) {
                    //fprintf(stderr, "[in HI_postponed_free()] Freeing on stream %d, address %x\n", asyncID, hostPtrIter->second);
                    if(hostPtrIter->second == hostPtr) {
                        (*devicememstatusmap)[hostPtr] = status;
                        break;
                    }
                    hostPtrIter++;
                }
            } else {
                //Set the status to stale if the variable is reduction one.
                (*devicememstatusmap)[hostPtr] = status;
            }
        } else {
            (*devicememstatusmap)[hostPtr] = status;
        }
    } else {
        memstatusmap_t * hostmemstatusmap = tconf->hostmemstatusmaptable;
        //HI_memstatus_t hoststatus = (*hostmemstatusmap)[hostPtr];
        (*hostmemstatusmap)[hostPtr] = status;
    }
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 0 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_reset_status(asyncID = %d, thread ID = %d)\n", asyncID, threadID);
	}
#endif
}

HI_error_t HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_bind_tex(thread ID = %d)\n", threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_bind_tex()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_bind_tex(texName, type, devPtr, size, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_bind_tex(thread ID = %d)\n", threadID);
	}
#endif
	return return_status;
}


HI_error_t HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_const(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_const()] NULL host pointer; exit!\n");
        exit(1);
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy_const()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy_const()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy_const(hostPtr, constName, kind, count, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_const(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_memcpy_const_async(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int async, int num_waits, int *waits, int threadID) {
	HI_error_t return_status = HI_success;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_memcpy_const_async(async = %d, count = %ld, thread ID = %d)\n",async, count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_memcpy_const_async()] NULL host pointer; exit!\n");
        exit(1);
	}
	if( count > 0 ) {
    	HostConf_t * tconf = getHostConf(threadID);
    	if( tconf->isOnAccDevice == 0 ) {
        	fprintf(stderr, "[ERROR in HI_memcpy_const_async()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        	exit(1);
    	}    
		int *waitslist = NULL;
		if( num_waits > 0 ) {
			waitslist = (int *)malloc(num_waits*sizeof(int));
			for( int i=0; i<num_waits; i++ ) {
				waitslist[i] = waits[i]+tconf->asyncID_offset;
			}
		}
		if(tconf->device == NULL) {
        	fprintf(stderr, "[ERROR in HI_memcpy_const_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
			exit(1);
		}
    	return_status = tconf->device->HI_memcpy_const_async(hostPtr, constName, kind, count, async+tconf->asyncID_offset, num_waits, waitslist, tconf->threadID);
	}
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_memcpy_const_async(async = %d, count = %ld, thread ID = %d)\n",async, count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

HI_error_t HI_present_or_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
	HI_error_t return_status;
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_present_or_memcpy_const(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	if( hostPtr == NULL ) {
    	fprintf(stderr, "[ERROR in HI_present_or_memcpy_const()] NULL host pointer; exit!\n");
        exit(1);
	}
    HostConf_t * tconf = getHostConf(threadID);
    if( tconf->isOnAccDevice == 0 ) {
        fprintf(stderr, "[ERROR in HI_present_or_memcpy_const()] Not supported operation for the current device type %d; exit!\n", tconf->acc_device_type_var);
        exit(1);
    }    
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_present_or_memcpy_const()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    return_status = tconf->device->HI_present_or_memcpy_const(hostPtr, constName, kind, count, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_present_or_memcpy_const(count = %ld, thread ID = %d)\n", count, threadID);
		fprintf(stderr, "                \tMemcpy Type: %s\n", HI_getMemcpyTypeString(kind));
	}
#endif
	return return_status;
}

//This call ensures that the corresponding queue exists. If not, it is created.
void HI_set_async(int asyncId, int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_async(asyncID = %d, thread ID = %d)\n", asyncId, threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_set_async()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_set_async(asyncId+tconf->asyncID_offset, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_async(asyncID = %d, thread ID = %d)\n", asyncId, threadID);
	}
	tconf->totalKernelSyncTime += (HI_get_localtime() - ltime);
#endif
}

const char* HI_get_device_type_string( acc_device_t devtype ) {
	static std::string str = "";
	switch ( devtype ) {
		case acc_device_none: { str = "acc_device_none"; break; }
		case acc_device_default: { str = "acc_device_default"; break; }
		case acc_device_host: { str = "acc_device_host"; break; }
		case acc_device_not_host: { str = "acc_device_not_host"; break; }
		case acc_device_nvidia: { str = "acc_device_nvidia"; break; }
		case acc_device_radeon: { str = "acc_device_radeon"; break; }
		case acc_device_gpu: { str = "acc_device_gpu"; break; }
		case acc_device_xeonphi: { str = "acc_device_xeonphi"; break; }
		case acc_device_current: { str = "acc_device_current"; break; }
		case acc_device_altera: { str = "acc_device_altera"; break; }
		case acc_device_altera_emulator: { str = "acc_device_altera_emulator"; break; }
		default: { str = "UNKNOWN TYPE"; break; }
	}
	return str.c_str();	
}

//This call ensures that the current host thread has correct device context.
void HI_set_context(int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_set_context(thread ID = %d)\n", threadID);
	}
	double ltime = HI_get_localtime();
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_set_context()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_set_context(tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_set_context(thread ID = %d)\n", threadID);
	}
	tconf->totalKernelSyncTime += (HI_get_localtime() - ltime);
#endif
}

void HI_set_context() {
	HI_set_context(NO_THREAD_ID);
}

void HI_enter_subregion(const char *label,  int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_enter_subregion(label = %s, thread ID = %d)\n", label, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_enter_subregion()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_enter_subregion(label, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_enter_subregion(label = %s, thread ID = %d)\n", label, threadID);
	}
#endif
}

void HI_enter_subregion(const char *label) {
	HI_enter_subregion(label, NO_THREAD_ID);
}

void HI_exit_subregion(const char *label,  int threadID) {
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\tenter HI_exit_subregion(label = %s, thread ID = %d)\n", label, threadID);
	}
#endif
    HostConf_t * tconf = getHostConf(threadID);
	if(tconf->device == NULL) {
        fprintf(stderr, "[ERROR in HI_exit_subregion()] Not supported in the current device type %d; exit!\n", tconf->acc_device_type_var);
		exit(1);
	}
    tconf->device->HI_exit_subregion(label, tconf->threadID);
#ifdef _OPENARC_PROFILE_
	if( HI_openarcrt_verbosity > 1 ) {
		fprintf(stderr, "[OPENARCRT-INFO]\texit HI_exit_subregion(label = %s, thread ID = %d)\n", label, threadID);
	}
#endif
}

void HI_exit_subregion(const char *label) {
	HI_exit_subregion(label,  NO_THREAD_ID);
}

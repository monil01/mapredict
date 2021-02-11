#include "openacc.h"
#include "openaccrt_ext.h"
#include <brisbane/rt/DeviceCUDA.h>
#include <brisbane/rt/LoaderCUDA.h>
#include <brisbane/rt/Mem.h>

//Below structures contain Brisbane device IDs for a given device type.
std::vector<int> BrisbaneDriver::NVIDIADeviceIDs;
std::vector<int> BrisbaneDriver::AMDDeviceIDs;
std::vector<int> BrisbaneDriver::GPUDeviceIDs;
std::vector<int> BrisbaneDriver::CPUDeviceIDs;
std::vector<int> BrisbaneDriver::FPGADeviceIDs;
std::vector<int> BrisbaneDriver::PhiDeviceIDs;
std::vector<int> BrisbaneDriver::DefaultDeviceIDs;

int BrisbaneDriver::HI_getBrisbaneDeviceID(acc_device_t devType, acc_device_t userInput, int devnum) {
	int brisbaneDeviceType = brisbane_default;
	int brisbaneDeviceID = devnum;
	switch (devType) {
		case acc_device_default: { brisbaneDeviceType = brisbane_default; break; }
		case acc_device_host: { brisbaneDeviceType = brisbane_cpu; break; }
		case acc_device_not_host: { brisbaneDeviceType = brisbane_default; break; }
		case acc_device_nvidia: { brisbaneDeviceType = brisbane_nvidia; break; }
		case acc_device_radeon: { brisbaneDeviceType = brisbane_amd; break; }
		case acc_device_gpu: { if( userInput == acc_device_nvidia ) {brisbaneDeviceType = brisbane_nvidia;} 
								else if( userInput == acc_device_radeon ) {brisbaneDeviceType = brisbane_amd;}
								else {brisbaneDeviceType = brisbane_gpu;}
								break; }
		case acc_device_xeonphi: { brisbaneDeviceType = brisbane_phi; break; }
		case acc_device_current: { brisbaneDeviceType = brisbane_default; break; }
		case acc_device_altera: { brisbaneDeviceType = brisbane_fpga; break; }
		case acc_device_altera_emulator: { brisbaneDeviceType = brisbane_fpga; break; }
		default: { brisbaneDeviceType = brisbane_default; break; }
	}
	switch (brisbaneDeviceType) {
		case brisbane_default: { if( devnum >= DefaultDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_default).\n",devnum, DefaultDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = DefaultDeviceIDs[devnum];
									}
									break; }
		case brisbane_nvidia: { if( devnum >= NVIDIADeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_nvidia).\n",devnum, NVIDIADeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = NVIDIADeviceIDs[devnum];
									}
									break; }
		case brisbane_amd: { if( devnum >= AMDDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_amd).\n",devnum, AMDDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = AMDDeviceIDs[devnum];
									}
									break; }
		case brisbane_gpu: { if( devnum >= GPUDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_gpu).\n",devnum, GPUDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = GPUDeviceIDs[devnum];
									}
									break; }
		case brisbane_phi: { if( devnum >= PhiDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_phi).\n",devnum, PhiDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = PhiDeviceIDs[devnum];
									}
									break; }
		case brisbane_fpga: { if( devnum >= FPGADeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_fpga).\n",devnum, FPGADeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = FPGADeviceIDs[devnum];
									}
									break; }
		case brisbane_cpu: { if( devnum >= CPUDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_cpu).\n",devnum, CPUDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = CPUDeviceIDs[devnum];
									}
									break; }
		default: { if( devnum >= DefaultDeviceIDs.size() ) {
      									fprintf(stderr, "[ERROR in BrisbaneDriver::HI_getBrisbaneDeviceID()] device number (%d) is greater than the number of available devices (%lu) of a given type (brisbane_default).\n",devnum, DefaultDeviceIDs.size());
      									exit(1);
									} else {
										brisbaneDeviceID = DefaultDeviceIDs[devnum];
									}
									break; }
	}
	return brisbaneDeviceID;
}

BrisbaneDriver::BrisbaneDriver(acc_device_t devType, int devNum, std::set<std::string>kernelNames, HostConf_t *conf, int numDevices, const char *baseFileName) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::BrisbaneDriver(%s, %d)\n", HI_get_device_type_string(devType), devNum);
    }    
#endif
  current_mempool_size = 0;
  fileNameBase = std::string(baseFileName);

#ifdef _THREAD_SAFETY
  pthread_mutex_lock(&mutex_set_device_num);
#else
#ifdef _OPENMP
  #pragma omp critical(acc_set_device_num_critical)
#endif
#endif
  for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
    kernelNameSet.insert(*it);
  }
#ifdef _THREAD_SAFETY
  pthread_mutex_unlock(&mutex_set_device_num);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::BrisbaneDriver(%s, %d)\n", HI_get_device_type_string(devType), devNum);
    }    
#endif
}

int bind_tex_handler(void* p, void* device) {
  char* params = (char*) p;
  int off = 0;
  size_t name_len = *((size_t*) params);
  off += sizeof(name_len);
  char* name = (char*) malloc(name_len);
  memcpy(name, params + off, name_len);
  off += name_len;
  int type = *((int*) (params + off));
  off += sizeof(type);
  void* dptr = *((void**) (params + off));
  off += sizeof(dptr);
  size_t size = *((size_t*) (params + off));
  //printf("[%s:%d] namelen[%lu] name[%s] type[%d] dptr[%p] size[%lu]\n", __FILE__, __LINE__, name_len, name, type, dptr, size);

  CUtexref tex;
  CUresult err;
  brisbane_mem m = (brisbane_mem) dptr;
  brisbane::rt::DeviceCUDA* dev = (brisbane::rt::DeviceCUDA*) device;
  brisbane::rt::LoaderCUDA* ld = dev->ld();
  brisbane::rt::Mem* mem = m->class_obj;
  err = ld->cuModuleGetTexRef(&tex, *(dev->module()), name);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = ld->cuTexRefSetAddress(0, tex, (CUdeviceptr) mem->arch(dev), size);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = ld->cuTexRefSetAddressMode(tex, 0, CU_TR_ADDRESS_MODE_WRAP);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = ld->cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = ld->cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  err = ld->cuTexRefSetFormat(tex, type == brisbane_int ? CU_AD_FORMAT_SIGNED_INT32 : CU_AD_FORMAT_FLOAT, 1);
  if (err != CUDA_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err);
  return 0;
}

HI_error_t BrisbaneDriver::init(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::init()\n");
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif

  current_mempool_size = 0;
  HostConf_t * tconf = getHostConf(threadID);
  int thread_id = tconf->threadID;

  int err;
  err = brisbane_init(NULL, NULL, 1);
  if (err == BRISBANE_OK) brisbane_register_command(0xdeadcafe, brisbane_nvidia, bind_tex_handler);
  int ndevs = 0;
  err = brisbane_device_count(&ndevs);
  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
  int defaultType = brisbane_default;
  for(int i=0; i<ndevs; i++) {
  	int type;
  	size_t size;
    err = brisbane_device_info(i, brisbane_type, &type, &size);  
  	if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
    if( i==0 ) {
      defaultType = type;
    }
    if( type == defaultType ) {
      DefaultDeviceIDs.push_back(i);
    }
    if( type == brisbane_nvidia ) {
      NVIDIADeviceIDs.push_back(i);
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_amd ) {
      AMDDeviceIDs.push_back(i);
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_gpu ) {
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_cpu ) {
      CPUDeviceIDs.push_back(i);
    } else if( type == brisbane_fpga ) {
      FPGADeviceIDs.push_back(i);
    } else if( type == brisbane_phi ) {
      PhiDeviceIDs.push_back(i);
    }
  }

  masterAddressTableMap[thread_id] = new addresstable_t();
  masterHandleTable[thread_id] = new addressmap_t();
  postponedFreeTableMap[thread_id] = new asyncfreetable_t();
  postponedTempFreeTableMap[thread_id] = new asynctempfreetable_t();
  postponedTempFreeTableMap2[thread_id] = new asynctempfreetable2_t();
  memPoolMap[thread_id] = new memPool_t();
  tempMallocSizeMap[thread_id] = new sizemap_t();
  threadTaskMap[thread_id] = NULL;
  threadTaskMapNesting[thread_id] = 0;
  threadHostMemFreeMap[thread_id] = new pointerset_t();

  createKernelArgMap(thread_id);
  init_done = 1;
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::init()\n");
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_register_kernels(std::set<std::string>kernelNames, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_register_kernels(thread ID = %d)\n", threadID);
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
#ifdef _THREAD_SAFETY
  pthread_mutex_lock(&mutex_set_device_num);
#else
#ifdef _OPENMP
  #pragma omp critical(acc_set_device_num_critical)
#endif
#endif
  {
  HostConf_t * tconf = getHostConf(threadID);
  for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
    if( kernelNameSet.count(*it) == 0 ) {
      int err;
      brisbane_kernel kernel;
      const char *kernelName = (*it).c_str();
      err = brisbane_kernel_create(kernelName, &kernel);
      if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      (tconf->kernelsMap[this])[*it] = kernel;
      kernelParams_t *kernelParams = new kernelParams_t;
      kernelParams->num_args = 0;
      (tconf->kernelArgsMap[this]).insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
    }
  }
  for (std::set<std::string>::iterator it = kernelNames.begin() ; it != kernelNames.end(); ++it) {
    if( kernelNameSet.count(*it) == 0 ) {
      kernelNameSet.insert(*it);
    }
  }
  }
#ifdef _THREAD_SAFETY
  pthread_mutex_unlock(&mutex_set_device_num);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_register_kernels(thread ID = %d)\n", threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_register_kernel_numargs(std::string kernel_name, int num_args, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_register_kernel_numargs(thread ID = %d)\n", threadID);
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  HostConf_t *tconf = getHostConf(threadID);
  kernelParams_t *kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
  if( kernelParams->num_args == 0 ) {
    if( num_args > 0 ) {
      kernelParams->num_args = num_args;
      kernelParams->kernelParams = (void**)malloc(sizeof(void*) * num_args);
      kernelParams->kernelParamsInfo = (int*)malloc(sizeof(int) * num_args);
    } else {
      fprintf(stderr, "[ERROR in BrisbaneDriver::HI_register_kernel_numargs(%s, %d)] num_args should be greater than zero.\n",kernel_name.c_str(), num_args);
      exit(1);
    }
  }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_register_kernel_numargs(thread ID = %d)\n", threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_register_kernel_arg(std::string kernel_name, int arg_index, size_t arg_size, void *arg_value, int arg_type, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_register_kernel_arg(thread ID = %d)\n", threadID);
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  HostConf_t *tconf = getHostConf(threadID);
  kernelParams_t * kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
  int err;
  if( arg_type == 0 ) {
    *(kernelParams->kernelParams + arg_index) = arg_value;
    *(kernelParams->kernelParamsInfo + arg_index) = (int) arg_size;
    //err = brisbane_kernel_setarg((brisbane_kernel)(tconf->kernelsMap.at(this).at(kernel_name)), arg_index, arg_size, arg_value);
  } else {
    HI_device_mem_handle_t tHandle;
    if( HI_get_device_mem_handle(*((void **)arg_value), &tHandle, tconf->threadID) == HI_success ) {
      *(kernelParams->kernelParams + arg_index) = tHandle.basePtr;
      *(kernelParams->kernelParamsInfo + arg_index) = 0 - tHandle.offset;
      //err = brisbane_kernel_setmem_off((brisbane_kernel)(tconf->kernelsMap.at(this).at(kernel_name)), arg_index, *((brisbane_mem*) &(tHandle.basePtr)), tHandle.offset, brisbane_rw);
      //if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
    } else {
		fprintf(stderr, "[ERROR in BrisbaneDriver::HI_register_kernel_arg()] Cannot find a device pointer to memory handle mapping; failed to add argument %d to kernel %s (Brisbane Device)\n", arg_index, kernel_name.c_str());
#ifdef _OPENARC_PROFILE_
		HI_print_device_address_mapping_entries(tconf->threadID);
#endif
		exit(1);
	}
  }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_register_kernel_arg(thread ID = %d)\n", threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_kernel_call(std::string kernel_name, size_t gridSize[3], size_t blockSize[3], int async, int num_waits, int *waits, int threadID) {
    const char* c_kernel_name = kernel_name.c_str();
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_kernel_call(%s, %d, thread ID = %d)\n",c_kernel_name, async, threadID);
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  HostConf_t *tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
  int err;
  brisbane_task task = threadTaskMap[threadID];
  int nestingLevel = threadTaskMapNesting[threadID];
  if( (task == NULL) && (nestingLevel == 0) ) {
    brisbane_task_create(&task);
  }
  size_t gws[3] = { gridSize[0] * blockSize[0], gridSize[1] * blockSize[1], gridSize[2] * blockSize[2] };
  kernelParams_t *kernelParams = tconf->kernelArgsMap.at(this).at(kernel_name);
  //brisbane_task_kernel_obsolete(task, tconf->kernelsMap.at(this).at(kernel_name), 3, off, idx, blockSize);
  brisbane_task_kernel(task, kernel_name.c_str(), 3, NULL, gws, blockSize, kernelParams->num_args, kernelParams->kernelParams, kernelParams->kernelParamsInfo);

#ifdef _OPENARC_PROFILE_
  if( HI_openarcrt_verbosity > 4 ) {
  	fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
  }
#endif
  if( nestingLevel == 0 ) {
    //brisbane_task_submit(task, brisbane_default, NULL, true);
    brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
	tconf->BTaskCnt++;	
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_kernel_call(%s, %d, thread ID = %d) submits a brisbane task\n",c_kernel_name, async, threadID);
    }
  if( HI_openarcrt_verbosity > 4 ) {
    fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
  }
#endif
    brisbane_task_release(task);
  }
#ifdef _OPENARC_PROFILE_
    if(tconf->KernelCNTMap.count(kernel_name) == 0) {
        tconf->KernelCNTMap[kernel_name] = 0;
    }        
    tconf->KernelCNTMap[kernel_name] += 1;
    if(tconf->KernelTimingMap.count(kernel_name) == 0) {
        tconf->KernelTimingMap[kernel_name] = 0.0;
    }        
    tconf->KernelTimingMap[kernel_name] += HI_get_localtime() - ltime;
#endif   

  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_kernel_call(%s, %d, thread ID = %d)\n",c_kernel_name, async, threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_synchronize( int forcedSync, int threadID ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_synchronize(forcedSync = %d, thread ID = %d)\n", forcedSync, threadID);
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  int err = BRISBANE_OK;
  if( forcedSync != 0 ) { 
  	err = brisbane_synchronize();
  }
  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_synchronize(forcedSync = %d, thread ID = %d)\n", forcedSync, threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::destroy(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::destroy()\n");
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		//fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
	NVIDIADeviceIDs.clear();
	AMDDeviceIDs.clear();
	GPUDeviceIDs.clear();
	CPUDeviceIDs.clear();
	FPGADeviceIDs.clear();
	PhiDeviceIDs.clear();
	DefaultDeviceIDs.clear();
#ifdef PRINT_TODO
	fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::destroy()\n");
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_malloc1D(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_malloc1D(hostPtr = %lx, asyncID = %d, size = %lu, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, count, threadID);
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  HostConf_t * tconf = getHostConf(threadID);
  int err;
  void * memHandle;
  if(HI_get_device_address(hostPtr, devPtr, NULL, NULL, asyncID, tconf->threadID) == HI_success ) {
#ifdef PRINT_DEBUG
    fprintf(stderr, "[%s:%d][%s] count[%lu]\n", __FILE__, __LINE__, __func__, count);
#endif
  } else {
#if VICTIM_CACHE_MODE > 0
    memPool_t *memPool = memPoolMap[tconf->threadID];
    std::multimap<size_t, void *>::iterator it = memPool->find(count);
	if( it != memPool->end()) {
      *devPtr = it->second;
      memPool->erase(it);
      current_mempool_size -= count;
      HI_set_device_address(hostPtr, *devPtr, count, asyncID, tconf->threadID);
	} else
#endif
	{
#if VICTIM_CACHE_MODE > 0
		if( current_mempool_size > tconf->max_mempool_size ) {
    		HI_device_mem_handle_t tHandle;
			void * tDevPtr;
			for( it = memPool->begin(); it != memPool->end(); ++it ) {
				tDevPtr = it->second;
    			if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      				err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      				if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      				HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      				free(tDevPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
    			}
			}	
			memPool->clear();
		}
#endif
      	err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
#if VICTIM_CACHE_MODE > 0
      	if (err != BRISBANE_OK) {
    		HI_device_mem_handle_t tHandle;
			void * tDevPtr;
			for( it = memPool->begin(); it != memPool->end(); ++it ) {
				tDevPtr = it->second;
    			if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      				err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      				if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      				HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      				free(tDevPtr);
                    current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    tconf->IDFreeCnt++;
                    tconf->CIDMemorySize -= it->first;
#endif
    			}
			}	
			memPool->clear();
      		err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
      		if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
		}
#endif
      	*devPtr = malloc(count);
      	HI_set_device_address(hostPtr, *devPtr, count, asyncID, tconf->threadID);
      	HI_set_device_mem_handle(*devPtr, memHandle, count, tconf->threadID);
#ifdef _OPENARC_PROFILE_
		tconf->IDMallocCnt++;
		tconf->IDMallocSize += count;
		tconf->CIDMemorySize += count;
		if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
			tconf->MIDMemorySize = tconf->CIDMemorySize;
		}    
#endif
    }
  }
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_malloc1D(hostPtr = %lx, asyncID = %d, size = %lu, thread ID = %d)\n",(long unsigned int)hostPtr, asyncID, count, threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_memcpy(%lu, thread ID = %d)\n", count, threadID);
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  HostConf_t * tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
  int err = BRISBANE_OK;
  switch( kind ) {
    case HI_MemcpyHostToHost:       memcpy(dst, src, count);                        break;
    case HI_MemcpyHostToDevice:
    {
      HI_device_mem_handle_t tHandle;
      if( HI_get_device_mem_handle(dst, &tHandle, tconf->threadID) == HI_success ) {
  		brisbane_task task = threadTaskMap[threadID];
  		int nestingLevel = threadTaskMapNesting[threadID];
  		if( (task == NULL) && (nestingLevel == 0) ) {
        	brisbane_task_create(&task);
		}
        brisbane_task_h2d(task, (brisbane_mem) tHandle.basePtr, tHandle.offset, count, (void*) src);
#ifdef _OPENARC_PROFILE_
  		if( HI_openarcrt_verbosity > 4 ) {
  			fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
		}
#endif
  		if( nestingLevel == 0 ) {
        	//brisbane_task_submit(task, brisbane_default, NULL, true);
  			brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
			tconf->BTaskCnt++;	
    	if( HI_openarcrt_verbosity > 2 ) {
        	fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_memcpy(%lu, thread ID = %d) submits a brisbane task\n", count, threadID);
    	}
  		if( HI_openarcrt_verbosity > 4 ) {
  			fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
		}
#endif
        	brisbane_task_release(task);
		}
      } else {
      }
      break;
    }
    case HI_MemcpyDeviceToHost:
    {
		HI_device_mem_handle_t tHandle;
		if( HI_get_device_mem_handle(src, &tHandle, tconf->threadID) == HI_success ) {
  			brisbane_task task = threadTaskMap[threadID];
  			int nestingLevel = threadTaskMapNesting[threadID];
  			if( (task == NULL) && (nestingLevel == 0) ) {
        		brisbane_task_create(&task);
			}
        	brisbane_task_d2h(task, (brisbane_mem) tHandle.basePtr, tHandle.offset, count, dst);
#ifdef _OPENARC_PROFILE_
  		if( HI_openarcrt_verbosity > 4 ) {
  			fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
		}
#endif
  			if( nestingLevel == 0 ) {
        		//brisbane_task_submit(task, brisbane_default, NULL, true);
  				brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
				tconf->BTaskCnt++;	
    			if( HI_openarcrt_verbosity > 2 ) {
        			fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_memcpy(%lu, thread ID = %d) submits a brisbane task\n", count, threadID);
    			}
  		if( HI_openarcrt_verbosity > 4 ) {
  			fprintf(stderr, "[%s:%d] %x\n", __FILE__, __LINE__, brisbane_default);
		}
#endif
        		brisbane_task_release(task);
			}
      } else {
      }
      break;
    }
    case HI_MemcpyDeviceToDevice:   fprintf(stderr, "[%s:%d][%s] not support D2D\n", __FILE__, __LINE__, __func__); break;
  }
  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
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

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_memcpy(%lu, thread ID = %d)\n", count, threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_malloc2D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

  return HI_success;
}

HI_error_t BrisbaneDriver::HI_malloc3D( const void *hostPtr, void** devPtr, size_t* pitch, size_t widthInBytes, size_t height, size_t depth, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

  return HI_success;
}

//[DEBUG] Implemented by Seyong; may be incorrect!
HI_error_t BrisbaneDriver::HI_free( const void *hostPtr, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_free(%d)\n", asyncID);
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  	HostConf_t * tconf = getHostConf(threadID);
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
  	void *devPtr;
	size_t size;
  	int err;
  	if( (HI_get_device_address(hostPtr, &devPtr, NULL, &size, asyncID, tconf->threadID) != HI_error) ) {
       //If this method is called for unified memory, memory deallocation
       //is skipped; unified memory will be deallocatedd only by 
       //HI_free_unified().
		if( hostPtr != devPtr ) {
#if VICTIM_CACHE_MODE > 0
            //We do not free the device memory; instead put it in the memory pool
            //and remove host-pointer-to-device-pointer mapping
            memPool_t *memPool = memPoolMap[tconf->threadID];
            memPool->insert(std::pair<size_t, void *>(size, devPtr));
            current_mempool_size += size;
            HI_remove_device_address(hostPtr, asyncID, tconf->threadID);
#else
    		HI_device_mem_handle_t tHandle;
    		if( HI_get_device_mem_handle(devPtr, &tHandle, tconf->threadID) == HI_success ) { 
      			err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      			if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
            	HI_remove_device_address(hostPtr, asyncID, tconf->threadID);
      			HI_remove_device_mem_handle(devPtr, tconf->threadID);
      			free(devPtr);
#ifdef _OPENARC_PROFILE_
      			tconf->IDFreeCnt++;
      			tconf->CIDMemorySize -= size;
#endif
    		}
#endif
		}
  	}

#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_free(%d)\n", asyncID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_pin_host_memory(const void* hostPtr, size_t size, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

  return HI_success;
}

void BrisbaneDriver::HI_unpin_host_memory(const void* hostPtr, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif

}

//[FIXME] HI_memcpy_async() temporariliy uses HI_memcpy().
HI_error_t BrisbaneDriver::HI_memcpy_async(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TOODO
  fprintf(stderr, "[%s:%d][%s] calls HI_memcpy() instead!\n", __FILE__, __LINE__, __func__);
#endif
  HI_error_t err = HI_memcpy(dst, src, count, kind, trType, threadID);
  return err;
}

HI_error_t BrisbaneDriver::HI_memcpy_asyncS(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy2D_async(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy2D_asyncS(void *dst, size_t dpitch, const void *src, size_t spitch, size_t widthInBytes, size_t height, HI_MemcpyKind_t kind, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

//[DEBUG] the current implementation does not exploit victim cache.
void BrisbaneDriver::HI_tempFree( void** tempPtr, acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
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
	if(  devType == acc_device_gpu || devType == acc_device_nvidia || 
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
  		void *devPtr;
		size_t size;
  		int err;
  		if( *tempPtr != 0 ) {
			//We do not free the device memory; instead put it in the memory pool 
			memPool_t *memPool = memPoolMap[tconf->threadID];
			sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
#if VICTIM_CACHE_MODE > 0
			if( tempMallocSize->count((const void *)*tempPtr) > 0 ) {
				size_t size = tempMallocSize->at((const void *)*tempPtr);
				memPool->insert(std::pair<size_t, void *>(size, *tempPtr));
            	current_mempool_size += size;
				tempMallocSize->erase((const void *)*tempPtr);
			} else 
#endif
			{
    			HI_device_mem_handle_t tHandle;
    			if( HI_get_device_mem_handle(*tempPtr, &tHandle, tconf->threadID) == HI_success ) { 
      				err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      				if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      				free(*tempPtr);
      				HI_remove_device_mem_handle(*tempPtr, tconf->threadID);
#ifdef _OPENARC_PROFILE_
            		tconf->IDFreeCnt++;
#endif
					if( tempMallocSize->count((const void *)*tempPtr) > 0 ) {
#ifdef _OPENARC_PROFILE_
						size_t size = tempMallocSize->at((const void *)*tempPtr);
						tconf->CIDMemorySize -= size;
#endif
						tempMallocSize->erase((const void *)*tempPtr);
					}
    			}
			}
		}
  	} else {
		if( *tempPtr != 0 ) {
			free(*tempPtr);
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
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_tempFree(tempPtr = %lx, devType = %d, thread ID = %d)\n",  (long unsigned int)(*tempPtr), devType, threadID);
    }
#endif
}

//[DEBUG] the current implementation does not exploit victim cache.
void BrisbaneDriver::HI_tempMalloc1D( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_tempMalloc1D()\n");
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
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
	if(  devType == acc_device_gpu || devType == acc_device_nvidia || 
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
#if VICTIM_CACHE_MODE > 0
        memPool_t *memPool = memPoolMap[tconf->threadID];
        std::multimap<size_t, void *>::iterator it = memPool->find(count);
        if (it != memPool->end()) {
#ifdef _OPENARC_PROFILE_
            if( HI_openarcrt_verbosity > 2 ) {
                fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_tempMalloc1D(%lu) reuses memories in the memPool\n", count);
            }
#endif
            *tempPtr = it->second;
            memPool->erase(it);
            current_mempool_size -= count;
            (*tempMallocSize)[(const void *)*tempPtr] = count;
      		//HI_set_device_address(hostPtr, *tempPtr, count, asyncID, tconf->threadID);
        } else 
#endif
		{
			int err;
			void * memHandle;
#if VICTIM_CACHE_MODE > 0
			if( current_mempool_size > tconf->max_mempool_size ) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_tempMalloc1D(%lu) releases memories in the memPool\n", count);
                }
#endif
    			HI_device_mem_handle_t tHandle;
				void * tDevPtr;
                for (it = memPool->begin(); it != memPool->end(); ++it) {
					tDevPtr = it->second;
    				if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      					err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      					if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      					HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      					free(tDevPtr);
                        current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    	tconf->IDFreeCnt++;
                    	tconf->CIDMemorySize -= it->first;
#endif
    				}
                }
				memPool->clear();
			}
#endif
			err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
#if VICTIM_CACHE_MODE > 0
			if (err != BRISBANE_OK) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_tempMalloc1D(%lu) releases memories in the memPool\n", count);
                }
#endif
    			HI_device_mem_handle_t tHandle;
				void * tDevPtr;
                for (it = memPool->begin(); it != memPool->end(); ++it) {
					tDevPtr = it->second;
    				if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      					err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      					if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      					HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      					free(tDevPtr);
                        current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    	tconf->IDFreeCnt++;
                    	tconf->CIDMemorySize -= it->first;
#endif
    				}
                }
				memPool->clear();
      			err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
      			if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
			}
#endif
			*tempPtr = malloc(count);
			HI_set_device_mem_handle(*tempPtr, memHandle, count, tconf->threadID);
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
		}
	} else {
		*tempPtr = malloc(count);
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
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_tempMalloc1D()\n");
    }
#endif
}

//[DEBUG] the current implementation does not exploit victim cache.
void BrisbaneDriver::HI_tempMalloc1D_async( void** tempPtr, size_t count, acc_device_t devType, HI_MallocKind_t flags, int asyncID, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_tempMalloc1D_async()\n");
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
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
	if(  devType == acc_device_gpu || devType == acc_device_nvidia || 
    devType == acc_device_radeon || devType == acc_device_xeonphi || 
    devType == acc_device_altera || devType == acc_device_altera_emulator ||
    devType == acc_device_current) {
        sizemap_t *tempMallocSize = tempMallocSizeMap[tconf->threadID];
#if VICTIM_CACHE_MODE > 0
        memPool_t *memPool = memPoolMap[tconf->threadID];
        std::multimap<size_t, void *>::iterator it = memPool->find(count);
        if (it != memPool->end()) {
#ifdef _OPENARC_PROFILE_
            if( HI_openarcrt_verbosity > 2 ) {
                fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_tempMalloc1D_async(%lu) reuses memories in the memPool\n", count);
            }
#endif
            *tempPtr = it->second;
            memPool->erase(it);
            current_mempool_size -= it->first;
            (*tempMallocSize)[(const void *)*tempPtr] = count;
      		//HI_set_device_address(hostPtr, *tempPtr, count, asyncID, tconf->threadID);
        } else 
#endif
		{
			int err;
			void * memHandle;
#if VICTIM_CACHE_MODE > 0
			if( current_mempool_size > tconf->max_mempool_size ) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_malloc1D_async(%lu) releases memories in the memPool\n", count);
                }
#endif
    			HI_device_mem_handle_t tHandle;
				void * tDevPtr;
                for (it = memPool->begin(); it != memPool->end(); ++it) {
					tDevPtr = it->second;
    				if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      					err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      					if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      					HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      					free(tDevPtr);
                        current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    	tconf->IDFreeCnt++;
                    	tconf->CIDMemorySize -= it->first;
#endif
    				}
                }
				memPool->clear();
			}
#endif
			err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
#if VICTIM_CACHE_MODE > 0
			if (err != BRISBANE_OK) {
#ifdef _OPENARC_PROFILE_
                if( HI_openarcrt_verbosity > 2 ) {
                    fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_malloc1D_async(%lu) releases memories in the memPool\n", count);
                }
#endif
    			HI_device_mem_handle_t tHandle;
				void * tDevPtr;
                for (it = memPool->begin(); it != memPool->end(); ++it) {
					tDevPtr = it->second;
    				if( HI_get_device_mem_handle(tDevPtr, &tHandle, tconf->threadID) == HI_success ) { 
      					err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      					if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      					HI_remove_device_mem_handle(tDevPtr, tconf->threadID);
      					free(tDevPtr);
                        current_mempool_size -= it->first;
#ifdef _OPENARC_PROFILE_
                    	tconf->IDFreeCnt++;
                    	tconf->CIDMemorySize -= it->first;
#endif
    				}
                }
				memPool->clear();
      			err = brisbane_mem_create(count, (brisbane_mem*) &memHandle);
      			if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
			}
#endif
			*tempPtr = malloc(count);
			HI_set_device_mem_handle(*tempPtr, memHandle, count, tconf->threadID);
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
		}
	} else {
		*tempPtr = malloc(count);
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
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_tempMalloc1D_async()\n");
    }
#endif
}

// Experimental API to support unified memory //
HI_error_t BrisbaneDriver::HI_malloc1D_unified(const void *hostPtr, void **devPtr, size_t count, int asyncID, HI_MallocKind_t flags, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy_unified(void *dst, const void *src, size_t count, HI_MemcpyKind_t kind, int trType, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_free_unified( const void *hostPtr, int asyncID, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

int BrisbaneDriver::HI_get_num_devices(acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_get_num_devices()\n");
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  int count = 0;
	switch (devType) {
		case acc_device_default: { count = DefaultDeviceIDs.size(); break; }
		case acc_device_host: { count = CPUDeviceIDs.size(); break; }
		case acc_device_not_host: { count = DefaultDeviceIDs.size(); break; }
		case acc_device_nvidia: { count = NVIDIADeviceIDs.size(); break; }
		case acc_device_radeon: { count = AMDDeviceIDs.size(); break; }
		case acc_device_gpu: { count = GPUDeviceIDs.size(); break; }
		case acc_device_xeonphi: { count = PhiDeviceIDs.size(); break; }
		case acc_device_current: { count = DefaultDeviceIDs.size(); break; }
		case acc_device_altera: { count = FPGADeviceIDs.size(); break; }
		case acc_device_altera_emulator: { count = FPGADeviceIDs.size(); break; }
		default: { count = DefaultDeviceIDs.size(); break; }
	}
#ifdef _OPENARC_PROFILE_
  if( HI_openarcrt_verbosity > 4 ) {
  	fprintf(stderr, "[%s:%d][%s] count[%d]\n", __FILE__, __LINE__, __func__, count);
  }
#endif

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_get_num_devices()\n");
    }
#endif
  return count;
}

int BrisbaneDriver::HI_get_num_devices_init(acc_device_t devType, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_get_num_devices_init()\n");
    }
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  int err;
  err = brisbane_init(NULL, NULL, 1);
  if (err == BRISBANE_OK) brisbane_register_command(0xdeadcafe, brisbane_nvidia, bind_tex_handler);
  else if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
  int ndevs = 0;
  err = brisbane_device_count(&ndevs);
  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
  int defaultType = brisbane_default;
  for(int i=0; i<ndevs; i++) {
  	int type;
  	size_t size;
    err = brisbane_device_info(i, brisbane_type, &type, &size);  
  	if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
    if( i==0 ) {
      defaultType = type;
    }
    if( type == defaultType ) {
      DefaultDeviceIDs.push_back(i);
    }
    if( type == brisbane_nvidia ) {
      NVIDIADeviceIDs.push_back(i);
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_amd ) {
      AMDDeviceIDs.push_back(i);
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_gpu ) {
      GPUDeviceIDs.push_back(i);
    } else if( type == brisbane_cpu ) {
      CPUDeviceIDs.push_back(i);
    } else if( type == brisbane_fpga ) {
      FPGADeviceIDs.push_back(i);
    } else if( type == brisbane_phi ) {
      PhiDeviceIDs.push_back(i);
    }
  }

  int count = 1;
  if (err != BRISBANE_OK) { count = 0; }
#ifdef _OPENARC_PROFILE_
  if( HI_openarcrt_verbosity > 4 ) {
  	fprintf(stderr, "[%s:%d][%s] count[%d]\n", __FILE__, __LINE__, __func__, count);
  }
#endif

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_get_num_devices_init()\n");
    }
#endif
  return count;
}

void BrisbaneDriver::HI_malloc(void **devPtr, size_t size, HI_MallocKind_t flags, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_malloc()\n");
    }    
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
  HostConf_t * tconf = getHostConf(threadID);
  int err;
  void * memHandle;
  err = brisbane_mem_create(size, (brisbane_mem*) &memHandle);
  if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
  *devPtr = malloc(size);
  HI_set_device_mem_handle(*devPtr, memHandle, size, tconf->threadID);
#ifdef _OPENARC_PROFILE_
    tconf->IDMallocCnt++;
    tconf->IDMallocSize += size;
    tconf->CIDMemorySize += size;
    if( tconf->MIDMemorySize < tconf->CIDMemorySize ) {
        tconf->MIDMemorySize = tconf->CIDMemorySize;
    }    
#endif
#ifdef _OPENARC_PROFILE_
    tconf->totalMallocTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_malloc()\n");
    }    
#endif
}

void BrisbaneDriver::HI_free(void *devPtr, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_free(thread ID = %d)\n",  threadID);
    }    
  	if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
#ifdef _OPENARC_PROFILE_
    double ltime = HI_get_localtime();
#endif
  HostConf_t * tconf = getHostConf(threadID);
  void *devPtr2;
  int err;
  size_t memSize = 0;
  if( (HI_get_device_address(devPtr, &devPtr2, NULL, &memSize, DEFAULT_QUEUE+tconf->asyncID_offset, tconf->threadID) == HI_error) || (devPtr != devPtr2) ) {
    HI_device_mem_handle_t tHandle;
    if( HI_get_device_mem_handle(devPtr, &tHandle, tconf->threadID) == HI_success ) { 
      err = brisbane_mem_release((brisbane_mem) tHandle.basePtr);
      if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
      free(devPtr);
      HI_remove_device_mem_handle(devPtr, tconf->threadID);
#ifdef _OPENARC_PROFILE_
      tconf->IDFreeCnt++;
      tconf->CIDMemorySize -= memSize;
#endif
    }
  }
#ifdef _OPENARC_PROFILE_
    tconf->totalFreeTime += HI_get_localtime() - ltime;
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_free(thread ID = %d)\n",  threadID);
    }    
#endif
}

HI_error_t BrisbaneDriver::createKernelArgMap(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::createKernelArgMap(thread ID = %d)\n", threadID);
    }
    if( HI_openarcrt_verbosity > 4 ) {
  		fprintf(stderr, "[%s:%d][%s]\n", __FILE__, __LINE__, __func__);
	}
#endif
  int err;
  HostConf_t * tconf = getHostConf(threadID);
#ifdef _THREAD_SAFETY
  pthread_mutex_lock(&mutex_clContext);
#else
#ifdef _OPENMP
#pragma omp critical(clContext_critical)
#endif
#endif
  {
  std::map<std::string, brisbane_kernel> kernelMap;
  std::map<std::string, kernelParams_t*> kernelArgs;
  for(std::set<std::string>::iterator it=kernelNameSet.begin(); it!=kernelNameSet.end(); ++it) {
    brisbane_kernel kernel;
    const char *kernelName = (*it).c_str();
    err = brisbane_kernel_create(kernelName, &kernel);
    if (err != BRISBANE_OK) fprintf(stderr, "[%s:%d][%s] error[%d]\n", __FILE__, __LINE__, __func__, err);
    kernelMap[*it] = kernel;

    kernelParams_t *kernelParams = new kernelParams_t;
    kernelParams->num_args = 0;
    kernelArgs.insert(std::pair<std::string, kernelParams_t*>(std::string(kernelName), kernelParams));
  }

  tconf->kernelArgsMap[this] = kernelArgs;
  tconf->kernelsMap[this]=kernelMap;
  }
#ifdef _THREAD_SAFETY
  pthread_mutex_unlock(&mutex_clContext);
#endif

	int thread_id = tconf->threadID;
  if (masterHandleTable.count(thread_id) == 0) {
    masterAddressTableMap[thread_id] = new addresstable_t();
    masterHandleTable[thread_id] = new addressmap_t();
    postponedFreeTableMap[thread_id] = new asyncfreetable_t();
    postponedTempFreeTableMap[thread_id] = new asynctempfreetable_t();
    postponedTempFreeTableMap2[thread_id] = new asynctempfreetable2_t();
    memPoolMap[thread_id] = new memPool_t();
    tempMallocSizeMap[thread_id] = new sizemap_t();
  	threadTaskMap[thread_id] = NULL;
  	threadTaskMapNesting[thread_id] = 0;
    threadHostMemFreeMap[thread_id] = new pointerset_t();
  }

#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::createKernelArgMap(thread ID = %d)\n", threadID);
    }
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_bind_tex(std::string texName,  HI_datatype_t type, const void *devPtr, size_t size, int threadID) {
  HostConf_t * tconf = getHostConf(threadID);
  HI_device_mem_handle_t tHandle;
  const char* name = texName.c_str();
  brisbane_task task = threadTaskMap[threadID];
  int nestingLevel = threadTaskMapNesting[threadID];
  pointerset_t *tmpHostMemSet = threadHostMemFreeMap[threadID];
  if( HI_get_device_mem_handle(devPtr, &tHandle, tconf->threadID) == HI_success ) {
  	if( (task == NULL) && (nestingLevel == 0) ) {
    	brisbane_task_create(&task);
	}
    void* tmp = malloc(size);
    brisbane_task_h2d(task, (*((brisbane_mem*) &tHandle)), 0, size, tmp);
  	if( nestingLevel == 0 ) {
    	brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
		tconf->BTaskCnt++;	
    	if( HI_openarcrt_verbosity > 2 ) {
        	fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_bind_tex(%s, thread ID = %d) submits a brisbane task\n", name, threadID);
    	}
#endif
    	free(tmp);
		//[DEBUG] Why don't release the task?
    	//brisbane_task_release(task);
	} else {
		tmpHostMemSet->insert((const void *)tmp);
	}
  }
  if( HI_get_device_mem_handle(devPtr, &tHandle, tconf->threadID) == HI_success ) {
  }
  void* dptr = (*(void**) &tHandle);

  size_t name_len = strlen(name) + 1;
  int tp = (type == HI_int) ? brisbane_int : (type == HI_float) ? brisbane_float : -1;
  size_t params_size = sizeof(name_len) + name_len + sizeof(tp) + sizeof(dptr) + sizeof(size);
  char* params = (char*) malloc(params_size);
  int off = 0;
  memcpy(params + off, &name_len, sizeof(name_len));
  off += sizeof(name_len);
  memcpy(params + off, name, name_len);
  off += name_len;
  memcpy(params + off, &tp, sizeof(tp));
  off += sizeof(tp);
  memcpy(params + off, &dptr, sizeof(dptr));
  off += sizeof(dptr);
  memcpy(params + off, &size, sizeof(size));
  off += sizeof(size);

  //brisbane_task task;
  if( nestingLevel == 0 ) {
  	brisbane_task_create(&task);
  }
  brisbane_task_custom(task, 0xdeadcafe, params, params_size);
  if( nestingLevel == 0 ) {
    //brisbane_task_submit(task, brisbane_default, NULL, true);
    brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
	tconf->BTaskCnt++;	
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_bind_tex(%s, thread ID = %d) submits a brisbane task2\n", name, threadID);
    }
#endif
	//[DEBUG] Why don't release the task?
    //brisbane_task_release(task);
  }

  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_memcpy_const_async(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int async, int num_waits, int *waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

HI_error_t BrisbaneDriver::HI_present_or_memcpy_const(void *hostPtr, std::string constName, HI_MemcpyKind_t kind, size_t count, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return HI_success;
}

void BrisbaneDriver::HI_set_async(int asyncId, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_set_context(int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_set_context(thread ID = %d)\n", threadID);
    }
#endif
#ifdef PRINT_TODO
  //fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_set_context(thread ID = %d)\n", threadID);
    }
#endif
}

void BrisbaneDriver::HI_wait(int arg, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_wait_ifpresent(int arg, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_waitS1(int arg, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_waitS2(int arg, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_wait_all(int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_wait_async(int arg, int async, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_wait_async_ifpresent(int arg, int async, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_wait_all_async(int async, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

int BrisbaneDriver::HI_async_test(int asyncId, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return 1;
}

int BrisbaneDriver::HI_async_test_ifpresent(int asyncId, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return 1;
}

int BrisbaneDriver::HI_async_test_all(int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
  return 1;
}

void BrisbaneDriver::HI_wait_for_events(int async, int num_waits, int* waits, int threadID) {
#ifdef PRINT_TODO
  fprintf(stderr, "[%s:%d][%s] Not Implemented!\n", __FILE__, __LINE__, __func__);
#endif
}

void BrisbaneDriver::HI_enter_subregion(const char *label, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_enter_subregion (label = %s, thread ID = %d)\n", label, threadID);
    }
#endif
  	brisbane_task task = threadTaskMap[threadID];
  	int nestingLevel = threadTaskMapNesting[threadID];
	if( nestingLevel == 0 ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_enter_subregion (label = %s, thread ID = %d) creates a brisbane task\n", label, threadID);
    }
#endif
  		brisbane_task_create(&task);
  		threadTaskMap[threadID] = task;
	}
	threadTaskMapNesting[threadID] = ++nestingLevel;
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_enter_subregion (label = %s, thread ID = %d)\n", label, threadID);
    }
#endif
}

void BrisbaneDriver::HI_exit_subregion(const char *label, int threadID) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tenter BrisbaneDriver::HI_exit_subregion (label = %s, thread ID = %d)\n", label, threadID);
    }
#endif
  	brisbane_task task = threadTaskMap[threadID];
  	int nestingLevel = threadTaskMapNesting[threadID];
    pointerset_t *tmpHostMemSet = threadHostMemFreeMap[threadID];
	nestingLevel--;
	if( nestingLevel <= 0 ) {
		if( task != NULL ) {
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\tBrisbaneDriver::HI_exit_subregion (label = %s, thread ID = %d) submits a brisbane task\n", label, threadID);
    }
#endif
  			HostConf_t *tconf = getHostConf(threadID);
    		brisbane_task_submit(task, HI_getBrisbaneDeviceID(tconf->acc_device_type_var,tconf->user_set_device_type_var, tconf->acc_device_num_var), NULL, true);
#ifdef _OPENARC_PROFILE_
			tconf->BTaskCnt++;	
#endif
    		brisbane_task_release(task);
			if( !(tmpHostMemSet->empty()) ) {
				for(std::set<const void *>::iterator it=tmpHostMemSet->begin(); it!=tmpHostMemSet->end(); ++it) {
					free((void *)*it);
				}
				tmpHostMemSet->clear();
			}
		}
  		threadTaskMap[threadID] = NULL;
		threadTaskMapNesting[threadID] = 0;
	} else {
		threadTaskMapNesting[threadID] = nestingLevel;
	}
#ifdef _OPENARC_PROFILE_
    if( HI_openarcrt_verbosity > 2 ) {
        fprintf(stderr, "[OPENARCRT-INFO]\t\texit BrisbaneDriver::HI_exit_subregion (label = %s, thread ID = %d)\n", label, threadID);
    }
#endif
}

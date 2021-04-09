#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <algorithm>

#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0 || OPENARC_ARCH == 6
#include <cuda.h>
#include <cuda_runtime.h>
#else
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#ifndef PRINT_LOG
#define PRINT_LOG 0
#endif

#include <sstream>

#include "openacc.h"

#define MAX_SOURCE_SIZE (0x100000)

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
static const char *ALTERA_EMUL = "ALTERA_EMUL";
static const char *HOST = "HOST";
static const char *INTELGPU = "INTELGPU";

static acc_device_t user_set_device_type_var = acc_device_default;
static acc_device_t acc_device_type_var = acc_device_default;

//Function to convert input string to uppercases.
static char *convertToUpper(char *str) {
    char *newstr, *p;
    p = newstr = strdup(str);
    //while((*p++=toupper(*p)));
    while((*p=toupper(*p))) {p++;} //Changed to unambiguous way
    return newstr;
}

char * deblank(char *str) {
	char *out = str, *put = str; 
	for(; *str != '\0'; ++str) {
		if((*str != ' ') && (*str != ':') && (*str != '(') && (*str != ')') && (*str != '[') && (*str != ']') && (*str != '<') && (*str != '>')) {
			*put++ = *str; 
		}    
	}    
	*put = '\0';
	return out; 
}   

const char* get_device_type_string( acc_device_t devtype ) {
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
        case acc_device_intelgpu: { str = "acc_device_intelgpu"; break; }
        default: { str = "UNKNOWN TYPE"; break; }
    }    
    return str.c_str(); 
}

void setDefaultDevice() {
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
            //acc_device_type_var = acc_device_gpu;
            acc_device_type_var = acc_device_nvidia;
        } else if( strcmp(envVarU, RADEON) == 0 ) {
			user_set_device_type_var = acc_device_radeon;
            //acc_device_type_var = acc_device_gpu;
            acc_device_type_var = acc_device_radeon;
        } else if( strcmp(envVarU, INTELGPU) == 0 ) {
			user_set_device_type_var = acc_device_intelgpu;
            //acc_device_type_var = acc_device_gpu;
            acc_device_type_var = acc_device_intelgpu;
        } else if( strcmp(envVarU, "ACC_DEVICE_DEFAULT") == 0 ) {
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
        } else if( strcmp(envVarU, ALTERA) == 0 ) {
			user_set_device_type_var = acc_device_altera;
            acc_device_type_var = acc_device_altera;
        } else if( strcmp(envVarU, ALTERA_EMUL) == 0 ) {
			user_set_device_type_var = acc_device_altera_emulator;
            acc_device_type_var = acc_device_altera_emulator;
        } else if( strcmp(envVarU, XEONPHI) == 0 ) {
			user_set_device_type_var = acc_device_xeonphi;
            acc_device_type_var = acc_device_xeonphi;
        } else if( strcmp(envVarU, "ACC_DEVICE_NONE") == 0 ) {
			user_set_device_type_var = acc_device_none;
            acc_device_type_var = acc_device_none;
        } else if( (strcmp(envVarU, "ACC_DEVICE_HOST") == 0) || (strcmp(envVarU, HOST) == 0) ) {
			user_set_device_type_var = acc_device_host;
            acc_device_type_var = acc_device_host;
        } else if( strcmp(envVarU, "ACC_DEVICE_NOT_HOST") == 0 ) {
			user_set_device_type_var = acc_device_not_host;
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
			user_set_device_type_var = acc_device_none;
            acc_device_type_var = acc_device_none;
        }
        free(envVarU);
    }
}

int main (int argc, char * argv[]){
	std::string fileNameBase;
	if( argc == 2 ) {
		fileNameBase = argv[1];
	} else {
		fileNameBase = "openarc_kernel";
	}
	setDefaultDevice();
#if !defined(OPENARC_ARCH) || OPENARC_ARCH == 0 || OPENARC_ARCH == 6
	//Generate ptx files for .cu, only if nvcc is found on the system
	if (system("which nvcc")==0){
		CUresult err;
		int major, minor;
		int max_threads_per_block;
		CUdevice cuDevice;
		CUcontext cuContext;
		CUmodule cuModule;
		int numDevices = 0;
		if( cudaGetDeviceCount(&numDevices) != cudaSuccess ) {
			fprintf(stderr, "[ERROR in CUDA binary creation] no available NVIDIA GPU found!; exit!\n");
			exit(1);
		}
		
		for(int i=0 ; i < numDevices; i++) {
			cuDeviceGet(&cuDevice, i);
			#if CUDA_VERSION >= 5000
			cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
			cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
			#else
				cuDeviceComputeCapability(&major, &minor, cuDevice);
			#endif
			cuDeviceGetAttribute (&max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, cuDevice);

			std::stringstream ss;
			ss << major;
			ss << minor;
			std::string version = ss.str();
#if OPENARC_ARCH == 6
			std::string ptxName = std::string("kernel.ptx");
#else
			std::string ptxName = fileNameBase + std::string("_") + version + std::string(".ptx");
#endif
			fprintf(stderr, "[INFO] Create %s for device %d\n", ptxName.c_str(), i);
			fprintf(stderr, "[INFO] Max # of threads per thread block for device %d: %d\n", i, max_threads_per_block);
			std::string command = std::string("nvcc $OPENARC_JITOPTION -arch=sm_") + version + std::string(" ") + fileNameBase + std::string(".cu -ptx -o ") + ptxName;
			system(command.c_str());
		}
	} else {
		fprintf(stderr, "[ERROR in CUDA binary creation] cannot find the NVIDIA CUDA compiler (nvcc) \n");
	}

#else

	cl_platform_id clPlatform;
	cl_device_id clDevice;
	cl_context clContext;
	cl_command_queue clQueue;
	cl_program clProgram;
	char *platformName;
	cl_uint numDevices;
	cl_uint num_platforms = 0;
	cl_int err;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to get the number of platforms available on this device\n");
        exit(1);
    }    
    if ( num_platforms <= 0 ) {
        fprintf(stderr, "[ERROR] Failed to find any available platform on this device\n");
        exit(1);
    }    
    fprintf(stderr, "[INFO] Number of available platforms on this device: %d\n", num_platforms);
    cl_platform_id* platforms = new cl_platform_id[num_platforms];
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to get the list of platforms IDs available on this device\n");
        exit(1);
    }
	cl_device_id *devices;
	bool foundPlatform = false;
	for( unsigned i=0; i<num_platforms; i++ ) {
  		size_t sz; 
		clPlatform = platforms[i];
  		err = clGetPlatformInfo(clPlatform, CL_PLATFORM_NAME, 0, NULL, &sz);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR] Failed to get the platform name size\n");
			exit(1);
        }
        char* namestr = new char[sz];
  		err = clGetPlatformInfo(clPlatform, CL_PLATFORM_NAME, sz, namestr, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "[ERROR] Failed to get the platform name\n");
			exit(1);
        }
		platformName = namestr;
#ifdef PRINT_DEBUG
		fprintf(stderr, "[DEBUG] Found OpenCL platform[%d]: %s\n",i , platformName);
#endif
		std::string name = namestr;
		std::transform(name.begin(), name.end(), name.begin(), tolower);
		std::string search;
        if(acc_device_type_var == acc_device_altera) {
			search = "fpga";
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
					if( err != CL_SUCCESS ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
		} else if(acc_device_type_var == acc_device_altera_emulator) {
			search = "emulation";
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        			if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
        } else if( acc_device_type_var == acc_device_xeonphi ) {
			search = "intel";
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ACCELERATOR, numDevices, devices, NULL);
        			if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
        } else if( acc_device_type_var == acc_device_intelgpu ) {
			search = "intel";
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        			if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
		} else if( acc_device_type_var == acc_device_nvidia ) {
			search = "nvidia";
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        			if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
		} else if( acc_device_type_var == acc_device_radeon ) {
			search = "amd";
			if( name.find(search) == std::string::npos ) {
				search = "advanced";
			}
			if( name.find(search) == std::string::npos ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
					err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        			if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
						delete [] namestr;
						platformName = NULL;
						continue;
					} else {
						foundPlatform = true;
						break;
					}
				}
			}
		} else if(acc_device_type_var == acc_device_gpu) {
			err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        	if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					foundPlatform = true;
					break;
				}
			}
		} else if(acc_device_type_var == acc_device_host) {
			err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        	if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
				delete [] namestr;
				platformName = NULL;
				continue;
			} else {
				devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
				err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
        		if ( (err != CL_SUCCESS) || (numDevices <= 0) ) {
					delete [] namestr;
					platformName = NULL;
					continue;
				} else {
					foundPlatform = true;
					break;
				}
			}
		} else {
			delete [] namestr;
			platformName = NULL;
			break;
		}
	}
			
    if ( !foundPlatform ) {
		fprintf(stderr, "[ERROR] Failed to get device IDs for device type %s\n", get_device_type_string(acc_device_type_var));
        if (err != CL_SUCCESS) {
            fprintf(stderr, "                                              OpenCL error (%d)\n", err);
		}
		if(platformName != NULL) {
            fprintf(stderr, "                                              Current platform: %s\n",platformName);
		}
		exit(1);
	} else {
		if(platformName != NULL) {
            fprintf(stderr, "[INFO] Current platform: %s\n       Target device type: %s\n",platformName, get_device_type_string(acc_device_type_var));
		}
	}
	delete [] platforms;

	for(int i=0; i< numDevices; i++) {
		clDevice = devices[i];
		
		FILE *fp;
		char *source_str;
		size_t source_size;
		std::string outFile = fileNameBase + std::string(".cl");
		const char *filename = outFile.c_str();
		fp = fopen(filename, "r");
		if (!fp) {
			fprintf(stderr, "[INFO in OpenCL binary creation] Failed to read the kernel file %s, so skipping binary generation for OpenCL devices %d\n", filename, i);
			exit(1);
		}
		source_str = (char*)malloc(MAX_SOURCE_SIZE);
		source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
		fclose( fp );

		cl_int err;
		clContext = clCreateContext( NULL, 1, &clDevice, NULL, NULL, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL context with error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		clQueue = clCreateCommandQueue(clContext, clDevice, 0, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL queue with error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		size_t max_work_group_size;
		clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
		
		char cBuffer[1024];
		char *cBufferN;
		clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		cBufferN = deblank(cBuffer);
		
		std::string binaryName = fileNameBase + std::string("_") + cBufferN + std::string(".clbin");
		
		fprintf(stderr, "[INFO] Create %s for device %d\n", binaryName.c_str(), i);
		fprintf(stderr, "[INFO] Max # of work-items in a work-group for device %d: %lu\n", i, max_work_group_size);
		clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&source_str, (const size_t *)&source_size, &err);
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to create OPENCL program with error %d (OPENCL GPU)\n", err);
			exit(1);
		}
		
		char *envVar;
		envVar = getenv("OPENARC_JITOPTION");
		err = clBuildProgram(clProgram, 1, &clDevice, envVar, NULL, NULL);
#if PRINT_LOG == 0
		if(err != CL_SUCCESS)
		{
				printf("[ERROR in OpenCL binary creation] Error in clBuildProgram, Line %u in file %s : %d!!!\n\n", __LINE__, __FILE__, err);
				if (err == CL_BUILD_PROGRAM_FAILURE)
				{
						// Determine the size of the log
						size_t log_size;
						clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

						// Allocate memory for the log
						char *log = (char *) malloc(log_size);

						// Get the log
						clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

						// Print the log
						printf("%s\n", log);
				}
				exit(1);
		}
#else
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char *log = (char *) malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
#endif
		size_t size;
		err = clGetProgramInfo( clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL );
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to get OPENCL program info error %d (OPENCL GPU)\n", err);
			exit(1);
		}

		unsigned char * binary = new unsigned char [size];
		
		err = clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL);
		
		if(err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR in OpenCL binary creation] failed to dump OPENCL program binary error %d (OPENCL GPU)\n", err);
			exit(1);
		}
		
		FILE * fpbin = fopen(binaryName.c_str(), "wb" );
		fwrite(binary, 1 , size, fpbin);
		fclose(fpbin);
		delete[] binary;
	}	

#endif

}

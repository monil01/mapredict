#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "main.h"
#include "../rt.h"

#define min(a, b) ((a<b)?a:b)
#define max(a, b) ((a>b)?a:b)

#define VERBOSE

void parse(int argc, char* argv[], struct user_parameters* params)
{
    int i = 0;
    if (argc == 1){
    	char commandLine[255];
    	FILE* runFile = fopen("run","r");
    	if (runFile != NULL){
    		fgets (commandLine, 255, runFile);
    		argv = (char**)malloc(sizeof(char*)*30);
    		char *param = strtok(commandLine," ");
    		while (param != NULL){
    			argv[i]=new char(10);
    			strcpy(argv[i],param);
    			i++;

    			param = strtok(NULL, " ");
    		}
    	}
    	argc = i;
    }

    for(i=1; i<argc; i++) {
        if(!strcmp(argv[i], "-c"))
            params->check = 1;
        else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("----------------------------------------------\n");
            printf("-                Juggler                     -\n");
            printf("-   OpenMP 4.5 GPU Runtime Task FRamework    -\n");
            printf("----------------------------------------------\n");
            printf("-h, --help : Show help information\n");
            printf("-c : Ask to check result\n");
            printf("-w : Number of workers\n");
            printf("-r : Number ot timestep iterations\n");
            printf("-n : Matrix size (one dimension) \n");
            printf("-b : Block size (one dimension) \n");
            printf("-w : Worker count (default=nSMs)\n");
            //printf("-ib : Internal Block size\n");
            printf("-d : Operating Mode [1] Tasked Execution, [2] Global Barriers\n");
            exit(EXIT_SUCCESS);
        } else if(!strcmp(argv[i], "-i")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-i requires a number\n");
                exit(EXIT_FAILURE);
            }
				} else if(!strcmp(argv[i], "-a")) {
            if (++i < argc)
								strcpy(params->app, argv[i]);
				} else if(!strcmp(argv[i], "-s")) {
            if (++i < argc)
						strcpy(params->sched, argv[i]);
        } else if(!strcmp(argv[i], "-w")) {
            if (++i < argc)
                params->niter = atoi(argv[i]);
            else {
                fprintf(stderr, "-w requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-r")) {
            if (++i < argc)
                params->titer = atoi(argv[i]);
            else {
                fprintf(stderr, "-r requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-n")) {
            if (++i < argc)
                params->matrix_size = atoi(argv[i]);
            else {
                fprintf(stderr, "-n requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-b")) {
            if (++i < argc)
                params->blocksize = atoi(argv[i]);
            else {
                fprintf(stderr, "-b requires a number\n");
                exit(EXIT_FAILURE);
            }
        } else if(!strcmp(argv[i], "-f")) {
               params->file = 1;
        } else if(!strcmp(argv[i], "-d")) {
            if (++i < argc)
                params->mode = atoi(argv[i]);
            else {
                fprintf(stderr, "-d requires a mode selection\n");
                exit(EXIT_FAILURE);
            }
        } else
            fprintf(stderr, "Unknown parameter : %s\n", argv[i]);
    }
}

int comp (const void * elem1, const void * elem2)
{
    double f = *((double*)elem1);
    double s = *((double*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

int main(int argc, char* argv[])
{
    int num_threads = 1;
    struct user_parameters params;
    memset(&params, 0, sizeof(params));

	int device;
	cudaGetDevice(&device);
	struct cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);
    //printf("device %s\n", props.name);

    checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    /* default value */
    params.niter = 1;

    parse(argc, argv, &params);

// get Number of thread if OpenMP is activated
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    num_threads = omp_get_num_threads(); //TODO: how to translate this?
#endif

    double mean = 0.0;
    double meansqr = 0.0;
    double min_ = DBL_MAX;
    double max_ = -1;
    double* all_times = (double*)malloc(sizeof(double) * params.niter);


#if defined(APP_JACOBI)
		const char* appName = "JACOBI";
#elif defined(APP_SPARSELU)
		const char* appName = "LU";
#elif defined(APP_HEAT)
		const char* appName = "HEAT";
#elif defined(APP_SW)
		const char* appName = "SW";
#elif defined(APP_SAT)
		const char* appName = "SAT";
#elif defined(APP_DTW)
		const char* appName = "DTW";
#elif defined(APP_INT)
		const char* appName = "INT";
#elif defined(APP_SNAP)
		const char* appName = "SNAP";
#endif
		printf("app\t%s\n", appName);


		if (params.mode == MODE_GLOBAL){
			printf("mode\t%s\n", "GLOBAL");
		}
		else if (params.mode == MODE_TASK){
			printf("mode\t%s\n", "TASK");
			if (SCHED_POLICY == RR){
				printf("sched\t%s\n", "RR");
			}
			else if (SCHED_POLICY == LF){
				printf("sched\t%s\n", "LF");
			}
			else if (SCHED_POLICY == AL){
				printf("sched\t%s\n", "AL");
			}
		}
		printf("matrixSize\t%d\n", params.matrix_size);
    printf("blockSize\t%d\n", params.blocksize);
    printf("timeIters\t%d\n", params.titer);
    printf("nWorkers\t%d\n", N_WORKERS);
    printf("nThreads\t%d\n", num_threads);


    int i;
    for (i=0; i<params.niter; ++i)
    {
      double cur_time = run(&params);
      all_times[i] = cur_time;
      mean += cur_time;
      min_ = min(min_, cur_time);
      max_ = max(max_, cur_time);
      meansqr += cur_time * cur_time;
	}
    mean /= params.niter;
    meansqr /= params.niter;
    double stddev = sqrt(meansqr - mean * mean);

    qsort(all_times, params.niter, sizeof(double), comp);
    double median = all_times[params.niter / 2];

    free(all_times);

    printf("execTime\t%lf\n",median);

//    printf("Time(sec):: ");
//    printf("avg : %lf :: std : %lf :: min : %lf :: max : %lf :: median : %lf\n",
//           mean, stddev, min_, max_, median);
    if(params.check)
        printf("check %s\n", (params.succeed)?
                ((params.succeed > 1)?"not implemented":"success")
                :"fail");
    if (params.string2display !=0)
      printf("%s", params.string2display);
    printf("\n");

    return 0;
}

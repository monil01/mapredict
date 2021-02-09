#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <CL/cl.h>
#include "timer.h"

#define CL_CHECK() {  \
    if (err != CL_SUCCESS) printf("[%s:%d] err[%d]\n", __FILE__, __LINE__, err); }

#define AOCL_ALIGNMENT  64
#define GIGA (1024 * 1024 * 1024)

cl_int err;
cl_device_id dev;
cl_context ctx;
cl_mem mem;
char name[1024];
cl_uint ndevs;

void *host;

int main(int argc, char** argv) {
    size_t SIZE = 1024 * 1024 * 16;

    /*
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 4; i++) CPU_SET(i, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    */

    err = clGetPlatformIDs(0, NULL, &ndevs);
    CL_CHECK();
    printf("nplatforms[%d]\n", ndevs);

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CL_CHECK();

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 1034, &name, NULL);
    CL_CHECK();

    printf("platform[%s]\n", name);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &ndevs);
    CL_CHECK();

    printf("ndevs[%d]\n", ndevs);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    CL_CHECK();

    err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 1023, name, 0);
    CL_CHECK();

    printf("name[%s]\n", name);

    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    CL_CHECK();

    cl_command_queue cmq = clCreateCommandQueue(ctx, dev, 0, &err);
    CL_CHECK();

    mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SIZE, NULL, &err);
    CL_CHECK();

    if (argc > 1) {
        host = valloc(SIZE);
    } else {
        host = malloc(SIZE);
    }

    void* temp = valloc(SIZE);

    double t00 = wtime();
    memcpy(temp, host, SIZE);
    double t01 = wtime();
    printf("warming up[%lf]\n", t01 - t00);

    printf("host[%p] [%lu]\n", host, (size_t) host % 4096UL);

    double t0 = wtime();
    for (size_t i = 0; i < SIZE / sizeof(double); i++) {
        double* x = (double*) host;
        x[i] = x[i] + 1;
    }
    double t1 = wtime();

    printf("warming up[%lf]\n", t1 - t0);

    size_t prefix = AOCL_ALIGNMENT - ((size_t) host & (AOCL_ALIGNMENT - 1));
    if (0 && prefix != AOCL_ALIGNMENT) {
        printf("not aligned! [%lu]\n", prefix);
        double t2 = wtime();
        err = clEnqueueWriteBuffer(cmq, mem, CL_TRUE, 0, prefix, (const void*) host, 0, NULL, NULL);
        CL_CHECK();
        double t3 = wtime();
        err = clEnqueueWriteBuffer(cmq, mem, CL_TRUE, 0, SIZE - prefix, (const void*) ((char*) (host) + prefix), 0, NULL, NULL);
        CL_CHECK();
        double t4 = wtime();
        printf("size[%lu] HtoD[%lf]\n", prefix, (t3 - t2) * 1000000);
        printf("size[%lu] HtoD[%lf]\n", SIZE - prefix, (t4 - t3) * 1000000);

        double t5 = wtime();
        err = clEnqueueReadBuffer(cmq, mem, CL_TRUE, 0, prefix, host, 0, NULL, NULL);
        CL_CHECK();
        double t6 = wtime();
        err = clEnqueueReadBuffer(cmq, mem, CL_TRUE, 0, SIZE - prefix, (char*) host + prefix, 0, NULL, NULL);
        CL_CHECK();
        double t7 = wtime();
        printf("size[%lu] DtoH[%lf]\n", prefix, (t6 - t5) * 1000000);
        printf("size[%lu] DtoH[%lf]\n", SIZE - prefix, (t4 - t3) * 1000000);
    } else {
        printf("aligned! [%lu]\n", prefix);
        double t2 = wtime();
        err = clEnqueueWriteBuffer(cmq, mem, CL_TRUE, 0, SIZE, (const void*) host, 0, NULL, NULL);
        double t3 = wtime();
        printf("size[%lu] HtoD[%lf]\n", SIZE, (t3 - t2) * 1000000);
        CL_CHECK();
        double t4 = wtime();
        err = clEnqueueReadBuffer(cmq, mem, CL_TRUE, 0, SIZE, host, 0, NULL, NULL);
        CL_CHECK();
        double t5 = wtime();
        printf("size[%lu] DtoH[%lf]\n", SIZE, (t5 - t4) * 1000000);
    }

    /*
    for (size_t i = 1; i <= SIZE; i <<= 1) {
        double t2 = wtime();
        err = clEnqueueWriteBuffer(cmq, mem, CL_TRUE, 0, i, (const void*) host, 0, NULL, NULL);
        CL_CHECK();
        double t3 = wtime();
        printf("size[%lu] HtoD[%lf]\n", i, (t3 - t2) * 1000000);
    }

    for (size_t i = 1; i <= SIZE; i <<= 1) {
        double t2 = wtime();
        err = clEnqueueReadBuffer(cmq, mem, CL_TRUE, 0, i, host, 0, NULL, NULL);
        CL_CHECK();
        double t3 = wtime();
        printf("size[%lu] DtoH[%lf]\n", i, (t3 - t2) * 1000000);
    }
    */

    return 0;
}


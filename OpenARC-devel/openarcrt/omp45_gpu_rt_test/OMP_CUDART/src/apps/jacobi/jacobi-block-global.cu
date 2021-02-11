#include "poisson.cuh"
#include "../../rt.h"
#include "../timer.h"

#ifdef APP_JACOBI

__global__  void  copy_block_global(int nx, int ny, double *u, double *unew, int block_size){
	int nbx = nx/block_size;
	int nby = ny/block_size;
	int nblocks= nbx*nby;

	int taskPerWorker = nblocks/gridDim.x;

	for (int i = 0; i < taskPerWorker; ++i) {
		int blockx = (taskPerWorker*blockIdx.x+i)%nby;
		int blocky = (taskPerWorker*blockIdx.x+i)/nby ;
		copy_block(nx, ny, blockx, blocky, u, unew, block_size);
//		if (threadIdx.x == 0)
//			printf("Worker:%d\tcopy:\t[%d,%d]\n",blockIdx.x,blockx,blocky);
	}
}
__global__  void  copy_block_global2(int nx, int ny, double *u, double *unew, int block_size){
//	__shared__ int sharedMemDummy[12*1024];
//
//	if (threadIdx.x == 46622){
//		sharedMemDummy[threadIdx.x] = threadIdx.x;
//		if (sharedMemDummy[threadIdx.x] > (threadIdx.x - blockIdx.x)){
//			u[threadIdx.x] = 0;
//			sharedMemDummy[threadIdx.x]++;
//		}
//	}
//
	copy_block(nx, ny, blockIdx.x, blockIdx.y, u, unew, block_size);
}
__global__  void  compute_estimate_global(double *u, double *unew, double *f, double dx,double dy, int nx, int ny, int block_size){

	int nbx = nx/block_size;
	int nby = ny/block_size;
	int nblocks= nbx*nby;

	int taskPerWorker = nblocks/gridDim.x;

	for (int i = 0; i < taskPerWorker; ++i) {
		int blockx = (taskPerWorker*blockIdx.x+i)%nby;
		int blocky = (taskPerWorker*blockIdx.x+i)/nby ;
		compute_estimate(blockx,blocky, u, unew, f, dx, dy, nx, ny, block_size);
//		if (threadIdx.x == 0)
//			printf("Compute:%d\tcopy:\t[%d,%d]\n",blockIdx.x,blockx,blocky);
	}
}
__global__  void  compute_estimate_global2(double *u, double *unew, double *f, double dx,double dy, int nx, int ny, int block_size){
//	__shared__ int sharedMemDummy[12*1024];
//
//	if (threadIdx.x == 46622){
//		sharedMemDummy[threadIdx.x] = threadIdx.x;
//		if (sharedMemDummy[threadIdx.x] > (threadIdx.x - blockIdx.x)){
//			u[threadIdx.x] = 0;
//			sharedMemDummy[threadIdx.x]++;
//		}
//	}
	compute_estimate(blockIdx.x,blockIdx.y, u, unew, f, dx, dy, nx, ny, block_size);
}
/* #pragma omp task/taskwait version of SWEEP. */
double sweep_global (int nx, int ny, double dx, double dy, double *f_,
            int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);


    double* f_d;
    double* u_d;
    double* unew_d;
	size_t dataSize = nx*ny*sizeof(double);
	cudaMalloc(&f_d,dataSize);
	cudaMalloc(&u_d,dataSize);
	cudaMalloc(&unew_d,dataSize);

	cudaMemcpy(f_d, f_, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(u_d, u_, dataSize, cudaMemcpyHostToDevice);
	cudaMemcpy(unew_d, unew_, dataSize, cudaMemcpyHostToDevice);

	START_TIMER
	for (it = itold + 1; it <= itnew; it++) {
		int mode = 2;
		if (mode == 1){
			dim3 gridDim(N_WORKERS,1,1);
			dim3 blockDim(block_size,1,1);
			copy_block_global<<<gridDim,blockDim>>>(nx,ny,u_d,unew_d,block_size);
			cudaDeviceSynchronize();

			compute_estimate_global<<<gridDim,blockDim>>>(u_d,unew_d,f_d, dx, dy, nx,ny,block_size);
			cudaDeviceSynchronize();
		}else if(mode ==2){
			dim3 gridDim(max_blocks_x,max_blocks_y,1);
			dim3 blockDim(block_size,1,1);
			copy_block_global2<<<gridDim,blockDim>>>(nx,ny,u_d,unew_d,block_size);
			cudaDeviceSynchronize();

			compute_estimate_global2<<<gridDim,blockDim>>>(u_d,unew_d,f_d, dx, dy, nx,ny,block_size);
			cudaDeviceSynchronize();

		}


	}
	END_TIMER


	cudaMemcpy(f_, f_d, dataSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(u_, u_d, dataSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(unew_, unew_d, dataSize, cudaMemcpyDeviceToHost);

	return TIMER;
}

#endif



/******************************************************************************/
/* END: OPENARC GENERATED CODE ************************************************/
/******************************************************************************/

#include "../main.h"
#include "../../rt.h"
#include "../timer.h"
#include "wavefront-kernel.cuh"

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)


__global__ void wavefrontGlobal(APP_DATA appData, int waveId){


	// determine row and column of the tile
	int tileRow=waveId - blockIdx.x;
	int tileColumn=blockIdx.x;
	if (waveId > appData.n_tile_rows-1){
		tileRow=appData.n_tile_rows-1-blockIdx.x;
		tileColumn=waveId-tileRow;
	}

	//computeTile(tileRow, tileColumn, &appData);
	computeTile(tileRow, tileColumn,
			appData.tile_width, appData.tile_height,
			appData.n_tile_rows, appData.n_tile_columns,
			appData.data
#if defined(APP_SW) || defined(APP_DTW)
		//	,appData.seq1
		//	,appData.seq2
#elif defined(APP_INT)
			,appData.bin
#endif
	);
}

//__global__ void wavefrontGlobal(int data_width, int data_height,
//		int tile_width, int tile_height, int nt_rows, int nt_columns,
//		float* data, int waveId){
//
//	// determine row and column of the tile
//	int tileRow=waveId - blockIdx.x;
//	int tileColumn=blockIdx.x;
//	if (waveId > nt_rows-1){
//		tileRow=nt_rows-1-blockIdx.x;
//		tileColumn=waveId-tileRow;
//	}
//
//	computeTile(tileRow, tileColumn, tile_width, tile_height, nt_rows, nt_columns, data);
//
//}

double wavefront_global(APP_DATA appData,
		int n_tile_waves, int minDim, int maxDim, int tile_size)
{
	APP_DATA appData_d = appData;
	size_t dataSize = appData_d.data_width * appData_d.data_height * sizeof(float);

    cudaMalloc(&(appData_d.data),appData_d.data_width*appData_d.data_height*sizeof(float));
    cudaMemcpy(appData_d.data, appData.data,
    		appData_d.data_width*appData_d.data_height*sizeof(float), cudaMemcpyHostToDevice);

#if defined(APP_SW) || defined(APP_DTW)
    cudaMalloc(&(appData_d.seq1),appData_d.data_width*sizeof(float));
    cudaMalloc(&(appData_d.seq2),appData_d.data_height*sizeof(float));
    cudaMemcpy(appData_d.seq1, appData.seq1,
    		appData_d.data_width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(appData_d.seq2, appData.seq2,
    		appData_d.data_height*sizeof(float), cudaMemcpyHostToDevice);
#endif

#if defined(APP_INT)
    size_t binSize = 255*sizeof(int);
    cudaMalloc(&(appData_d.bin),binSize);
    cudaMemcpy(appData_d.bin, appData.bin,binSize, cudaMemcpyHostToDevice);
#endif

    START_TIMER
	// Populate dependency through loop inspection
	for (int waveId=0; waveId<n_tile_waves; waveId++)
	{
		int waveLength = waveId+1;
		if (waveId+1 > minDim){
			waveLength = min(minDim,n_tile_waves-waveId);
		}
//		wavefrontGlobal<<<waveLength,min(tile_height,tile_width)>>>(data_width, data_height,
//				tile_width, tile_height, nt_rows, nt_columns,
//				data_d, waveId);
		wavefrontGlobal<<<waveLength,min(appData_d.tile_height,appData_d.tile_width)>>>
				(appData_d, waveId);
		cudaDeviceSynchronize();
	}
    END_TIMER


    cudaMemcpy(appData.data, appData_d.data, dataSize, cudaMemcpyDeviceToHost);

    return TIMER;
}
#endif

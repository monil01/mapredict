#include "../../rt.h"


#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define ABS(X) ((X) < (0) ? -1*(X) : (X))

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)

__device__ inline void computeTile(int tileRow, int tileColumn, int tile_width, int tile_height,
		int n_tile_rows, int n_tile_columns,float* data
#if defined(APP_SW) || defined(APP_DTW)
//			, float* seq1
//			, float* seq2
#elif defined(APP_INT)
			, int* bin
#endif

){
//
//	__shared__ int sharedMemDummy[9*1024];
//
//	if (threadIdx.x == 46622){
//		sharedMemDummy[threadIdx.x] = threadIdx.x;
//		if (sharedMemDummy[threadIdx.x] > (threadIdx.x - blockIdx.x)){
//			data[threadIdx.x] = 0;
//			sharedMemDummy[threadIdx.x]++;
//		}
//	}

	const int tid = threadIdx.x;

#if DEBUG_TASK_PRINT_FORMAT
	if (tid == 0)
	printf("ComputeTile: row:%d - col:%d\n",tileRow,tileColumn);
#endif

	const int tile_size = tile_width * tile_height;
	const int tileBaseIndex = tile_size * n_tile_columns * tileRow
			+ tile_size * tileColumn;

	const int nWaves = tile_width * 2 - 1;

#if defined(use_shared)
	#if defined(APP_SW) || defined(APP_DTW)
		__shared__ float seq1_local[512];
		__shared__ float seq2_local[512];
		__syncthreads();
		seq1_local[tid] = seq1[tileColumn*tile_width+tid];
		seq2_local[tid] = seq2[tileRow*tile_height+tid];

	#endif
#endif


// calculate base index
	__syncthreads();

	for (int waveId = 0; waveId < nWaves; waveId++) {
		const int row = waveId - tid;
		const int column = tid;

		int index = tileBaseIndex + tile_height * column + row;

		if (row >= 0 && row < tile_height &&
				column >= 0 && column < tile_width)
		{
			// ********//
			// VARIABLE SETUP
			// ********//
#if defined(APP_HEAT) || defined(APP_SAT) || defined(APP_INT)
			float sum = 0.0;
#elif defined(APP_SW)
			int upleft, left, up;
#elif defined(APP_DTW)
#if defined(use_shared)
			const int cost = ABS(seq2_local[row] - seq1_local[column]);
#else
			//const int cost = ABS(seq2[row] - seq1[column]);
			const int cost = 123;
#endif
			int upleft=INT_MAX;
			int left = INT_MAX;
			int up = INT_MAX;
#endif


			// ********//
			// LEFT & UP
			// ********//
#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)
			if (tileColumn > 0 || column > 0) //left
			{
				const int leftIndex = index - tile_height;
#if defined(APP_HEAT) || defined(APP_INT)
				sum += 0.20 * data[leftIndex];
#elif defined(APP_SAT)
				sum += data[leftIndex];
#elif defined(APP_SW)|| defined(APP_DTW)
				left = data[leftIndex] - 1;
#endif
			}
			if (tileRow > 0 || row > 0) {
				int upIndex = index - 1;
				if (row == 0) {
					upIndex = tileBaseIndex - tile_size * n_tile_columns;
					upIndex += tile_height * (column + 1) - 1;
				}
#if defined(APP_HEAT)
				sum += 0.20 * data[upIndex];
#elif defined(APP_SAT)
				sum += data[upIndex];
#elif defined(APP_SW) || defined(APP_DTW)
				up = data[upIndex] - 1;
#endif
			}
#endif
			// ********//
			//  UPLEFT //
			// ********//

#if defined(APP_SW) || defined(APP_DTW) || defined (APP_INT) //|| defined (APP_SAT)
			if ((tileColumn > 0 || column > 0) && (tileColumn > 0 || column > 0)) {
				int upLeftIndex = index - 1 - tile_height ;
				if (row == 0) {
					upLeftIndex = tileBaseIndex - tile_size * n_tile_columns;
					upLeftIndex += tile_height * (column) - 1;
				}
#if defined(APP_SW)
#if defined(use_shared)
				if (seq2_local[row] != seq1_local[column]) //TODO use local
#else
				//if (seq2[tileRow*tile_height+row] != seq1[tileColumn*tile_height+column]) //TODO use local
#endif
					//upleft = data[upLeftIndex] - 1;
				//else
				//	upleft = data[upLeftIndex] + 2;
#elif defined(APP_DTW)
				upleft = data[upLeftIndex];
#elif defined(APP_SAT) || defined(APP_INT)
				sum -= data[upLeftIndex];
#endif
			}
#endif

			// ********//
			// BOTTOM & RIGHT
			// ********//

#if defined(APP_HEAT)
			if (tileColumn < (n_tile_columns) - 1 || column < tile_width - 1){ //right
				int rightIndex = index + tile_height;
				sum += 0.20 * data[rightIndex];
			}

			if (tileRow < (n_tile_rows) - 1 || row < tile_height - 1) { //down
				int downIndex = index + 1;
				if (row == tile_height - 1) {
					downIndex = tileBaseIndex + tile_size * n_tile_columns; // next tile row
					downIndex += column * tile_height;
				}
				sum += 0.20 * data[downIndex];
			}
#endif
			// ********//
			// FINALIZE
			// ********//
#if defined(APP_HEAT)
			data[index] = 0.20 * data[index] + sum;
#elif defined(APP_SAT)
			data[index] +=data[index] + sum;
#elif defined(APP_SW)
			int max = MAX(0, MAX(upleft, MAX(left, up)));
			data[index] = max;
#elif defined(APP_DTW)
			data[index] = cost+MIN(upleft,MIN(left,up));
#elif defined(APP_INT)
			data[index] += sum;
			bin[(int)sum%255]+=1;
#endif
		}

		__syncthreads();
	}

#if defined(APP_INT)
		//if (tid<255)
			//atomicAdd(&(bin[tid]), bin_local[tid]);
#endif


}
#endif

/******************************************************************************/
/* BEGIN: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/
/******************************************************************************/
/* END: OPENARC GENERATED CODE ******************************************/
/******************************************************************************/

// __device__ inline void computeTile_HT(int tileRow, int tileColumn, int data_width,
// 	 int data_height, int tile_width, int tile_height, int n_tile_rows, int n_tile_columns,
// 	 float* data){
// 	const int tid = threadIdx.x;
//
// 	int row,column;
// 	int row_t,column_t;
// 	int index, tileBaseIndex;
// 	float sum = 0.0;
//
// 	const int tile_size = tile_width*tile_height;
// 	const int tileBaseIndex = tile_size*n_tile_columns*tileRow + tile_size*tileColumn;
//
// 	const int nWaves = tile_width;
//
// 		// calculate base index
// 	__syncthreads();
//
// 	for (int waveId=0; waveId< nWaves ; waveId++){
// 		row=waveId;
// 		column=tid;
// 		row_t = AFF_TRANSFORM(waveId,tid,1,-1);
// 		column_t = AFF_TRANSFORM(waveId,tid,0,1);
//
// 		index = tileBaseIndex+tile_height*row+column;
//
// //#if HYPERTILING
// //		// Boundary check
// //		if ((tileRow > 0 ||  y_t > 0) &&
// //			(tileRow < nt_y-1 || y_t <= tile_height-1) &&
// //			(tileColumn > 0 || x_t > 0) &&
// //			(tileColumn >= (nt_x)-1 || x_t <= tile_width -1))
// //		{
// //#else
// //
// //#endif
// 		if (row >=0 && column >= 0 && row < tile_height && column < tile_width ){
// 			if (tileColumn > 0 ||  column > 0 ){  //left
// 				sum += 0.20 * data[index-tile_height];
// 			}
//
// 			if (tileRow > 0 || row > 0 ){ //up
// 				int neighborIndex = index - tile_width;
// 				if (row == 0){
// 					neighborIndex = tileBaseIndex - tile_size*n_tile_columns;
// 					neighborIndex += tile_height*(row-1)+ column;
// 				}
// 				sum += 0.20 * data[neighborIndex];
// 			}
// 			if (tileColumn < (n_tile_columns)-1  || column < tile_width-1 ) //right
// 				sum += 0.20 * data[index+tile_height];
// 			if (tileRow < (n_tile_rows)-1  || row < tile_height-1 )){ //down
// 				int neighborIndex = index + tile_width;
// 				if (row == tile_height-1){
// 					neighborIndex = tileBaseIndex + tile_size*n_tile_columns; // next tile row
// 					neighborIndex += column;
// 				}
// 				sum +=  0.20 * data[neighborIndex];
// 			}
//
// 			data[index] = 0.20*data[index] +sum;
// 		}
//
// 		__syncthreads();
// 	}
//
// }

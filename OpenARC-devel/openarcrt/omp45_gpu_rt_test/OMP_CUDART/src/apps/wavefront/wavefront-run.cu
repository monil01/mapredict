#include "../main.h"
#include "../../rt.h"
#include "../../helper/utils.h"

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)
double wavefront_global(APP_DATA appData, int n_tile_waves, int minDim, int maxDim, int tile_size);
double wavefront_task(APP_DATA appData, int n_tile_waves, int minDim, int maxDim, int tile_size);
double wavefront_task(float* data, int data_width, int data_height, int tile_width, int tile_height,
		int n_tile_columns, int n_tile_rows, int n_tile_waves, int minDim,
		int maxDim, int tile_size);
double wavefront_global(float* data, int data_width, int data_height, int tile_width, int tile_height,
		int n_tile_columns, int n_tile_rows, int n_tile_waves, int minDim,
		int maxDim, int tile_size);

double wavefront_serial(float* data, int data_width, int data_height, int tile_width, int tile_height,
		int n_tile_columns, int n_tile_rows, int n_tile_waves, int minDim,
		int maxDim, int tile_size){
	double sum=0;
	for (int tileRow = 0; tileRow < n_tile_rows; ++tileRow) {
		for (int tileColumn = 0; tileColumn < n_tile_columns; ++tileColumn) {
			const int tileBaseIndex = tile_size*n_tile_columns*tileRow + tile_size*tileColumn;
			for (int column = 0; column < tile_width; ++column) {
				for (int row = 0; row < tile_height; ++row) {
					int index = tileBaseIndex+tile_height*row+column;
					if (tileColumn > 0 ||  column > 0 ){  //left
						sum += 0.20 * data[index-tile_height];
					}

					if (tileRow > 0 || row > 0 ){ //up
						int neighborIndex = index - tile_width;
						if (row == 0){
							neighborIndex = tileBaseIndex - tile_size*n_tile_columns;
							neighborIndex += tile_height*(row-1)+ column;
						}
						sum += 0.20 * data[neighborIndex];
					}
					if (tileColumn < (n_tile_columns)-1  || column < tile_width-1 ) //right
						sum += 0.20 * data[index+tile_height];
					if (tileRow < (n_tile_rows)-1  || row < tile_height-1 ){ //down
						int neighborIndex = index + tile_width;
						if (row == tile_height-1){
							neighborIndex = tileBaseIndex + tile_size*n_tile_columns; // next tile row
							neighborIndex += column;
						}
						sum +=  0.20 * data[neighborIndex];
					}
					data[index] = 0.20*data[index] +sum;
				}
			}
		}

	}

	return 0;
}

/* R8MAT_RMS returns the RMS norm of a vector stored as a matrix. */
double r8mat_rms(float* a, int nx, int ny) {
	int i;
	double v;

	v = 0.0;

	for (i = 0; i < ny*nx; i++) {
		v+= a[i];
	}
	v = sqrt(v / (double) (nx * ny));

	return v;
}

double run(struct user_parameters* params)
{
	APP_DATA appData;
	appData.data_width = params->matrix_size;
	appData.data_height = params->matrix_size;
	appData.tile_width = params->blocksize;
	appData.tile_height = params->blocksize;
	appData.n_tile_columns = (appData.data_width+appData.tile_width-1)/appData.tile_width;
	appData.n_tile_rows = (appData.data_height+appData.tile_height-1)/appData.tile_height;
	int n_tile_waves = appData.n_tile_columns+appData.n_tile_rows-1;
	int minDim = min(appData.n_tile_rows, appData.n_tile_columns);
	int maxDim = max(appData.n_tile_rows, appData.n_tile_columns);
	int tile_size = appData.tile_width * appData.tile_height;
	size_t dataSize = appData.data_width*appData.data_height*sizeof(float);

	appData.data = (float*) malloc (dataSize);
	if (params->check){
		memset(appData.data,1,dataSize);
	}


#if defined(APP_SW) || defined (APP_DTW)
	appData.seq1 = (float*) malloc (appData.data_width*sizeof(float));
	if (params->check){
		memset(appData.seq1,2,appData.data_width*sizeof(float));
	}
	appData.seq2 = (float*) malloc (appData.data_height*sizeof(float));
	if (params->check){
		memset(appData.seq2,3,appData.data_height*sizeof(float));
	}
#endif

#if defined(APP_INT)
    size_t binSize = 255*sizeof(int);
	appData.bin = (int*) malloc (binSize);
#endif

	double time;
	if (params->mode == MODE_GLOBAL){
		time = wavefront_global(appData, n_tile_waves,minDim,maxDim,tile_size);
	}
	else if (params->mode == MODE_TASK){
		time = wavefront_task(appData, n_tile_waves,minDim,maxDim,tile_size);
	}

	if (params->check){
		float* data_serial = (float*) malloc (dataSize);
		memset(data_serial,1,dataSize);

		wavefront_serial(data_serial, appData.data_width,appData.data_height,
				appData.tile_width, appData.tile_height,
				appData.n_tile_columns, appData.n_tile_rows,
				n_tile_waves,minDim,maxDim,tile_size);
		double error1=r8mat_rms(appData.data, appData.data_width,appData.data_height);
		double error2=r8mat_rms(data_serial, appData.data_width,appData.data_height);
		params->succeed = fabs(error2 - error1) < 1.0E-5;

		FILE* data_alg_fp = fopen("data_alg","w");
		if (data_alg_fp){
			print2DData(data_alg_fp,appData.data, appData.data_width,appData.data_height);
		}
		fclose(data_alg_fp);

		FILE* data_ser_fp = fopen("data_seq","w");
		if (data_ser_fp){
			print2DData(data_ser_fp,data_serial, appData.data_width,appData.data_height);
		}
		fclose(data_ser_fp);

		free(data_serial);
	}
	free(appData.data);


	return time;
}
#endif

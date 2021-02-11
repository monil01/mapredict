#ifndef _APP_H
#define _APP_H

/******************************************************************************/
/* BEGIN: OPENARC GENERATED CODE **********************************************/
/******************************************************************************/

# define APP_JACOBI
//#define use_shared

//#define APP_JACOBI
//#define APP_SPARSELU
//#define APP_HEAT
//#define APP_SW
//#define APP_SAT
//#define APP_DTW
//#define APP_INT
//#define APP_SNAP

#ifdef APP_JACOBI
	#define KERNEL_TYPE_COPY_BLOCK 0
	#define KERNEL_TYPE_COMPUTE_BLOCK 0
#endif

#ifdef APP_SPARSELU
#define KERNEL_TYPE_LU0 0
#define KERNEL_TYPE_BDIV 1
#define KERNEL_TYPE_FWD 2
#define KERNEL_TYPE_BMOD 3
#endif

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)
#define KERNEL_TYPE_WF 0
#endif

typedef struct _APP_DATA{

#ifdef APP_JACOBI
	int nx;
	int ny;
	int blockSize;
	double dx;
	double dy;
	double *u;
	double *unew;
	double *f;
#endif

#ifdef APP_SPARSELU
	float *BENCH;
	int matrix_size;
	int submatrix_size;
#endif

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)
	int data_width;
	int data_height;
	int tile_width;
	int tile_height;
	int n_tile_columns;
	int n_tile_rows;
	float *data;

#if defined(APP_DTW) || defined(APP_SW)
	float* seq1;
	float* seq2;
#endif

#if defined(APP_INT)
	int* bin;
#endif

#endif

} APP_DATA;

typedef struct _TASK_DATA{

#ifdef APP_JACOBI
	int block_x;
	int block_y;
#endif
#ifdef APP_SPARSELU
	int colBlockStart;
	int rowBlockStart;
	int subBlockStart;
	int diagBlockStart;
#endif

#if defined(APP_HEAT) || defined(APP_SW) || defined(APP_SAT) || defined(APP_DTW) || defined (APP_INT)
	int tileRow;
	int tileColumn;
#endif
} TASK_DATA;


/******************************************************************************/
/* END: OPENARC GENERATED CODE ************************************************/
/******************************************************************************/

#endif

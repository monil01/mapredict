// extern int workers, gangs, size;
// extern float *A, *B, *D, *E;
#include "openacc.h"

void funcFPGA(
    float* X_result,
    float* VAL,
    int* COL_IND,
    int* ROW_PTR,
    float* B,
    int N,
    int K,
    int VAL_SIZE
);

// void initFPGA(
//     float* restrict X_result,
//     float* restrict VAL,
//     int* restrict COL_IND,
//     int* restrict ROW_PTR,
//     float* restrict B,
//     int N,
//     int K,
//     int VAL_SIZE
// );

// void shutdownFPGA(
//     float* restrict X_result,
//     float* restrict VAL,
//     int* restrict COL_IND,
//     int* restrict ROW_PTR,
//     float* restrict B,
//     int N,
//     int K,
//     int VAL_SIZE
// );

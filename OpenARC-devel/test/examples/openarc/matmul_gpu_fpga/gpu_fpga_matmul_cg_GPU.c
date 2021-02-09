#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "gpu_fpga.h"

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0); 

    return time.tv_sec + time.tv_usec / 1000000.0;
}

double matmul(float *a, float *b, float *c, int N, float *e, float *f)
{
	int i, j, k;
	double gpu_time = my_timer();
	acc_set_device_type(acc_device_gpu);

#pragma acc data create(a[:N*N], b[:N*N], c[:N*N], e[:N], f[:N])
{
	gpu_time = my_timer();

#pragma acc update device(a[:N*N], b[:N*N], e[:N])

#pragma acc kernels
	{
#pragma acc loop gang vector(32) independent
		for (i = 0; i < N; ++i)
		{
#pragma acc loop gang vector(32) independent
			for (j = 0; j < N; ++j)
			{
				float sum = 0.0;
#pragma acc loop independent reduction(+:sum)
				for (k = 0; k < N; ++k) {
//#pragma acc cache(readonly:a)
					sum += a[i * N + k] * b[k * N + j];
}				c[i * N + j] = sum;
			}
		}
	}

//matrix_vector_malti

#pragma acc kernels
	{
#pragma acc loop gang vector independent
		for (i = 0; i < N; ++i)
		{
			float sum = 0.0;
#pragma acc loop independent reduction(+:sum)
			for (j = 0; j < N; ++j)
				sum += c[i * N + j] * e[j];
			f[i] = sum;
		}
	}

#pragma acc update self(f[:N])
	gpu_time = my_timer() - gpu_time;
}

return gpu_time;
}

// void matrix_vector_malti(float *a, float *b, float *c, int N)
// {
// 	int i, j;

// // <<<dim3(numblock), dim3(numthread)>>>
// #pragma acc update device(a[:N*N], b[:N])

// #pragma acc kernels present(c, a, b)
// 	{
// #pragma acc loop independent gang(N/32) vector(32)
// 		for (i = 0; i < N; ++i)
// 		{
// 			float sum = 0.0;
// #pragma acc loop reduction(+:sum)
// 			for (j = 0; j < N; ++j)
// 				sum += a[i * N + j] * b[j];
// 			c[i] = sum;
// 		}
// 	}

// #pragma acc update self(c[:N])
// }

void MatrixMultiplication_openmp(float *a, float *b, float *c, int N)
{
	int i, j, k;
	int chunk;
// #ifdef _OPENMP
	// omp_set_num_threads(numstream);
	if (omp_get_thread_num() == 0)
	{
		printf("Number of OpenMP threads %d\n", omp_get_max_threads());
		chunk = N / omp_get_max_threads();
	}
// #endif

#pragma omp parallel shared(a, b, c, chunk) private(i, j, k)
	{
#pragma omp for
		for (i = 0; i < N; ++i)
		{
			for (j = 0; j < N; ++j)
			{
				float sum = 0.0;
				for (k = 0; k < N; ++k)
					sum += a[i * N + k] * b[k * N + j];
				c[i * N + j] = sum;
			}
		}
	}
}

void h_matrix_vector_malti(float *a, float *b, float *c, int N)
{
	int i, j;
	int chunk;
// #ifdef _OPENMP
	if (omp_get_thread_num() == 0)
	{
		printf("Number of OpenMP threads %d\n", omp_get_max_threads());
		chunk = N / omp_get_max_threads();
	}
// #endif

#pragma omp parallel shared(a, b, c, chunk) private(i, j)
	{
#pragma omp for
		for (i = 0; i < N; ++i)
		{
			float sum = 0.0;
			for (j = 0; j < N; ++j)
				sum += a[i * N + j] * b[j];
			c[i] = sum;
		}
	}
}

void verify_gpu(float *h_c, float *c_CPU, unsigned long N)
{
	double cpu_sum = 0.0;
	double gpu_sum = 0.0;
	double rel_err = 0.0;

#pragma omp parallel for reduction(+:cpu_sum, gpu_sum)
	for (unsigned long i = 0; i < N; ++i)
	{
		// printf("(CPU) %f\n", c_CPU[i]);
		// printf("(GPU) %f\n", h_c[i]);
		cpu_sum += (double)c_CPU[i] * c_CPU[i];
		gpu_sum += (double)h_c[i] * h_c[i];
	}

	cpu_sum = sqrt(cpu_sum);
	gpu_sum = sqrt(gpu_sum);
	if (cpu_sum > gpu_sum)
	{
		rel_err = (cpu_sum - gpu_sum) / cpu_sum;
	}
	else
	{
		rel_err = (gpu_sum - cpu_sum) / cpu_sum;
	}

	if (rel_err < 1e-6)
	{
		printf("Verification Successful err = %e\n", rel_err);
	}
	else
	{
		printf("Verification Fail err = %e\n", rel_err);
	}
	printf("ResultGPU = %lf\n", gpu_sum);
	printf("ResultCPU = %lf\n", cpu_sum);
}

void verify_fpga(
    float* FPGA_calc_result,
    float* VAL,
    int* COL_IND,
    int* ROW_PTR,
    float* B,
    int N,
    int K,
    int VAL_SIZE
	  )
{
	// float *x = new float[N], *r = new float[N], *p = new float[N], *y = new float[N], alfa, beta;
	// float *VAL_local = new float[VAL_SIZE];
	// int *COL_IND_local = new int[VAL_SIZE], *ROW_PTR_local = new int[N + 1];
	// float temp_sum, temp_pap, temp_rr1, temp_rr2;
  int error = N;

/*
	float x[N], r[N], p[N], y[N], alfa, beta;
	float VAL_local[VAL_SIZE];
	int COL_IND_local[VAL_SIZE], ROW_PTR_local[N + 1];
*/
	float *x;
	float *r;
	float *p;
	float *y;
	float alfa, beta;
	float *VAL_local;
	int *COL_IND_local;
	int *ROW_PTR_local;
	float temp_sum, temp_pap, temp_rr1, temp_rr2, sum = 0, sum_cpu = 0;

	x = (float *)malloc(N*sizeof(float));
	r = (float *)malloc(N*sizeof(float));
	p = (float *)malloc(N*sizeof(float));
	y = (float *)malloc(N*sizeof(float));
	VAL_local = (float *)malloc(VAL_SIZE*sizeof(float));
	COL_IND_local = (int *)malloc(VAL_SIZE*sizeof(int));
	ROW_PTR_local = (int *)malloc((N+1)*sizeof(int));

    printf("Calculate the reference results\n");
	double reference_time = my_timer();

	temp_rr1 = 0.0f;
	for(int i = 0; i < N; ++i){
		ROW_PTR_local[i] = ROW_PTR[i];
		x[i] = 0.0f;
		r[i] = B[i];
		p[i] = B[i];
		temp_rr1 += r[i] * r[i];
	}
	ROW_PTR_local[N] = ROW_PTR[N];

	for(int i = 0; i < VAL_SIZE; ++i){
		COL_IND_local[i] = COL_IND[i];
		VAL_local[i] = VAL[i];
	}

	for(int i = 0; i < K; ++i){
		temp_pap = 0.0f;
		for(int j = 0; j < N; ++j){
			temp_sum = 0.0f;
			for(int l = ROW_PTR_local[j]; l < ROW_PTR_local[j + 1]; ++l){
				temp_sum += p[COL_IND_local[l]] * VAL_local[l];
			}
			y[j] = temp_sum;
			temp_pap += p[j] * temp_sum;
		}

		alfa = temp_rr1 / temp_pap;

		temp_rr2 = 0.0f;
		for(int j = 0; j < N; ++j){
			x[j] += alfa * p[j];
			r[j] -= alfa * y[j];
			temp_rr2 += r[j] * r[j];
		}

		beta = temp_rr2 / temp_rr1;

		for(int j = 0; j < N; ++j){
			p[j] = r[j] + beta * p[j];
		}
		temp_rr1 = temp_rr2;

	}

	reference_time = my_timer() - reference_time;

// if (fetestexcept(FE_INVALID)) {
//    puts("浮動小数点例外が発生しました");
// }
// if (fetestexcept(FE_DIVBYZERO)) {
//    puts("ゼロ除算が発生しました");
// }
// if (fetestexcept(FE_OVERFLOW)) {
//    puts("オーバーフローが発生しました");
// }
// if (fetestexcept(FE_UNDERFLOW)) {
//    puts("アンダーフローが発生しました");
// }
// if (fetestexcept(FE_INEXACT)) {
//    puts("不正確な結果が発生しました");
// }

	for(int j = 0; j < N; ++j){
    // std::cout << "FPGA" << FPGA_calc_result[j] << ", CPU"<< x[j] << std::endl;
		if(FPGA_calc_result[j] != x[j]) {
      error = j;
      // break;
    }
    sum += FPGA_calc_result[j];
    sum_cpu += x[j];
	}

  if (error == N) {
	printf("------------------------------\n");
    printf("FPGA Verification: PASS\n");
    printf("ResultFPGA = %lf\n", sum); 
  } else {
    printf("Error! FPGA Verification failed...\n");
    printf("ResultFPGA = %lf\n",sum);
    printf("ResultCPU  = %lf\n", sum_cpu);
   }
  printf("CG CPU elapsed time: %lf sec\n", reference_time);
}

int main(int argc, char *argv[])
{
	acc_init(acc_device_default);
  	const int  numdata_h = atoi(argv[1]);
	int N = numdata_h;
  	const int  valsize   = atoi(argv[2]);
	int VAL_SIZE = valsize;
	const int numtry = atoi(argv[3]);
	const unsigned long numbyte = numdata_h * numdata_h * sizeof(float); // this sample uses "float"

    if( valsize != numdata_h ) {printf("numdata_h (%d) should have the same value as valsize (%d); exit!\n", numdata_h, valsize); return 1; }
    else { printf("numdata_h: %d, valsize: %d, numtry: %d\n", numdata_h, valsize, numtry); }

	// host memory settings
	///////////////////////////////////////////

	/***** GPU *****/
	// static const int numthread = 16;
	// const int numblock = (numdata_h % numthread) ? (numdata_h / numthread) + 1 : (numdata_h / numthread);
	float *h_a, *h_b, *h_c, *c_CPU, *h_vec_b, *h_vec_mul, *vec_b_CPU;

	h_a = (float *)malloc(numbyte);
	h_b = (float *)malloc(numbyte);
	h_c = (float *)malloc(numbyte);
	c_CPU = (float *)malloc(numbyte);
	vec_b_CPU = (float *)malloc(numdata_h * sizeof(float));
	h_vec_mul = (float *)malloc(numdata_h * sizeof(float));
	h_vec_b = (float *)malloc(numdata_h * sizeof(float));

	for (int i = 0; i < numdata_h; ++i)
	{
		for (int j = 0; j < numdata_h; ++j)
		{
			h_a[i * numdata_h + j] = (j + 1) / 2 * 0.0001f;
			h_b[i * numdata_h + j] = 0.5f;
			h_c[i * numdata_h + j] = 0.0f;
			c_CPU[i * numdata_h + j] = 0.0f;
		}
		h_vec_b[i] = 0.0f;
		h_vec_mul[i] = 0.01f;
		vec_b_CPU[i] = 0.0f;
	}

	/***** FPGA *****/
	int K = numtry;
	float *FPGA_calc_result; // X_result;
	float *VAL;
	int *COL_IND;
	int *ROW_PTR;
	float *B;

  posix_memalign((void **)&FPGA_calc_result, 64, N * sizeof(float));
  posix_memalign((void **)&VAL, 64, VAL_SIZE * sizeof(float));
  posix_memalign((void **)&COL_IND, 64, VAL_SIZE * sizeof(int));
  posix_memalign((void **)&ROW_PTR, 64, (N+1) * sizeof(int));
  posix_memalign((void **)&B, 64, N * sizeof(float));

  double *VAL_temp;
  posix_memalign((void **)&VAL_temp, 64, VAL_SIZE * sizeof(double));
   
/*
  memcpy(VAL_temp, A->values, VAL_SIZE * sizeof (double));
  memcpy(COL_IND, A->colidx, VAL_SIZE * sizeof (int));
  memcpy(ROW_PTR, A->rowptr, (N+1) * sizeof (int));
*/
  //[DEBUG] Randomly initialize input data.
  float floatN = (float)N;
  float floatVAL_SIZE = (float)VAL_SIZE;
  float delta = floatN/floatVAL_SIZE;
  int temp = 0;
  for (int i = 0; i < VAL_SIZE; ++i)
  {  	
        float floatI = (float)i;
        temp = (int)(delta * floatI);
        if( temp >= N ) { temp = N-1; }
        VAL_temp[i] = floatI;
		COL_IND[i] = temp;
  }
  delta = floatVAL_SIZE/floatN;
  for (int i = 0; i < N; ++i)
  {  	
        float floatI = (float)i;
        temp = (int)(delta * floatI);
        if( temp >= VAL_SIZE ) { temp = VAL_SIZE-1; }
        ROW_PTR[i] = temp;
  }
  ROW_PTR[N] = VAL_SIZE;

  for (int i = 0; i < VAL_SIZE; ++i)
  {
        VAL[i] = (float)VAL_temp[i];
  }

	// device memory settings
	///////////////////////////////////////////

	// main routine
	///////////////////////////////////////////


	/***** GPU *****/

	double gpu_time = matmul(h_a, h_b, h_c, numdata_h, h_vec_mul, h_vec_b);
	// matrix_vector_malti(h_c, h_vec_mul, h_vec_b, numdata_h);

// #pragma acc exit data delete(h_a, h_b, h_c)
// #pragma acc exit data delete(h_vec_mul, h_vec_b)

	printf("GPU  elapsed time: %lf seconds\n", gpu_time);
	printf("------------------------------\n");
	///////////////////////////////////////////
	// verification
	///////////////////////////////////////////
	MatrixMultiplication_openmp(h_a, h_b, c_CPU, numdata_h);    // 本番はコメントアウトして良い
	h_matrix_vector_malti(c_CPU, h_vec_mul, vec_b_CPU, numdata_h);    // 本番はコメントアウトして良い
	verify_gpu(h_vec_b, vec_b_CPU, numdata_h); // h_vec_b チェック

	acc_set_device_type(acc_device_altera);

	/***** FPGA *****/
	for (int j = 0; j < N; ++j)
	{
		FPGA_calc_result[j] = 0;
		// ROW_PTR[j] = A->rowptr[j];
		B[j] = h_vec_b[j] - VAL[j] * 1; //000000.0; // b - Ax
	}
	// ROW_PTR[N] = N;

	// initFPGA(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);

	double fpga_time = my_timer();

	// sendDataToFPGA(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);

	funcFPGA(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);

	// recvDataFromFPGA(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);
	
	fpga_time = my_timer() - fpga_time;

	// shutdownFPGA(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);


	printf("FPGA  elapsed time: %lf seconds\n", fpga_time);
	printf("------------------------------\n");

	///////////////////////////////////////////
	// verification
	///////////////////////////////////////////

	verify_fpga(FPGA_calc_result, VAL, COL_IND, ROW_PTR, B, N, K, VAL_SIZE);

	// cleanup
	///////////////////////////////////////////
  // destroy_sparse_matrix(A_);
	// destroy_csr_matrix(A);
	

	return 0;
}

#include <stdio.h>
#include <math.h>
#include <omp.h>


#define N (1024*512)
int main()
{
	int num_blocks = 1024;
	int i,b,a;
	float A[N], B[N], C[N], D[N];
	for(i=0; i<N; i++) {
		A[i] = 0.0F;
		B[i] = 1.0F;
		C[i] = 2.0F;
		D[i] = 3.0F;
	}

	#pragma omp target device(0) map(tofrom:B,b,a) map(alloc:A) map(to:C) map(from:D) 
	{
		#pragma omp teams parallel for 
		for (i=0; i<N; i ++)
		{	
			#pragma omp atomic read
			a = b++;

			#pragma omp atomic write
			b = i;
			
			#pragma omp atomic capture
			b = b - 10 * 2;

			#pragma omp atomic capture
			b = 10 * i - b;

			#pragma omp atomic capture
			b /= 10;

			#pragma omp atomic capture
			b++;

//			#pragma omp atomic capture
//			{
//				a = x;
//				x++;
//			}
		}
	}
	printf("exit target teams\n");
}

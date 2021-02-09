#include <stdio.h>
#include <math.h>
#include <omp.h>


#define N (1024*512)
int main()
{
	int num_blocks = 1024;
	int i,b;
	int cond = 1;
	float A[N], B[N], C[N], D[N];
	for(i=0; i<N; i++) {
		A[i] = 0.0F;
		B[i] = 1.0F;
		C[i] = 2.0F;
		D[i] = 3.0F;
	}

	#pragma omp target device(0) map(tofrom:B) map(alloc:A) map(to:C) map(from:D) if(cond)
	{
		#pragma omp teams parallel for 
		for (i=0; i<N; i += num_blocks)
		{
			for (b = i; b < i+num_blocks; b++)
			{
				B[b] += sin(B[b]);
			}
			printf("exit parallel for teams\n");	
		}
	}
	printf("exit target teams\n");
}

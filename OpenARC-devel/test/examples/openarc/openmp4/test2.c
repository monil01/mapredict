#include <stdio.h>
#include <math.h>
#include <omp.h>


#define N (1024*512)
int main()
{
	float A[N], B[N], C[N], D[N];
	int num_blocks = 1024;
	int i,b;
	int cond = 1;
	for(i=0; i<N; i++) {
		A[i] = 0.0F;
		B[i] = 1.0F;
		C[i] = 2.0F;
		D[i] = 3.0F;
	}

	#pragma omp target teams device(0) map(tofrom:B) map(alloc:A) map(to:C) map(from:D) if(cond) num_teams(num_blocks) thread_limit(512)
	{
		#pragma omp distribute
		for (i=0; i<N; i += num_blocks)
		{
			#pragma omp parallel for simd
			for (b = i; b < i+num_blocks; b++)
			{
				B[b] += sin(B[b]);
			}
			printf("exit parallel for teams\n");	
		}
		printf("exit distribute teams\n");
	}
	printf("exit target teams\n");
}

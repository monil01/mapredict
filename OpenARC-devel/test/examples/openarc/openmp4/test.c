#include <stdio.h>
#include <math.h>


#define N (1024*512)
#ifdef _OPENARC_
#pragma openarc #define N (1024*512)
#endif

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

	#pragma omp target device(0) map(tofrom:B) map(alloc:A[0:N]) map(to:C) map(from:D) if(cond)
	#pragma omp teams num_teams(num_blocks) thread_limit(512)
	{
		#pragma omp distribute
		for (i=0; i<N; i += num_blocks)
		{
			#pragma omp parallel for simd
			for (b = i; b < i+num_blocks; b++)
			{
				A[b] = C[b];
				D[b] = A[b] + C[b];
				B[b] += sin(B[b]);
			}
			printf("exit parallel for teams\n");	
		}
		printf("exit distribute teams\n");
	}
	printf("exit target teams\n");
}

#include <stdio.h>
#include <math.h>
#include <omp.h>


#define N (1024*512)
int main()
{
	float B[N];
	int num_blocks = 1024;
	int i,b;
	int n, m;


	int A [] = {84, 30, 95, 94, 36, 73, 52, 23, 2, 13};
	int S [10] = {0};

	#pragma omp target device(0) map(tofrom:A[0:10],S)
	#pragma omp teams num_teams(num_blocks) thread_limit(512)
	{
		int S_private[10] = {0};
		#pragma omp distribute parallel for shared(S, A)
		for (n=0 ; n<10 ; ++n ) {
			#pragma omp critical
			{
				for(m = 0; m < n; m++)
					S[m] += A[n];
			}
		}
	}
	printf("exit target teams\n");
}

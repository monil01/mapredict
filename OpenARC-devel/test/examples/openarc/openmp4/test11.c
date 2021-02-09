#include <stdio.h>
#include <math.h>
#include <omp.h>

#define N (1024*512)

#pragma omp declare target
int C [][][] = {{{1},{2}}, {{3},{4}}, {{5},{6}}};
#pragma omp end declare target

int main()
{
	int num_blocks = 1024;
	int i,b;

	#pragma omp declare target
	int A [] = {84, 30, 95, 94, 36, 73, 52, 23, 2, 13};
	#pragma omp end declare target
	int S [] = {0};
	int B [][] = {{1,2}, {3,4}, {5,6}};
	#pragma omp declare target (B)

	#pragma omp target enter data map(S)

	#pragma omp target device(0) 
	#pragma omp teams 
	{
	    	int S_private[10] = {0};
    		#pragma omp distribute parallel for 
    		for (int n=0 ; n<10 ; ++n ) {
 	   		{
            			b += S[0] + A[n] + C[0][0];
	    		}
		}
    	}
	printf("exit target teams\n");

	#pragma omp target exit data map(S)
}

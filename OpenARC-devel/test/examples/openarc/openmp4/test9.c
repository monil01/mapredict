#include <stdio.h>
#include <math.h>
#include <omp.h>


#define N (1024*512)
int main()
{
	int num_blocks = 1024;
	int i,b;


	int A [10] = {84, 30, 95, 94, 36, 73, 52, 23, 2, 13};
	int S [10] = {0};

	#pragma omp target device(0) map(tofrom:A, b)
	#pragma omp teams 
	{
	    	int S_private[10] = {0};
    		#pragma omp distribute parallel for shared(b, A)
    		for (int n=0 ; n<10 ; ++n ) {
    			#pragma omp critical
 	   		{
            			b += A[n];
	    		}
		}
    	}
	printf("exit target teams\n");
}

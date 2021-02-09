#include <stdio.h>

int main(int argc, char *argv[])
{
	int result = 0;
    #pragma acc parallel
	{ 
    	#pragma acc loop gang reduction(+:result)
    	for (int m = 0; m < 100; m++) {
    		int res = -1000;
    		#pragma acc loop worker reduction(max:res)
    		for (int i = 0; i < 100; i++) {
        		int neg = -i - 1;
        		res = res > neg ? res : neg;
    		}
			result += res;
		}
	}

    printf("result = %d\n", result);
	if( result == -100 ) {
    	printf("Reduction test #1 passed!\n");
	} else {
    	printf("Reduction test #1 failed!\n");
	}
    return 0;
}


#include <stdio.h>

int main(int argc, char *argv[])
{
    int res = -1000;
    #pragma acc parallel loop reduction(max:res)
    for (int i = 0; i < 100; i++) {
        int neg = -1 * i - 1;
        res = res > neg ? res : neg;
    }

    printf("res = %d\n", res);
	if( res == -1 ) {
    	printf("Reduction test #1 passed!\n");
	} else {
    	printf("Reduction test #1 failed!\n");
	}
    return 0;
}


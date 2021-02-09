#include <stdio.h>

int main(int argc, char *argv[])
{
    int res = 0;
    #pragma acc parallel loop reduction(+:res) copy(res)
    for (int i = 0; i < 100; i++) {
        res += i;
    }

    printf("res = %d\n", res);
	if( res == 4950 ) {
    	printf("Reduction test #1 passed!\n");
	} else {
    	printf("Reduction test #1 failed!\n");
	}
    return 0;
}


#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <math.h>

int main(int argc, char *argv[])
{
	float ftemp;
	double dtemp;
	printf("INT_MIN = %d\n", INT_MIN);
	printf("INT_MAX = %d\n", INT_MAX);
	ftemp = FLT_MIN;
	printf("FLT_MIN = %e\n", ftemp);
	ftemp = FLT_MAX;
	printf("FLT_MAX = %e\n", ftemp);
	ftemp = -FLT_MAX;
	printf("-FLT_MAX = %e\n", ftemp);
	dtemp = DBL_MIN;
	printf("DBL_MIN = %e\n", dtemp);
	dtemp = DBL_MAX;
	printf("DBL_MAX = %e\n", dtemp);
	dtemp = -DBL_MAX;
	printf("-DBL_MAX = %e\n", dtemp);
	printf("INFINITY = %e\n", INFINITY);
	printf("-INFINITY = %e\n", -INFINITY);

	return 0;
}

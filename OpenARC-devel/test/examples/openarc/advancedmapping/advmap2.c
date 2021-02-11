#include <stdlib.h>
#include <stdio.h>

#ifndef MAPPING
#define MAPPING 1
#endif

#ifndef _NSIZE
#define _NSIZE 32
#endif
int a[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
int b[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
int c[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
int d[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
#ifdef VERIFICATION
int c_ref[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
int d_ref[_NSIZE][_NSIZE][_NSIZE][_NSIZE];
#endif

int main() {
	int i, j, k, l;
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					a[i][j][k][l] = i+j+k+l;
					b[i][j][k][l] = i*j+k+l;
					c[i][j][k][l] = 0;
					d[i][j][k][l] = 0;
#ifdef VERIFICATION
					c_ref[i][j][k][l] = 0;
					d_ref[i][j][k][l] = 0;
#endif
				}
			}
		}
	}

#pragma acc data copyin(a, b) copyout(c, d)
{

#if MAPPING == 1
#pragma acc parallel loop gang
	for(i=0; i<_NSIZE; i++) {
#pragma acc loop worker
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 2
#pragma acc parallel loop gang
	for(i=0; i<_NSIZE; i++) {
#pragma acc loop worker
		for(j=0; j<_NSIZE; j++) {
#pragma acc loop vector collapse(2)
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
#pragma acc loop vector collapse(2)
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 3
#pragma acc parallel loop gang
	for(i=0; i<_NSIZE; i++) {
#pragma acc loop worker
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector 
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 4
#pragma acc parallel loop gang worker collapse(2) 
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 5
#pragma acc parallel loop gang worker collapse(2) 
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
#pragma acc loop vector collapse(2)
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
#pragma acc loop vector collapse(2)
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 6
#pragma acc parallel loop gang worker collapse(2) 
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

#if MAPPING == 7
#pragma acc parallel loop gang collapse(2)
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
#pragma acc loop worker
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector
				for(l=0; l<_NSIZE; l++) {
					c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
#pragma acc loop worker
			for(k=0; k<_NSIZE; k++) {
#pragma acc loop vector
				for(l=0; l<_NSIZE; l++) {
					d[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
#endif

}

#ifdef VERIFICATION
printf(stdout, "Calculate reference CPU outputs!\n");
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					c_ref[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
				}
			}
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					d_ref[i][j][k][l] = a[i][j][k][l] - b[i][j][k][l];
				}
			}
		}
	}
	for(i=0; i<_NSIZE; i++) {
		for(j=0; j<_NSIZE; j++) {
			for(k=0; k<_NSIZE; k++) {
				for(l=0; l<_NSIZE; l++) {
					if(c_ref[i][j][k][l] != c[i][j][k][l]) {
						printf(stderr, "Error on the array c[%d][%d][%d][%d]\n", i, j, k, l);
						exit();
					}
					if(d_ref[i][j][k][l] != d[i][j][k][l]) {
						printf(stderr, "Error on the array d[%d][%d][%d][%d]\n", i, j, k, l);
						exit();
					}
				}
			}
		}
	}
	printf(stderr, "Verification is successful!\n");
#endif
	return 0;
}

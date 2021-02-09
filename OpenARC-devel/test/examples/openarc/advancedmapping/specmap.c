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
        }
      }
    }
  }

#pragma omp target data map(to:a, b) map(from:c, d)
  {
// 503 
// CPU: collapse(2), simd on innermost loop:
/*
    #pragma omp teams distribute parallel for private (i) collapse(2)
    for(j=0; j<_NSIZE; j++) {
      for(k=0; k<_NSIZE; k++) {
        #pragma omp simd
        for(l=0; l<_NSIZE; l++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
        }
      }
    }
*/
// GPU: No change 
#if MAPPING == 1
    i = 0;
    #pragma omp target teams distribute parallel for private (i) collapse(3)
    for(j=0; j<_NSIZE; j++) {
      for(k=0; k<_NSIZE; k++) {
        for(l=0; l<_NSIZE; l++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
        }
      }
    }
#endif
    
// 503 SIMD
// CPU: no change 
// GPU: collapse(3) no simd
/*
    #pragma omp teams distribute parallel for private (i) collapse(3)
    for(j=0; j<_NSIZE; j++) {
      for(k=0; k<_NSIZE; k++) {
        #pragma openarc transform vectorfriendly
        for(l=0; l<_NSIZE; l++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
        }
      }
    }
*/
#if MAPPING == 2
    i = 0;
    #pragma omp target teams distribute parallel for private (i) collapse(2)
    for(j=0; j<_NSIZE; j++) {
      for(k=0; k<_NSIZE; k++) {
        #pragma omp simd
        for(l=0; l<_NSIZE; l++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
        }
      }
    }
#endif

// 552
// CPU: parallel for on outer loops
/*
    #pragma omp target teams distribute parallel for
    for(k=0; k<_NSIZE; k++) {
      #pragma omp simd
      for(l=0; l<_NSIZE; l++) {
        c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
      #pragma omp simd
      for(l=0; l<_NSIZE; l++) {
        d[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
    }

    #pragma omp target teams distribute parallel for 
    for(k=0; k<_NSIZE; k++) {
      sum = d[0][0][k][0];

      #pragma omp simd //reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += d[i][j][k][l];
      }
    }
*/
// GPU: parallel for on inner loops
/*
    #pragma omp target teams distribute 
    for(k=0; k<_NSIZE; k++) {
      #pragma omp parallel for (requires identifying loop as parallelizable) 
      for(l=0; l<_NSIZE; l++) {
        c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
      #pragma omp parallel for (requires identifying loop as parallelizable) 
      for(l=0; l<_NSIZE; l++) {
        d[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
    }

    #pragma omp target teams distribute 
    for(k=0; k<_NSIZE; k++) {
      sum = d[0][0][k][0];

      #pragma omp parallel for simd //reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += d[i][j][k][l];
      }
    }
*/
#if MAPPING == 3
    i=0; j=0;
    int sum=0;
    
    #pragma omp target teams distribute parallel for
    for(k=0; k<_NSIZE; k++) {
      #pragma omp simd
      for(l=0; l<_NSIZE; l++) {
        c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
      #pragma omp simd
      for(l=0; l<_NSIZE; l++) {
        d[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      }
    }

    #pragma omp target teams distribute 
    for(k=0; k<_NSIZE; k++) {
      sum = d[0][0][k][0];

      #pragma omp parallel for simd //reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += d[i][j][k][l];
      }
    }
#endif

// 554
// CPU: no change 
// GPU: parallel for on inner loop
/*
    #pragma omp target teams distribute
    for(k=0; k<_NSIZE; k++) {
      sum = a[0][0][k][0];
      #pragma omp parallel for simd reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += c[i][j][k][l];
      }
    }

    #pragma omp target teams distribute 
    for(k=0; k<_NSIZE; k++) {
      sum = a[0][0][k][0];
      #pragma simd parallel for simd reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += d[i][j][k][l];
      }
    }
*/
#if MAPPING == 4
    i=0; j=0;
    int sum=0;
    
    #pragma omp target teams distribute parallel for
    for(k=0; k<_NSIZE; k++) {
      sum = a[0][0][k][0];
      #pragma omp simd reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += c[i][j][k][l];
      }
    }

    #pragma omp target teams distribute parallel for
    for(k=0; k<_NSIZE; k++) {
      sum = a[0][0][k][0];
      #pragma omp simd reduction(+:sum)
      for(l=0; l<_NSIZE; l++) {
        sum += d[i][j][k][l];
      }
    }
#endif

// 570 inner simd
#if MAPPING == 5
    i = 0;
	l = 0;
	int UB = _NSIZE;
    #pragma omp target teams distribute parallel for private (j,k)
    for(j=1; j<=UB; j++) {
      #pragma omp simd
      for(k=1; k<=UB; k++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      } /* end k */
    } /* end j */
#endif

// 570 without inner simd
#if MAPPING == 6
    i = 0;
	l = 0;
	int UB = _NSIZE;
    #pragma omp target teams distribute parallel for simd private (j,k) collapse(2)
    for(j=1; j<=UB; j++) {
      for(k=1; k<=UB; k++) {
          c[i][j][k][l] = a[i][j][k][l] + b[i][j][k][l];
      } /* end k */
    } /* end j */
#endif

  }
  return 0;
}

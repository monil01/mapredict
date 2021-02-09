#include <stdio.h>
#include <stdlib.h>

#define N 10
#define M 10

int main(int argc, char** argv) {
  int size = N*M; 

  float *A = (float*) malloc(size * sizeof(float));
  float *B = (float*) malloc(size * sizeof(float));
  float delta = 0;

  for (int i = 0; i < size; i++) {
    A[i] = (float) i;
  }
  
  #if 1
  // ---------------------------------------
  // Hoist 1. 
  // Redundant operations
  //
  // I think this should be doable, and OpenARC may already support something 
  //  like this. 
  #if 1
  int step = 2;
  int power = 3;
  // Before
  for (int i = 0; i < size; i++) {

    if (i == 0) 
      delta = step * (power + A[i] + A[i+1]);
    else if (i == size - 1) 
      delta = step * (power + A[i-1] + A[i]);
    else
      delta = step * (power + A[i-1] + A[i] + A[i+1]);

    B[i] = delta;
  }

  #else
  // After 
  int step = 2;
  int power = 3;
  for (int i = 0; i < size; i++) {
    int delta_tmp;
    if (i == 0) 
      delta_tmp = A[i] + A[i+1];
    else if (i == size - 1) 
      delta_tmp = A[i-1] + A[i];
    else
      delta_tmp = A[i-1] + A[i] + A[i+1];

    delta = step * (power + delta_tmp);

    B[i] = delta;
  }
  #endif
  #endif

  #if 0
  // ---------------------------------------
  // Hoist 2. 
  // Redundant formulas 
  //   Can we hoist a redundant formula with variables X, Y, Z?
  //
  // >  if (...) 
  // >     delta = step * (power + X_1*r_1 + Y_1*r_2 + Z_1*r_3);
  // >  else 
  // >     delta = step * (power + X_2*r_1 + Y_2*r_2 + Z_2*r_3);
  //
  // <  if (...) 
  // <     X = X_1; Y = Y_1; Z = Z_1;
  // <  else 
  // <     X = X_2; Y = Y_2; Z = Z_2;
  // 
  // <     delta = step * (power + X*r_1 + Y*r_2 + Z*r_3);
  #if 1
  // Before
  int step = 2;
  int power = 3;
  int r_1 = 4, r_2 = 5, r_3 = 6;
  for (int i = 0; i < size; i++) {

    if (i == 0) 
      delta = step * (power + A[i]*r_1 + A[i]*r_2 + A[i+1]*r_3);
    else if (i == size - 1) 
      delta = step * (power + A[i-1]*r_1 + A[i]*r_2 + A[i]*r_3);
    else
      delta = step * (power + A[i-1]*r_1 + A[i]*r_2 + A[i+1]*r_3);

    B[i] = delta;
  }
  #else
  // After 
  int step = 2;
  int power = 3;
  int r_1 = 4, r_2 = 5, r_3 = 6;
  for (int i = 0; i < size; i++) {

    int X, Y, Z;

    if (i == 0) {
      X = A[i]; Y = A[i]; Z = A[i+1];
    } else if (i == size - 1) {
      X = A[i-1]; Y = A[i]; Z = A[i];
    } else {
      X = A[i-1]; Y = A[i]; Z = A[i+1];
    }

    delta = step * (power + X*r_1 + Y*r_2 + Z*r_3);

    B[i] = delta;
  }
  #endif
  #endif

  int ver = 0;
  for (int i = 0; i < size; ++i) {
    ver += B[i];
  } 

  printf("ver: %d\n", ver);

  return 0;
}

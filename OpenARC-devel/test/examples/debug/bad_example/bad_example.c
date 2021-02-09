#include <stdio.h>
#include <stdlib.h>
#define R 12
#define C 16
#define N (R*C)

#pragma openarc #define R 12
#pragma openarc #define C 16
#pragma openarc #define N (R*C)

int main(int argc, char** argv) {

  int *input, *output;
  int verified = 1;

  input = (int *) malloc(N * sizeof(int));
  output = (int *) malloc(N * sizeof(int));
  int value;

  for (int i = 0; i < N; ++i) input[i] = i;

  #pragma acc parallel \
    copyin(input[0:N]) copyout(output[0:N]) 
  {
    #pragma acc loop collapse(2)
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < C; ++c) {

        int value = 0;
		int temp;

        value += input[r*C + c]; // C

        //output[r*C + c] = value;
        output[(temp = r*C, temp + c)] = value;
      } // c
    } // r
  } // acc

  // Verification
  
  for (int i = 0; i < N; ++i) {
    if (output[i] != input[i]) {
      printf("Verfication Failed:\n  output[%d] = %d\n  input[%d] = %d\n",
          i, output[i], i, input[i]);
      verified = 0;
      //exit(0);
    }
  }

  if( verified == 1 ) {
    printf("Verification Successful!\n");
  } else {
    printf("Verification Failed!\n");
  }

  return 0;
}

#include "defines.h"
#include "stdio.h"

#if SW_MULTI
void sobel_ACC(unsigned int * frame_in, 
               unsigned int * frame_out, 
               int iterations, int threshold)
{
  #pragma acc parallel num_gangs(1) num_workers(1) \
           copyin(threshold) \
           present(frame_in[0:ROWS*COLS], frame_out[0:ROWS*COLS]) 
  {
    #define SSIZE (16)
    #define TARGET (2*COLS + 2) 
    #define SW_BASE_SIZE (2*COLS + 2)
    #define SW_SIZE (SW_BASE_SIZE + SSIZE)

    int sw[SW_SIZE];

    #pragma unroll
    for (int i = 0; i < SW_SIZE; ++i) sw[i] = 0;
  
    // Filter coefficients
    int Gx[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int Gy[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

    for (int count = 0; count < ROWS*COLS; count += SSIZE) {

      // slide the window
      #pragma unroll
      for (int i = 0; i < SW_BASE_SIZE; ++i) 
        sw[i] = sw[i + SSIZE];

      // shift in new inputs
      int read_offset = count + SW_BASE_SIZE - TARGET;

      #pragma unroll
      for (int i = 0; i < SSIZE; ++i) {
        sw[SW_BASE_SIZE + i] = frame_in[read_offset + i];
      }

      int value[SSIZE];

      #pragma unroll
      for (int ss = 0; ss < SSIZE; ++ss) {

        int x_dir = 0;
        int y_dir = 0;

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
          #pragma unroll
          for (int j = 0; j < 3; ++j) {

            unsigned int pixel;
            if (count+ss - (i * COLS) - j >= 0) {
              pixel = sw[TARGET+ss +  0 - (i * COLS)  - j];
            } else {
              pixel = 0;
            }

            unsigned int b = pixel & 0xff;
            unsigned int g = (pixel >> 8) & 0xff;
            unsigned int r = (pixel >> 16) & 0xff;

            // RGB -> Luma conversion approximation
            // Avoiding floating point math operators greatly reduces
            // resource usage.
            unsigned int luma = r * 66 + g * 129 + b * 25;
            luma = (luma + 128) >> 8;
            luma += 16;

            x_dir += luma * Gx[i][j];
            y_dir += luma * Gy[i][j];
          }
        }

        int temp = abs(x_dir) + abs(y_dir);
        unsigned int clamped;
        if (temp > threshold) {
          clamped = 0xffffff;
        } else {
          clamped = 0;
        }

        value[ss] = clamped;
      } // exit SSIZE

      #pragma unroll
      for (int i = 0; i < SSIZE; ++i) {
        frame_out[count + i] = value[i];
      }

    } // exit main loop
  } // exit acc
}
#endif

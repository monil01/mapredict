#include "defines.h"
#include "stdio.h"

#if SW_BASE 
void sobel_ACC(unsigned int * frame_in, 
               unsigned int * frame_out, 
               int iterations, int threshold)
{


  #pragma acc parallel num_gangs(1) num_workers(1) \
           copyin(threshold) \
           present(frame_in[0:ROWS*COLS], frame_out[0:ROWS*COLS]) 
  {
    #define TARGET (2*COLS + 2) 
    #define SW_BASE_SIZE (2*COLS + 2)
    #define SW_SIZE (SW_BASE_SIZE + 1)

    int sw[SW_SIZE];

    #pragma unroll
    for (int i = 0; i < SW_SIZE; ++i) sw[i] = 0;
  
    // Filter coefficients
    int Gx[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    int Gy[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

    for (int count = 0; count < ROWS*COLS; count++) {

      #pragma unroll
      for (int i = 0; i < SW_BASE_SIZE; ++i) 
        sw[i] = sw[i + 1];

      int read_offset = count + SW_BASE_SIZE - TARGET;
      if (read_offset >= 0 && read_offset < ROWS*COLS) 
        sw[SW_BASE_SIZE] = frame_in[read_offset];
      else
        sw[SW_BASE_SIZE] = 0; 

      int x_dir = 0;
      int y_dir = 0;

      #pragma unroll
      for (int i = 0; i < 3; ++i) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {

          unsigned int pixel;
          if (count - (i * COLS) - j >= 0) {
            pixel = sw[TARGET + 0 - (i * COLS)  - j];
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

      if (count >= 0) {
        frame_out[count] = clamped;
      }
    }
  }
}
#endif

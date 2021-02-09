
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file.h"
#include "common.h"
#include "kernels.h"
#ifdef _OPENACC
#include "openacc.h"
#endif

static int generate_data(float *A0, int nx, int ny, int nz, int reset)
{
  srand(54321);
  int i,j,k;
  for(k=0;k<nz;k++)
    for(j=0;j<ny;j++)
      for(i=0;i<nx;i++)
	if(reset)
	  A0[Index3D(nx,ny,i,j,k)] = 0.0f;
	else
	  A0[Index3D(nx,ny,i,j,k)] = rand()/(float)RAND_MAX;
  return 0;
}



static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
  int s=0;
  int i, j, k;
  for(i=0;i<nz;i++)
    {
      for(j=0;j<ny;j++)
	{
	  for(k=0;k<nx;k++)
	    {
	      fread(A0+s,sizeof(float),1,fp);
	      s++;
	    }
	}
    }
  return 0;
}

int main(int argc, char** argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;
       
	
  printf("CPU-based 7 points stencil codes****\n");
  printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
  printf("This version maintained by Chris Rodrigues  ***********\n");
  parameters = pb_ReadParameters(&argc, argv);

  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
  //declaration
  int nx,ny,nz;
  int size;
  int iteration;
  //float c0=1.0f/6.0f;
  //float c1=1.0f/6.0f/6.0f;
  float c1=1.0f/7.0f;
  float c0=-1.0f/7.0f;


  if (argc<5) 
    {
      printf("Usage: ostencil nx ny nz t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
	     "t: the iteration time\n");
      return -1;
    }

  nx = atoi(argv[1]);
	
  if (nx<1) {
    printf("ERROR: nx=%d\n",nx);
    return -1;
  }
  ny = atoi(argv[2]);
  if (ny<1) {
    printf("ERROR: ny=%d\n",ny);
    return -1;
  }
  nz = atoi(argv[3]);
  if (nz<1) {
    printf("ERROR: nz=%d\n",nz);
    return -1;
  }
  iteration = atoi(argv[4]);
  if(iteration<1) {
    printf("ERROR: iteration=%d\n",iteration);
    return -1;
  }

	
  //host data
  float *h_A0;
  float *h_Anext;

  size=nx*ny*nz;
	
#ifdef USE_UNIFIEDMEM
  h_A0=(float*)acc_create_unified(NULL, sizeof(float)*size);
  h_Anext=(float*)acc_create_unified(NULL, sizeof(float)*size);
#else
  h_A0=(float*)malloc(sizeof(float)*size);
  h_Anext=(float*)malloc(sizeof(float)*size);
#endif

  /*
    FILE *fp = fopen(parameters->inpFiles[0], "rb");
    read_data(h_A0, nx,ny,nz,fp);
    fclose(fp);
    memcpy (h_Anext,h_A0 ,sizeof(float)*size);
  */
  generate_data(h_A0,nx,ny,nz,0);
  generate_data(h_Anext,nx,ny,nz,0);

  int t;
#pragma acc data copy(h_A0[0:size],  h_Anext[0:size])
  {
    for(t=0;t<iteration;t++)
      {
	cpu_stencil(c0, c1, h_A0, h_Anext, nx, ny,  nz);
	float *temp=h_A0;
	h_A0 = h_Anext;
	h_Anext = temp;
      }
  }

  if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
#ifdef USE_UNIFIEDMEM
  acc_delete_unified (h_A0, sizeof(float)*size);
  acc_delete_unified (h_Anext, sizeof(float)*size);
#else
  free (h_A0);
  free (h_Anext);
#endif
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}

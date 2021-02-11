/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"

void cpu_stencil(float c0,float c1, float *A0,float * Anext,const int nx, const int ny, const int nz)
{

  int i, j, k;  
#ifdef USE_ASYNC
#pragma acc kernels  pcopyin(A0[0:nx*ny*nz]), pcopyout(Anext[0:nx*ny*nz]) async(0)
#else
#pragma acc kernels  pcopyin(A0[0:nx*ny*nz]), pcopyout(Anext[0:nx*ny*nz])
#endif
  {
#pragma acc loop independent gang worker
	for(k=1;k<nz-1;k++)
	{
#pragma acc loop independent gang worker
		for(j=1;j<ny-1;j++)
		{
#pragma acc loop independent gang worker
			for(i=1;i<nx-1;i++)
			{
				Anext[Index3D (nx, ny, i, j, k)] = 
				(A0[Index3D (nx, ny, i, j, k + 1)] +
				A0[Index3D (nx, ny, i, j, k - 1)] +
				A0[Index3D (nx, ny, i, j + 1, k)] +
				A0[Index3D (nx, ny, i, j - 1, k)] +
				A0[Index3D (nx, ny, i + 1, j, k)] +
				A0[Index3D (nx, ny, i - 1, j, k)])*c1
				- A0[Index3D (nx, ny, i, j, k)]*c0;
			}
		}
	}
  }
#ifdef USE_ASYNC
#pragma acc wait(0)
#endif

}



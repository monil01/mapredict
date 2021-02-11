/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#if defined(__APPLE__)
# include <machine/byte_order.h>
#else
# include <endian.h>
#endif
#include <stdlib.h>
#if !defined(__APPLE__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <inttypes.h>

#include "file.h"

#ifdef _OPENACC
#include "openacc.h"
#endif

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

/*extern "C"*/
void inputData(char* fName, int* _numK, int* _numX,
               float** kx, float** ky, float** kz,
               float** x, float** y, float** z,
               float** phiR, float** phiI)
{
  int numK, numX;
  FILE* fid = fopen(fName, "r");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
  fread (&numK, sizeof (int), 1, fid);
  *_numK = numK;
  fread (&numX, sizeof (int), 1, fid);
  *_numX = numX;
#ifdef USE_UNIFIEDMEM
  *kx = (float *) acc_create_unified(NULL, numK * sizeof (float));
  *ky = (float *) acc_create_unified(NULL, numK * sizeof (float));
  *kz = (float *) acc_create_unified(NULL, numK * sizeof (float));
  *x = (float *) acc_create_unified(NULL, numX * sizeof (float));
  *y = (float *) acc_create_unified(NULL, numX * sizeof (float));
  *z = (float *) acc_create_unified(NULL, numX * sizeof (float));
  *phiR = (float *) acc_create_unified(NULL, numK * sizeof (float));
  *phiI = (float *) acc_create_unified(NULL, numK * sizeof (float));
#else
  *kx = (float *) memalign(16, numK * sizeof (float));
  *ky = (float *) memalign(16, numK * sizeof (float));
  *kz = (float *) memalign(16, numK * sizeof (float));
  *x = (float *) memalign(16, numX * sizeof (float));
  *y = (float *) memalign(16, numX * sizeof (float));
  *z = (float *) memalign(16, numX * sizeof (float));
  *phiR = (float *) memalign(16, numK * sizeof (float));
  *phiI = (float *) memalign(16, numK * sizeof (float));
#endif
  fread (*kx, sizeof (float), numK, fid);
  fread (*ky, sizeof (float), numK, fid);
  fread (*kz, sizeof (float), numK, fid);
  fread (*x, sizeof (float), numX, fid);
  fread (*y, sizeof (float), numX, fid);
  fread (*z, sizeof (float), numX, fid);
  fread (*phiR, sizeof (float), numK, fid);
  fread (*phiI, sizeof (float), numK, fid);
  fclose (fid); 
}

/*extern "C"*/
void outputData(char* fName, float* outR, float* outI, int numX)
{
  FILE* fid = fopen(fName, "w");
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }

  /* Write the data size */
  fprintf (fid, "%d\n", numX);

  /* Write the reconstructed data */
  int x;
  for (x = 0; x < numX; ++x)
    fprintf (fid, "%f,%f\n",  outR[x], outI[x]);
  fclose (fid);
}

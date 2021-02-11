/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <stdlib.h>
#if !defined(__APPLE__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <string.h>
#include <math.h>
#define SPEC_NO_INLINE
#ifdef SPEC_NO_INLINE
#define INLINE 
#else
#define INLINE inline
#endif

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

INLINE
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI, float* phiMag);

INLINE
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi);

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi);

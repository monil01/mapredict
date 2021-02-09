/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include "computeQ.h"
#ifdef _OPENACC
#include "openacc.h"
#endif

INLINE
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI, float* phiMag) {
  int indexK = 0;
#ifdef USE_ASYNC
#pragma acc parallel loop pcopyout(phiMag[0:numK]) pcopyin(phiR[0:numK],phiI[0:numK]) async(0)
#else
#pragma acc parallel loop pcopyout(phiMag[0:numK]) pcopyin(phiR[0:numK],phiI[0:numK])
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
#ifdef USE_ASYNC
#pragma acc wait(0)
#endif
}

INLINE
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;
#ifdef USE_ASYNC
#pragma acc parallel pcopyin(kVals[0:numK], x[0:numX], y[0:numX], z[0:numX]), \
                     pcopy(Qr[0:numX], Qi[0:numX]) async(0) 
#else
#pragma acc parallel pcopyin(kVals[0:numK], x[0:numX], y[0:numX], z[0:numX]), \
                     pcopy(Qr[0:numX], Qi[0:numX]) 
#endif
  {
#pragma acc loop gang
    for (indexX = 0; indexX < numX; indexX++) {

      float QrSum = 0.0;
      float QiSum = 0.0;
      
#pragma acc loop vector reduction(+:QrSum), reduction(+:QiSum) 
      for (indexK = 0; indexK < numK; indexK++) {
	expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
			 kVals[indexK].Ky * y[indexX] +
			 kVals[indexK].Kz * z[indexX]);

	cosArg = cosf(expArg);
	sinArg = sinf(expArg);

	float phi = kVals[indexK].PhiMag;
	QrSum += phi * cosArg;
	QiSum += phi * sinArg;
      }

      Qr[indexX] += QrSum;
      Qi[indexX] += QiSum;
    }
  }
#ifdef USE_ASYNC
#pragma acc wait(0)
#endif
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
#ifdef USE_UNIFIEDMEM
  *phiMag = (float* ) acc_create_unified(NULL, numK * sizeof(float));
  *Qr = (float*) acc_create_unified(NULL, numX * sizeof (float));
  *Qi = (float*) acc_create_unified(NULL, numX * sizeof (float));
#else
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
#endif
  memset((void *)*Qr, 0, numX * sizeof(float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}

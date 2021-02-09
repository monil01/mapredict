/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef __MINGW32__
# if defined(__APPLE__)
#  include <machine/byte_order.h>
# else
#include <endian.h>
# endif
#endif
#include <stdlib.h>
#if !defined(__APPLE__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <inttypes.h>
#include <common.h>

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#ifdef __cplusplus
extern "C"
#endif
void inputData(char* fName, int* nx, int* ny, int* nz)
{
  FILE* fid = fopen(fName, "r");

  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open input file\n");
      exit(-1);
    }
	
  fread(nx, sizeof(int ),1,fid);
  fread(ny, sizeof(int ),1,fid);
  fread(nz, sizeof(int ),1,fid);
  fclose (fid); 
}

#ifdef __cplusplus
extern "C"
#endif
void outputData(char* fName, float *h_A0,int nx,int ny,int nz)
{
  FILE* fid = fopen(fName, "w");
  uint32_t tmp32;
  int i,j,k;
  if (fid == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }
  tmp32 = nx*ny*nz;
  for (i = 0; i < nz; i+=2) {
    for (j = 0; j < ny; j+=2) {
      for (k = 0; k < nx; k+=2) {
	fprintf(fid, "%f\n", h_A0[((i*ny)+j)*nx + k]);
      }
    }
  }
  fclose (fid);
}

#ifdef __cplusplus
extern "C"
#endif
char* readFile(const char* fileName)
  {
        FILE* fp;
	long size;
	char* buffer;
	size_t res;

        fp = fopen(fileName,"r");
        if(fp == NULL)
        {
                printf("Error 1!\n");
                exit(1);
        }

        fseek(fp,0,SEEK_END);
        size = ftell(fp);
        rewind(fp);

        buffer = (char*)malloc(sizeof(char)*(size+1));
        if(buffer  == NULL)
        {
                printf("Error 2!\n");
                fclose(fp);
                exit(1);
        }

        res = fread(buffer,1,size,fp);
        if(res != size)
        {
                printf("Error 3!\n");
                fclose(fp);
                exit(1);
        }

	buffer[size] = 0;
        fclose(fp);
        return buffer;
}

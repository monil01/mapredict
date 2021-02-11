/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include "sparselu.h"


/***********************************************************************
 * lu0:
 **********************************************************************/
void lu0(float *diag, int submatrix_size)
{
    int i, j, k;

    for (k=0; k<submatrix_size; k++)
        for (i=k+1; i<submatrix_size; i++)
        {
            diag[i+submatrix_size*k] = diag[i+submatrix_size*k] / diag[k+submatrix_size*k];
            for (j=k+1; j<submatrix_size; j++)
                diag[i+submatrix_size*j] = diag[i+submatrix_size*j] - diag[i+submatrix_size*k] * diag[k+submatrix_size*j];
        }
}

/***********************************************************************
 * bdiv:
 **********************************************************************/
void bdiv(float *diag, float *row, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (k=0; k<submatrix_size; k++)
        {
            row[i+submatrix_size*k] = row[i+submatrix_size*k] / diag[k+submatrix_size*k];
            for (j=k+1; j<submatrix_size; j++)
                row[i+submatrix_size*j] = row[i+submatrix_size*j] - row[i+submatrix_size*k]*diag[k+submatrix_size*j];
        }
}
/***********************************************************************
 * bmod:
 **********************************************************************/
void bmod(float *row, float *col, float *inner, int submatrix_size)
{
    int i, j, k;
    for (i=0; i<submatrix_size; i++)
        for (j=0; j<submatrix_size; j++)
            for (k=0; k<submatrix_size; k++)
                inner[i+submatrix_size*j] = inner[i+submatrix_size*j] - row[i+submatrix_size*k]*col[k+submatrix_size*j];
}
/***********************************************************************
 * fwd:
 **********************************************************************/
void fwd(float *diag, float *col, int submatrix_size)
{
    int i, j, k;
    for (j=0; j<submatrix_size; j++)
        for (k=0; k<submatrix_size; k++)
            for (i=k+1; i<submatrix_size; i++)
                col[i+submatrix_size*j] = col[i+submatrix_size*j] - diag[i+submatrix_size*k]*col[k+submatrix_size*j];
}



void sparselu_seq_call(float *BENCH, int matrix_size, int submatrix_size)
{
   int ii, jj, kk;

   for (kk=0; kk<matrix_size; kk++)
   {
      lu0(&BENCH[kk*matrix_size+kk], submatrix_size);
      for (jj=kk+1; jj<matrix_size; jj++)
//         if (BENCH[kk*matrix_size+jj] != NULL)
         {
            fwd(&BENCH[kk*matrix_size+kk], &BENCH[kk*matrix_size+jj], submatrix_size);
         }
      for (ii=kk+1; ii<matrix_size; ii++)
//         if (BENCH[ii*matrix_size+kk] != NULL)
         {
            bdiv (&BENCH[kk*matrix_size+kk], &BENCH[ii*matrix_size+kk], submatrix_size);
         }
      for (ii=kk+1; ii<matrix_size; ii++)
//         if (BENCH[ii*matrix_size+kk] != NULL)
            for (jj=kk+1; jj<matrix_size; jj++)
//               if (BENCH[kk*matrix_size+jj] != NULL)
               {
                     bmod(&BENCH[ii*matrix_size+kk], &BENCH[kk*matrix_size+jj], &BENCH[ii*matrix_size+jj], submatrix_size);
               }

   }
}

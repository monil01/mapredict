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


void sparselu_par_call(float **BENCH, int matrix_size, int submatrix_size)
{
    int ii, jj, kk;
	int nblocks;
	int submatrix_sqr;
	nblocks = matrix_size/submatrix_size;
	submatrix_sqr = submatrix_size*submatrix_size;
#pragma omp parallel private(kk,ii,jj) shared(BENCH)
#pragma omp single /* nowait */
    {
        /*#pragma omp task untied*/
        for (kk=0; kk<nblocks; kk++)
        {
			// Apply LU factorization to the panel ??
			int diagBlockStart = kk*submatrix_size*matrix_size+kk*submatrix_size;
#pragma omp task firstprivate(kk) shared(BENCH) depend(inout: BENCH[diagBlockStart:submatrix_sqr)
			lu0(BENCH[diagBlockStart], submatrix_size);
			
			// update trailing submatrix (towards right)
            for (jj=kk+1; jj<nblocks; jj++)
				int colBlockStart = kk*submatrix_size*matrix_size+jj*submatrix_size;
                if (BENCH[colBlockStart] != NULL)
                {
#pragma omp task firstprivate(kk, jj) shared(BENCH) depend(in: BENCH[diagBlockStart:submatrix_size*submatrix_size]) depend(inout: BENCH[colBlockStart:submatrix_size*submatrix_size])
                    fwd(BENCH[diagBlockStart], BENCH[colBlockStart], submatrix_size);
                }
				
			// update panel
            for (ii=kk+1; ii<matrix_size; ii++)
				int rowBlockStart = ii*submatrix_size*matrix_size+kk*submatrix_size;

                if (BENCH[rowBlockStart] != NULL)
                {
#pragma omp task firstprivate(kk, ii) shared(BENCH) depend(in: BENCH[diagBlockStart:submatrix_size*submatrix_size]) depend(inout: BENCH[rowBlockStart:submatrix_size*submatrix_size])
                    bdiv (BENCH[diagBlockStart], BENCH[rowBlockStart], submatrix_size);
                }
			// update submatrix
            for (ii=kk+1; ii<matrix_size; ii++){
 				int rowBlockStart = ii*submatrix_size*matrix_size+kk*submatrix_size;
              if (BENCH[rowBlockStart] != NULL)
                    for (jj=kk+1; jj<matrix_size; jj++){
						int colBlockStart = kk*submatrix_size*matrix_size+jj*submatrix_size;
                       if (BENCH[colBlockStart] != NULL)
                        {
							int subBlockStart = ii*submatrix_size*matrix_size+jj*submatrix_size;
                            if (BENCH[subBlockStart]==NULL) 
								BENCH[subBlockStart] = allocate_clean_block(submatrix_size);
#pragma omp task firstprivate(kk, jj, ii) shared(BENCH) \
                            depend(in: BENCH[rowBlockStart:submatrix_size*submatrix_size], BENCH[colBlockStart:submatrix_size*submatrix_size]) \
                            depend(inout: BENCH[subBlockStart:submatrix_size*submatrix_size])
                            bmod(BENCH[rowBlockStart], BENCH[colBlockStart], BENCH[subBlockStart], submatrix_size);
                        }
					}
			}
        }
#pragma omp taskwait
    }
}

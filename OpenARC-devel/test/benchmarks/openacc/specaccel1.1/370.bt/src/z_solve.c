//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB BT code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

#include "header.h"
#include "work_lhs.h"
//#include "timers.h"
#ifdef _OPENACC
#include "openacc.h"
#endif

//---------------------------------------------------------------------
// Performs line solves in Z direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
//---------------------------------------------------------------------
void z_solve()
{
  int i, j, k, m, n, ksize, z;
  double pivot, coeff;
  int gp12, gp02;
/*
  double fjacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double njacZ[5][5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double lhsZ[5][5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1];
*/
  double temp1, temp2, temp3;
  double (*fjacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double (*njacZ)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1];
  double (*lhsZ)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1];
#ifdef USE_UNIFIEDMEM
  fjacZ = (double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1])acc_create_unified(NULL, sizeof(double)*5*5*(PROBLEM_SIZE+1)*(IMAXP-1)*(JMAXP-1));
  njacZ = (double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1])acc_create_unified(NULL, sizeof(double)*5*5*(PROBLEM_SIZE+1)*(IMAXP-1)*(JMAXP-1));
  lhsZ = (double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1])acc_create_unified(NULL, sizeof(double)*5*5*3*(PROBLEM_SIZE)*(IMAXP-1)*(JMAXP-1));
#else
  fjacZ = (double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1])malloc(sizeof(double)*5*5*(PROBLEM_SIZE+1)*(IMAXP-1)*(JMAXP-1));
  njacZ = (double (*)[5][PROBLEM_SIZE+1][IMAXP-1][JMAXP-1])malloc(sizeof(double)*5*5*(PROBLEM_SIZE+1)*(IMAXP-1)*(JMAXP-1));
  lhsZ = (double (*)[5][3][PROBLEM_SIZE][IMAXP-1][JMAXP-1])malloc(sizeof(double)*5*5*3*(PROBLEM_SIZE)*(IMAXP-1)*(JMAXP-1));
#endif

  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

  //---------------------------------------------------------------------
  // This function computes the left hand side for the three z-factors   
  //---------------------------------------------------------------------

  ksize = grid_points[2]-1;

  //---------------------------------------------------------------------
  // Compute the indices for storing the block-diagonal matrix;
  // determine c (labeled f) and s jacobians
  //---------------------------------------------------------------------
#pragma acc data present(u,rhs,square,qs) pcreate(njacZ[0:5][0:5][0:PROBLEM_SIZE+1][0:IMAXP-1][0:JMAXP-1],fjacZ[0:5][0:5][0:PROBLEM_SIZE+1][0:IMAXP-1][0:JMAXP-1],lhsZ[0:5][0:5][0:3][0:PROBLEM_SIZE][0:IMAXP-1][0:JMAXP-1])
{
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
      for (k = 0; k <= ksize; k++) {
        temp1 = 1.0 / u[k][j][i][0];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;

        fjacZ[0][0][k][i][j] = 0.0;
        fjacZ[0][1][k][i][j] = 0.0;
        fjacZ[0][2][k][i][j] = 0.0;
        fjacZ[0][3][k][i][j] = 1.0;
        fjacZ[0][4][k][i][j] = 0.0;

        fjacZ[1][0][k][i][j] = - ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
        fjacZ[1][1][k][i][j] = u[k][j][i][3] * temp1;
        fjacZ[1][2][k][i][j] = 0.0;
        fjacZ[1][3][k][i][j] = u[k][j][i][1] * temp1;
        fjacZ[1][4][k][i][j] = 0.0;

        fjacZ[2][0][k][i][j] = - ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
        fjacZ[2][1][k][i][j] = 0.0;
        fjacZ[2][2][k][i][j] = u[k][j][i][3] * temp1;
        fjacZ[2][3][k][i][j] = u[k][j][i][2] * temp1;
        fjacZ[2][4][k][i][j] = 0.0;

        fjacZ[3][0][k][i][j] = - (u[k][j][i][3]*u[k][j][i][3] * temp2 ) 
          + c2 * qs[k][j][i];
        fjacZ[3][1][k][i][j] = - c2 *  u[k][j][i][1] * temp1;
        fjacZ[3][2][k][i][j] = - c2 *  u[k][j][i][2] * temp1;
        fjacZ[3][3][k][i][j] = ( 2.0 - c2 ) *  u[k][j][i][3] * temp1;
        fjacZ[3][4][k][i][j] = c2;

        fjacZ[4][0][k][i][j] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
          * u[k][j][i][3] * temp2;
        fjacZ[4][1][k][i][j] = - c2 * ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
        fjacZ[4][2][k][i][j] = - c2 * ( u[k][j][i][2]*u[k][j][i][3] ) * temp2;
        fjacZ[4][3][k][i][j] = c1 * ( u[k][j][i][4] * temp1 )
          - c2 * ( qs[k][j][i] + u[k][j][i][3]*u[k][j][i][3] * temp2 );
        fjacZ[4][4][k][i][j] = c1 * u[k][j][i][3] * temp1;

        njacZ[0][0][k][i][j] = 0.0;
        njacZ[0][1][k][i][j] = 0.0;
        njacZ[0][2][k][i][j] = 0.0;
        njacZ[0][3][k][i][j] = 0.0;
        njacZ[0][4][k][i][j] = 0.0;

        njacZ[1][0][k][i][j] = - c3c4 * temp2 * u[k][j][i][1];
        njacZ[1][1][k][i][j] =   c3c4 * temp1;
        njacZ[1][2][k][i][j] =   0.0;
        njacZ[1][3][k][i][j] =   0.0;
        njacZ[1][4][k][i][j] =   0.0;

        njacZ[2][0][k][i][j] = - c3c4 * temp2 * u[k][j][i][2];
        njacZ[2][1][k][i][j] =   0.0;
        njacZ[2][2][k][i][j] =   c3c4 * temp1;
        njacZ[2][3][k][i][j] =   0.0;
        njacZ[2][4][k][i][j] =   0.0;

        njacZ[3][0][k][i][j] = - con43 * c3c4 * temp2 * u[k][j][i][3];
        njacZ[3][1][k][i][j] =   0.0;
        njacZ[3][2][k][i][j] =   0.0;
        njacZ[3][3][k][i][j] =   con43 * c3 * c4 * temp1;
        njacZ[3][4][k][i][j] =   0.0;

        njacZ[4][0][k][i][j] = - (  c3c4
            - c1345 ) * temp3 * (u[k][j][i][1]*u[k][j][i][1])
          - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][2]*u[k][j][i][2])
          - ( con43 * c3c4
              - c1345 ) * temp3 * (u[k][j][i][3]*u[k][j][i][3])
          - c1345 * temp2 * u[k][j][i][4];

        njacZ[4][1][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[k][j][i][1];
        njacZ[4][2][k][i][j] = (  c3c4 - c1345 ) * temp2 * u[k][j][i][2];
        njacZ[4][3][k][i][j] = ( con43 * c3c4
            - c1345 ) * temp2 * u[k][j][i][3];
        njacZ[4][4][k][i][j] = ( c1345 )* temp1;
      }
	}
  }
      //---------------------------------------------------------------------
      // now jacobians set, so form left hand side in z direction
      //---------------------------------------------------------------------
      //lhsZ[j][i]init(lhsZ[j][i], ksize);
  // zero the whole left hand side for starters
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(4) async(0)
#else
  #pragma acc kernels loop independent collapse(4)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
  		for (n = 0; n < 5; n++) {
    		for (m = 0; m < 5; m++) {
      			lhsZ[m][n][0][0][i][j] = 0.0;
      			lhsZ[m][n][1][0][i][j] = 0.0;
      			lhsZ[m][n][2][0][i][j] = 0.0;
      			lhsZ[m][n][0][ksize][i][j] = 0.0;
      			lhsZ[m][n][1][ksize][i][j] = 0.0;
      			lhsZ[m][n][2][ksize][i][j] = 0.0;
    		}
  		}
	}
  }

  // next, set all diagonal values to 1. This is overkill, but convenient
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
    		lhsZ[0][0][1][0][i][j] = 1.0;
    		lhsZ[0][0][1][ksize][i][j] = 1.0;
    		lhsZ[1][1][1][0][i][j] = 1.0;
    		lhsZ[1][1][1][ksize][i][j] = 1.0;
    		lhsZ[2][2][1][0][i][j] = 1.0;
    		lhsZ[2][2][1][ksize][i][j] = 1.0;
    		lhsZ[3][3][1][0][i][j] = 1.0;
    		lhsZ[3][3][1][ksize][i][j] = 1.0;
    		lhsZ[4][4][1][0][i][j] = 1.0;
    		lhsZ[4][4][1][ksize][i][j] = 1.0;
	}
  }

#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
      for (k = 1; k <= ksize-1; k++) {
        temp1 = dt * tz1;
        temp2 = dt * tz2;

        lhsZ[0][0][AA][k][i][j] = - temp2 * fjacZ[0][0][k-1][i][j]
          - temp1 * njacZ[0][0][k-1][i][j]
          - temp1 * dz1; 
        lhsZ[0][1][AA][k][i][j] = - temp2 * fjacZ[0][1][k-1][i][j]
          - temp1 * njacZ[0][1][k-1][i][j];
        lhsZ[0][2][AA][k][i][j] = - temp2 * fjacZ[0][2][k-1][i][j]
          - temp1 * njacZ[0][2][k-1][i][j];
        lhsZ[0][3][AA][k][i][j] = - temp2 * fjacZ[0][3][k-1][i][j]
          - temp1 * njacZ[0][3][k-1][i][j];
        lhsZ[0][4][AA][k][i][j] = - temp2 * fjacZ[0][4][k-1][i][j]
          - temp1 * njacZ[0][4][k-1][i][j];

        lhsZ[1][0][AA][k][i][j] = - temp2 * fjacZ[1][0][k-1][i][j]
          - temp1 * njacZ[1][0][k-1][i][j];
        lhsZ[1][1][AA][k][i][j] = - temp2 * fjacZ[1][1][k-1][i][j]
          - temp1 * njacZ[1][1][k-1][i][j]
          - temp1 * dz2;
        lhsZ[1][2][AA][k][i][j] = - temp2 * fjacZ[1][2][k-1][i][j]
          - temp1 * njacZ[1][2][k-1][i][j];
        lhsZ[1][3][AA][k][i][j] = - temp2 * fjacZ[1][3][k-1][i][j]
          - temp1 * njacZ[1][3][k-1][i][j];
        lhsZ[1][4][AA][k][i][j] = - temp2 * fjacZ[1][4][k-1][i][j]
          - temp1 * njacZ[1][4][k-1][i][j];

        lhsZ[2][0][AA][k][i][j] = - temp2 * fjacZ[2][0][k-1][i][j]
          - temp1 * njacZ[2][0][k-1][i][j];
        lhsZ[2][1][AA][k][i][j] = - temp2 * fjacZ[2][1][k-1][i][j]
          - temp1 * njacZ[2][1][k-1][i][j];
        lhsZ[2][2][AA][k][i][j] = - temp2 * fjacZ[2][2][k-1][i][j]
          - temp1 * njacZ[2][2][k-1][i][j]
          - temp1 * dz3;
        lhsZ[2][3][AA][k][i][j] = - temp2 * fjacZ[2][3][k-1][i][j]
          - temp1 * njacZ[2][3][k-1][i][j];
        lhsZ[2][4][AA][k][i][j] = - temp2 * fjacZ[2][4][k-1][i][j]
          - temp1 * njacZ[2][4][k-1][i][j];

        lhsZ[3][0][AA][k][i][j] = - temp2 * fjacZ[3][0][k-1][i][j]
          - temp1 * njacZ[3][0][k-1][i][j];
        lhsZ[3][1][AA][k][i][j] = - temp2 * fjacZ[3][1][k-1][i][j]
          - temp1 * njacZ[3][1][k-1][i][j];
        lhsZ[3][2][AA][k][i][j] = - temp2 * fjacZ[3][2][k-1][i][j]
          - temp1 * njacZ[3][2][k-1][i][j];
        lhsZ[3][3][AA][k][i][j] = - temp2 * fjacZ[3][3][k-1][i][j]
          - temp1 * njacZ[3][3][k-1][i][j]
          - temp1 * dz4;
        lhsZ[3][4][AA][k][i][j] = - temp2 * fjacZ[3][4][k-1][i][j]
          - temp1 * njacZ[3][4][k-1][i][j];

        lhsZ[4][0][AA][k][i][j] = - temp2 * fjacZ[4][0][k-1][i][j]
          - temp1 * njacZ[4][0][k-1][i][j];
        lhsZ[4][1][AA][k][i][j] = - temp2 * fjacZ[4][1][k-1][i][j]
          - temp1 * njacZ[4][1][k-1][i][j];
        lhsZ[4][2][AA][k][i][j] = - temp2 * fjacZ[4][2][k-1][i][j]
          - temp1 * njacZ[4][2][k-1][i][j];
        lhsZ[4][3][AA][k][i][j] = - temp2 * fjacZ[4][3][k-1][i][j]
          - temp1 * njacZ[4][3][k-1][i][j];
        lhsZ[4][4][AA][k][i][j] = - temp2 * fjacZ[4][4][k-1][i][j]
          - temp1 * njacZ[4][4][k-1][i][j]
          - temp1 * dz5;

        lhsZ[0][0][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[0][0][k][i][j]
          + temp1 * 2.0 * dz1;
        lhsZ[0][1][BB][k][i][j] = temp1 * 2.0 * njacZ[0][1][k][i][j];
        lhsZ[0][2][BB][k][i][j] = temp1 * 2.0 * njacZ[0][2][k][i][j];
        lhsZ[0][3][BB][k][i][j] = temp1 * 2.0 * njacZ[0][3][k][i][j];
        lhsZ[0][4][BB][k][i][j] = temp1 * 2.0 * njacZ[0][4][k][i][j];

        lhsZ[1][0][BB][k][i][j] = temp1 * 2.0 * njacZ[1][0][k][i][j];
        lhsZ[1][1][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[1][1][k][i][j]
          + temp1 * 2.0 * dz2;
        lhsZ[1][2][BB][k][i][j] = temp1 * 2.0 * njacZ[1][2][k][i][j];
        lhsZ[1][3][BB][k][i][j] = temp1 * 2.0 * njacZ[1][3][k][i][j];
        lhsZ[1][4][BB][k][i][j] = temp1 * 2.0 * njacZ[1][4][k][i][j];

        lhsZ[2][0][BB][k][i][j] = temp1 * 2.0 * njacZ[2][0][k][i][j];
        lhsZ[2][1][BB][k][i][j] = temp1 * 2.0 * njacZ[2][1][k][i][j];
        lhsZ[2][2][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[2][2][k][i][j]
          + temp1 * 2.0 * dz3;
        lhsZ[2][3][BB][k][i][j] = temp1 * 2.0 * njacZ[2][3][k][i][j];
        lhsZ[2][4][BB][k][i][j] = temp1 * 2.0 * njacZ[2][4][k][i][j];

        lhsZ[3][0][BB][k][i][j] = temp1 * 2.0 * njacZ[3][0][k][i][j];
        lhsZ[3][1][BB][k][i][j] = temp1 * 2.0 * njacZ[3][1][k][i][j];
        lhsZ[3][2][BB][k][i][j] = temp1 * 2.0 * njacZ[3][2][k][i][j];
        lhsZ[3][3][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[3][3][k][i][j]
          + temp1 * 2.0 * dz4;
        lhsZ[3][4][BB][k][i][j] = temp1 * 2.0 * njacZ[3][4][k][i][j];

        lhsZ[4][0][BB][k][i][j] = temp1 * 2.0 * njacZ[4][0][k][i][j];
        lhsZ[4][1][BB][k][i][j] = temp1 * 2.0 * njacZ[4][1][k][i][j];
        lhsZ[4][2][BB][k][i][j] = temp1 * 2.0 * njacZ[4][2][k][i][j];
        lhsZ[4][3][BB][k][i][j] = temp1 * 2.0 * njacZ[4][3][k][i][j];
        lhsZ[4][4][BB][k][i][j] = 1.0
          + temp1 * 2.0 * njacZ[4][4][k][i][j] 
          + temp1 * 2.0 * dz5;

        lhsZ[0][0][CC][k][i][j] =  temp2 * fjacZ[0][0][k+1][i][j]
          - temp1 * njacZ[0][0][k+1][i][j]
          - temp1 * dz1;
        lhsZ[0][1][CC][k][i][j] =  temp2 * fjacZ[0][1][k+1][i][j]
          - temp1 * njacZ[0][1][k+1][i][j];
        lhsZ[0][2][CC][k][i][j] =  temp2 * fjacZ[0][2][k+1][i][j]
          - temp1 * njacZ[0][2][k+1][i][j];
        lhsZ[0][3][CC][k][i][j] =  temp2 * fjacZ[0][3][k+1][i][j]
          - temp1 * njacZ[0][3][k+1][i][j];
        lhsZ[0][4][CC][k][i][j] =  temp2 * fjacZ[0][4][k+1][i][j]
          - temp1 * njacZ[0][4][k+1][i][j];

        lhsZ[1][0][CC][k][i][j] =  temp2 * fjacZ[1][0][k+1][i][j]
          - temp1 * njacZ[1][0][k+1][i][j];
        lhsZ[1][1][CC][k][i][j] =  temp2 * fjacZ[1][1][k+1][i][j]
          - temp1 * njacZ[1][1][k+1][i][j]
          - temp1 * dz2;
        lhsZ[1][2][CC][k][i][j] =  temp2 * fjacZ[1][2][k+1][i][j]
          - temp1 * njacZ[1][2][k+1][i][j];
        lhsZ[1][3][CC][k][i][j] =  temp2 * fjacZ[1][3][k+1][i][j]
          - temp1 * njacZ[1][3][k+1][i][j];
        lhsZ[1][4][CC][k][i][j] =  temp2 * fjacZ[1][4][k+1][i][j]
          - temp1 * njacZ[1][4][k+1][i][j];

        lhsZ[2][0][CC][k][i][j] =  temp2 * fjacZ[2][0][k+1][i][j]
          - temp1 * njacZ[2][0][k+1][i][j];
        lhsZ[2][1][CC][k][i][j] =  temp2 * fjacZ[2][1][k+1][i][j]
          - temp1 * njacZ[2][1][k+1][i][j];
        lhsZ[2][2][CC][k][i][j] =  temp2 * fjacZ[2][2][k+1][i][j]
          - temp1 * njacZ[2][2][k+1][i][j]
          - temp1 * dz3;
        lhsZ[2][3][CC][k][i][j] =  temp2 * fjacZ[2][3][k+1][i][j]
          - temp1 * njacZ[2][3][k+1][i][j];
        lhsZ[2][4][CC][k][i][j] =  temp2 * fjacZ[2][4][k+1][i][j]
          - temp1 * njacZ[2][4][k+1][i][j];

        lhsZ[3][0][CC][k][i][j] =  temp2 * fjacZ[3][0][k+1][i][j]
          - temp1 * njacZ[3][0][k+1][i][j];
        lhsZ[3][1][CC][k][i][j] =  temp2 * fjacZ[3][1][k+1][i][j]
          - temp1 * njacZ[3][1][k+1][i][j];
        lhsZ[3][2][CC][k][i][j] =  temp2 * fjacZ[3][2][k+1][i][j]
          - temp1 * njacZ[3][2][k+1][i][j];
        lhsZ[3][3][CC][k][i][j] =  temp2 * fjacZ[3][3][k+1][i][j]
          - temp1 * njacZ[3][3][k+1][i][j]
          - temp1 * dz4;
        lhsZ[3][4][CC][k][i][j] =  temp2 * fjacZ[3][4][k+1][i][j]
          - temp1 * njacZ[3][4][k+1][i][j];

        lhsZ[4][0][CC][k][i][j] =  temp2 * fjacZ[4][0][k+1][i][j]
          - temp1 * njacZ[4][0][k+1][i][j];
        lhsZ[4][1][CC][k][i][j] =  temp2 * fjacZ[4][1][k+1][i][j]
          - temp1 * njacZ[4][1][k+1][i][j];
        lhsZ[4][2][CC][k][i][j] =  temp2 * fjacZ[4][2][k+1][i][j]
          - temp1 * njacZ[4][2][k+1][i][j];
        lhsZ[4][3][CC][k][i][j] =  temp2 * fjacZ[4][3][k+1][i][j]
          - temp1 * njacZ[4][3][k+1][i][j];
        lhsZ[4][4][CC][k][i][j] =  temp2 * fjacZ[4][4][k+1][i][j]
          - temp1 * njacZ[4][4][k+1][i][j]
          - temp1 * dz5;
      }
	}
  }
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // performs guaussian elimination on this cell.
      // 
      // assumes that unpacking routines for non-first cells 
      // preload C' and rhs' from previous cell.
      // 
      // assumed send happens outside this routine, but that
      // c'(KMAX) and rhs'(KMAX) will be sent to next cell.
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // outer most do loops - sweeping in i direction
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // multiply c[0][j][i] by b_inverse and copy back to c
      // multiply rhs(0) by b_inverse(0) and copy to rhs
      //---------------------------------------------------------------------
      //binvcrhs( lhsZ[0][i][BB], lhsZ[j][0][i][j][CC], rhs[0][j][i] );
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
/*
	  for(m = 0; m < 5; m++){
	  	pivot = 1.00/lhsZ[m][m][BB][0][i][j];
		for(n = m+1; n < 5; n++){
			lhsZ[m][n][BB][0][i][j] = lhsZ[m][n][BB][0][i][j]*pivot;
		}
		lhsZ[m][0][CC][0][i][j] = lhsZ[m][0][CC][0][i][j]*pivot;
		lhsZ[m][1][CC][0][i][j] = lhsZ[m][1][CC][0][i][j]*pivot;
		lhsZ[m][2][CC][0][i][j] = lhsZ[m][2][CC][0][i][j]*pivot;
		lhsZ[m][3][CC][0][i][j] = lhsZ[m][3][CC][0][i][j]*pivot;
		lhsZ[m][4][CC][0][i][j] = lhsZ[m][4][CC][0][i][j]*pivot;
		rhs[0][j][i][m] = rhs[0][j][i][m]*pivot;

		for(n = 0; n < 5; n++){
			if(n != m){
				coeff = lhsZ[n][m][BB][0][i][j];
				for(z = m+1; z < 5; z++){
					lhsZ[n][z][BB][0][i][j] = lhsZ[n][z][BB][0][i][j] - coeff*lhsZ[m][z][BB][0][i][j];
				}
				lhsZ[n][0][CC][0][i][j] = lhsZ[n][0][CC][0][i][j] - coeff*lhsZ[m][0][CC][0][i][j];
				lhsZ[n][1][CC][0][i][j] = lhsZ[n][1][CC][0][i][j] - coeff*lhsZ[m][1][CC][0][i][j];
				lhsZ[n][2][CC][0][i][j] = lhsZ[n][2][CC][0][i][j] - coeff*lhsZ[m][2][CC][0][i][j];
				lhsZ[n][3][CC][0][i][j] = lhsZ[n][3][CC][0][i][j] - coeff*lhsZ[m][3][CC][0][i][j];
				lhsZ[n][4][CC][0][i][j] = lhsZ[n][4][CC][0][i][j] - coeff*lhsZ[m][4][CC][0][i][j];
				rhs[0][j][i][n] = rhs[0][j][i][n] - coeff*rhs[0][j][i][m];
			}
		}
	  }
*/
  pivot = 1.00/lhsZ[0][0][BB][0][i][j];
  lhsZ[0][1][BB][0][i][j] = lhsZ[0][1][BB][0][i][j]*pivot;
  lhsZ[0][2][BB][0][i][j] = lhsZ[0][2][BB][0][i][j]*pivot;
  lhsZ[0][3][BB][0][i][j] = lhsZ[0][3][BB][0][i][j]*pivot;
  lhsZ[0][4][BB][0][i][j] = lhsZ[0][4][BB][0][i][j]*pivot;
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j]*pivot;
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j]*pivot;
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j]*pivot;
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j]*pivot;
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j]*pivot;
  rhs[0][j][i][0]   = rhs[0][j][i][0]  *pivot;

  coeff = lhsZ[1][0][BB][0][i][j];
  lhsZ[1][1][BB][0][i][j]= lhsZ[1][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[1][2][BB][0][i][j]= lhsZ[1][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][0];

  coeff = lhsZ[2][0][BB][0][i][j];
  lhsZ[2][1][BB][0][i][j]= lhsZ[2][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][0];

  coeff = lhsZ[3][0][BB][0][i][j];
  lhsZ[3][1][BB][0][i][j]= lhsZ[3][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][0];

  coeff = lhsZ[4][0][BB][0][i][j];
  lhsZ[4][1][BB][0][i][j]= lhsZ[4][1][BB][0][i][j] - coeff*lhsZ[0][1][BB][0][i][j];
  lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[0][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[0][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[0][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[0][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[0][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[0][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[0][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[0][4][CC][0][i][j];
  rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][0];


  pivot = 1.00/lhsZ[1][1][BB][0][i][j];
  lhsZ[1][2][BB][0][i][j] = lhsZ[1][2][BB][0][i][j]*pivot;
  lhsZ[1][3][BB][0][i][j] = lhsZ[1][3][BB][0][i][j]*pivot;
  lhsZ[1][4][BB][0][i][j] = lhsZ[1][4][BB][0][i][j]*pivot;
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j]*pivot;
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j]*pivot;
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j]*pivot;
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j]*pivot;
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j]*pivot;
  rhs[0][j][i][1]   = rhs[0][j][i][1]  *pivot;

  coeff = lhsZ[0][1][BB][0][i][j];
  lhsZ[0][2][BB][0][i][j]= lhsZ[0][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][1];

  coeff = lhsZ[2][1][BB][0][i][j];
  lhsZ[2][2][BB][0][i][j]= lhsZ[2][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j]= lhsZ[2][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][1];

  coeff = lhsZ[3][1][BB][0][i][j];
  lhsZ[3][2][BB][0][i][j]= lhsZ[3][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][1];

  coeff = lhsZ[4][1][BB][0][i][j];
  lhsZ[4][2][BB][0][i][j]= lhsZ[4][2][BB][0][i][j] - coeff*lhsZ[1][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[1][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[1][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[1][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[1][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[1][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[1][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[1][4][CC][0][i][j];
  rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][1];


  pivot = 1.00/lhsZ[2][2][BB][0][i][j];
  lhsZ[2][3][BB][0][i][j] = lhsZ[2][3][BB][0][i][j]*pivot;
  lhsZ[2][4][BB][0][i][j] = lhsZ[2][4][BB][0][i][j]*pivot;
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j]*pivot;
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j]*pivot;
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j]*pivot;
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j]*pivot;
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j]*pivot;
  rhs[0][j][i][2]   = rhs[0][j][i][2]  *pivot;

  coeff = lhsZ[0][2][BB][0][i][j];
  lhsZ[0][3][BB][0][i][j]= lhsZ[0][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][2];

  coeff = lhsZ[1][2][BB][0][i][j];
  lhsZ[1][3][BB][0][i][j]= lhsZ[1][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][2];

  coeff = lhsZ[3][2][BB][0][i][j];
  lhsZ[3][3][BB][0][i][j]= lhsZ[3][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j]= lhsZ[3][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][2];

  coeff = lhsZ[4][2][BB][0][i][j];
  lhsZ[4][3][BB][0][i][j]= lhsZ[4][3][BB][0][i][j] - coeff*lhsZ[2][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[2][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[2][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[2][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[2][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[2][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[2][4][CC][0][i][j];
  rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][2];


  pivot = 1.00/lhsZ[3][3][BB][0][i][j];
  lhsZ[3][4][BB][0][i][j] = lhsZ[3][4][BB][0][i][j]*pivot;
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j]*pivot;
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j]*pivot;
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j]*pivot;
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j]*pivot;
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j]*pivot;
  rhs[0][j][i][3]   = rhs[0][j][i][3]  *pivot;

  coeff = lhsZ[0][3][BB][0][i][j];
  lhsZ[0][4][BB][0][i][j]= lhsZ[0][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][3];

  coeff = lhsZ[1][3][BB][0][i][j];
  lhsZ[1][4][BB][0][i][j]= lhsZ[1][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][3];

  coeff = lhsZ[2][3][BB][0][i][j];
  lhsZ[2][4][BB][0][i][j]= lhsZ[2][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][3];

  coeff = lhsZ[4][3][BB][0][i][j];
  lhsZ[4][4][BB][0][i][j]= lhsZ[4][4][BB][0][i][j] - coeff*lhsZ[3][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j] - coeff*lhsZ[3][0][CC][0][i][j];
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j] - coeff*lhsZ[3][1][CC][0][i][j];
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j] - coeff*lhsZ[3][2][CC][0][i][j];
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j] - coeff*lhsZ[3][3][CC][0][i][j];
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j] - coeff*lhsZ[3][4][CC][0][i][j];
  rhs[0][j][i][4]   = rhs[0][j][i][4]   - coeff*rhs[0][j][i][3];


  pivot = 1.00/lhsZ[4][4][BB][0][i][j];
  lhsZ[4][0][CC][0][i][j] = lhsZ[4][0][CC][0][i][j]*pivot;
  lhsZ[4][1][CC][0][i][j] = lhsZ[4][1][CC][0][i][j]*pivot;
  lhsZ[4][2][CC][0][i][j] = lhsZ[4][2][CC][0][i][j]*pivot;
  lhsZ[4][3][CC][0][i][j] = lhsZ[4][3][CC][0][i][j]*pivot;
  lhsZ[4][4][CC][0][i][j] = lhsZ[4][4][CC][0][i][j]*pivot;
  rhs[0][j][i][4]   = rhs[0][j][i][4]  *pivot;

  coeff = lhsZ[0][4][BB][0][i][j];
  lhsZ[0][0][CC][0][i][j] = lhsZ[0][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[0][1][CC][0][i][j] = lhsZ[0][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[0][2][CC][0][i][j] = lhsZ[0][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[0][3][CC][0][i][j] = lhsZ[0][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[0][4][CC][0][i][j] = lhsZ[0][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[0][j][i][0]   = rhs[0][j][i][0]   - coeff*rhs[0][j][i][4];

  coeff = lhsZ[1][4][BB][0][i][j];
  lhsZ[1][0][CC][0][i][j] = lhsZ[1][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[1][1][CC][0][i][j] = lhsZ[1][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[1][2][CC][0][i][j] = lhsZ[1][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[1][3][CC][0][i][j] = lhsZ[1][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[1][4][CC][0][i][j] = lhsZ[1][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[0][j][i][1]   = rhs[0][j][i][1]   - coeff*rhs[0][j][i][4];

  coeff = lhsZ[2][4][BB][0][i][j];
  lhsZ[2][0][CC][0][i][j] = lhsZ[2][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[2][1][CC][0][i][j] = lhsZ[2][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[2][2][CC][0][i][j] = lhsZ[2][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[2][3][CC][0][i][j] = lhsZ[2][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[2][4][CC][0][i][j] = lhsZ[2][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[0][j][i][2]   = rhs[0][j][i][2]   - coeff*rhs[0][j][i][4];

  coeff = lhsZ[3][4][BB][0][i][j];
  lhsZ[3][0][CC][0][i][j] = lhsZ[3][0][CC][0][i][j] - coeff*lhsZ[4][0][CC][0][i][j];
  lhsZ[3][1][CC][0][i][j] = lhsZ[3][1][CC][0][i][j] - coeff*lhsZ[4][1][CC][0][i][j];
  lhsZ[3][2][CC][0][i][j] = lhsZ[3][2][CC][0][i][j] - coeff*lhsZ[4][2][CC][0][i][j];
  lhsZ[3][3][CC][0][i][j] = lhsZ[3][3][CC][0][i][j] - coeff*lhsZ[4][3][CC][0][i][j];
  lhsZ[3][4][CC][0][i][j] = lhsZ[3][4][CC][0][i][j] - coeff*lhsZ[4][4][CC][0][i][j];
  rhs[0][j][i][3]   = rhs[0][j][i][3]   - coeff*rhs[0][j][i][4];

	}
  }
      //---------------------------------------------------------------------
      // begin inner most do loop
      // do all the elements of the cell unless last 
      //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
      for (k = 1; k <= ksize-1; k++) {
        //-------------------------------------------------------------------
        // subtract A*lhsZ[j][i]_vector(k-1) from lhsZ[j][i]_vector(k)
        // 
        // rhs(k) = rhs(k) - A*rhs(k-1)
        //-------------------------------------------------------------------
        //matvec_sub(lhsZ[i][j][AA], rhs[k-1][k][i][j], rhs[k][j][i]);
		/*
		for(m = 0; m < 5; m++){
			rhs[k][j][i][m] = rhs[k][j][i][m] - lhsZ[m][0][AA][k][i][j]*rhs[k-1][j][i][0]
											  - lhsZ[m][1][AA][k][i][j]*rhs[k-1][j][i][1]
											  - lhsZ[m][2][AA][k][i][j]*rhs[k-1][j][i][2]
											  - lhsZ[m][3][AA][k][i][j]*rhs[k-1][j][i][3]
											  - lhsZ[m][4][AA][k][i][j]*rhs[k-1][j][i][4];
		}
		*/
  rhs[k][j][i][0] = rhs[k][j][i][0] - lhsZ[0][0][AA][k][i][j]*rhs[k-1][j][i][0]
                    - lhsZ[0][1][AA][k][i][j]*rhs[k-1][j][i][1]
                    - lhsZ[0][2][AA][k][i][j]*rhs[k-1][j][i][2]
                    - lhsZ[0][3][AA][k][i][j]*rhs[k-1][j][i][3]
                    - lhsZ[0][4][AA][k][i][j]*rhs[k-1][j][i][4];
  rhs[k][j][i][1] = rhs[k][j][i][1] - lhsZ[1][0][AA][k][i][j]*rhs[k-1][j][i][0]
                    - lhsZ[1][1][AA][k][i][j]*rhs[k-1][j][i][1]
                    - lhsZ[1][2][AA][k][i][j]*rhs[k-1][j][i][2]
                    - lhsZ[1][3][AA][k][i][j]*rhs[k-1][j][i][3]
                    - lhsZ[1][4][AA][k][i][j]*rhs[k-1][j][i][4];
  rhs[k][j][i][2] = rhs[k][j][i][2] - lhsZ[2][0][AA][k][i][j]*rhs[k-1][j][i][0]
                    - lhsZ[2][1][AA][k][i][j]*rhs[k-1][j][i][1]
                    - lhsZ[2][2][AA][k][i][j]*rhs[k-1][j][i][2]
                    - lhsZ[2][3][AA][k][i][j]*rhs[k-1][j][i][3]
                    - lhsZ[2][4][AA][k][i][j]*rhs[k-1][j][i][4];
  rhs[k][j][i][3] = rhs[k][j][i][3] - lhsZ[3][0][AA][k][i][j]*rhs[k-1][j][i][0]
                    - lhsZ[3][1][AA][k][i][j]*rhs[k-1][j][i][1]
                    - lhsZ[3][2][AA][k][i][j]*rhs[k-1][j][i][2]
                    - lhsZ[3][3][AA][k][i][j]*rhs[k-1][j][i][3]
                    - lhsZ[3][4][AA][k][i][j]*rhs[k-1][j][i][4];
  rhs[k][j][i][4] = rhs[k][j][i][4] - lhsZ[4][0][AA][k][i][j]*rhs[k-1][j][i][0]
                    - lhsZ[4][1][AA][k][i][j]*rhs[k-1][j][i][1]
                    - lhsZ[4][2][AA][k][i][j]*rhs[k-1][j][i][2]
                    - lhsZ[4][3][AA][k][i][j]*rhs[k-1][j][i][3]
                    - lhsZ[4][4][AA][k][i][j]*rhs[k-1][j][i][4];



        //-------------------------------------------------------------------
        // B(k) = B(k) - C(k-1)*A(k)
        // matmul_sub(AA,i,j,k,c,CC,i,j,k-1,c,BB,i,j,k)
        //-------------------------------------------------------------------
        //matmul_sub(lhsZ[k-1][i][AA], lhsZ[j][k][i][j][CC], lhsZ[j][i][k][BB]);
		/*
	  for(m = 0; m < 5; m++){
	  	for(n = 0; n < 5; n++){
			lhsZ[n][m][BB][k][i][j] = lhsZ[n][m][BB][k][i][j] - lhsZ[n][0][AA][k][i][j]*lhsZ[0][m][CC][k-1][i][j]
												- lhsZ[n][1][AA][k][i][j]*lhsZ[1][m][CC][k-1][i][j]
												- lhsZ[n][2][AA][k][i][j]*lhsZ[2][m][CC][k-1][i][j]
												- lhsZ[n][3][AA][k][i][j]*lhsZ[3][m][CC][k-1][i][j]
												- lhsZ[n][4][AA][k][i][j]*lhsZ[4][m][CC][k-1][i][j];
		}
	  }
	  */
  lhsZ[0][0][BB][k][i][j] = lhsZ[0][0][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[1][0][BB][k][i][j] = lhsZ[1][0][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[2][0][BB][k][i][j] = lhsZ[2][0][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[3][0][BB][k][i][j] = lhsZ[3][0][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[4][0][BB][k][i][j] = lhsZ[4][0][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][0][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][0][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][0][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][0][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][0][CC][k-1][i][j];
  lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[1][1][BB][k][i][j] = lhsZ[1][1][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[2][1][BB][k][i][j] = lhsZ[2][1][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[3][1][BB][k][i][j] = lhsZ[3][1][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[4][1][BB][k][i][j] = lhsZ[4][1][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][1][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][1][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][1][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][1][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][1][CC][k-1][i][j];
  lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[2][2][BB][k][i][j] = lhsZ[2][2][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[3][2][BB][k][i][j] = lhsZ[3][2][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[4][2][BB][k][i][j] = lhsZ[4][2][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][2][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][2][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][2][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][2][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][2][CC][k-1][i][j];
  lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[3][3][BB][k][i][j] = lhsZ[3][3][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[4][3][BB][k][i][j] = lhsZ[4][3][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][3][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][3][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][3][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][3][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][3][CC][k-1][i][j];
  lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j] - lhsZ[0][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[0][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[0][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[0][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[0][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j] - lhsZ[1][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[1][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[1][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[1][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[1][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j] - lhsZ[2][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[2][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[2][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[2][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[2][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j] - lhsZ[3][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[3][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[3][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[3][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[3][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];
  lhsZ[4][4][BB][k][i][j] = lhsZ[4][4][BB][k][i][j] - lhsZ[4][0][AA][k][i][j]*lhsZ[0][4][CC][k-1][i][j]
                              - lhsZ[4][1][AA][k][i][j]*lhsZ[1][4][CC][k-1][i][j]
                              - lhsZ[4][2][AA][k][i][j]*lhsZ[2][4][CC][k-1][i][j]
                              - lhsZ[4][3][AA][k][i][j]*lhsZ[3][4][CC][k-1][i][j]
                              - lhsZ[4][4][AA][k][i][j]*lhsZ[4][4][CC][k-1][i][j];

        //-------------------------------------------------------------------
        // multiply c[k][j][i] by b_inverse and copy back to c
        // multiply rhs[0][j][i] by b_inverse[0][j][i] and copy to rhs
        //-------------------------------------------------------------------
        //binvcrhs( lhsZ[k][i][BB], lhsZ[j][k][i][j][CC], rhs[k][j][i] );
		/*
	  	for(m = 0; m < 5; m++){
	  		pivot = 1.00/lhsZ[m][m][BB][k][i][j];
			for(n = m+1; n < 5; n++){
				lhsZ[m][n][BB][k][i][j] = lhsZ[m][n][BB][k][i][j]*pivot;
			}
			lhsZ[m][0][CC][k][i][j] = lhsZ[m][0][CC][k][i][j]*pivot;
			lhsZ[m][1][CC][k][i][j] = lhsZ[m][1][CC][k][i][j]*pivot;
			lhsZ[m][2][CC][k][i][j] = lhsZ[m][2][CC][k][i][j]*pivot;
			lhsZ[m][3][CC][k][i][j] = lhsZ[m][3][CC][k][i][j]*pivot;
			lhsZ[m][4][CC][k][i][j] = lhsZ[m][4][CC][k][i][j]*pivot;
			rhs[k][j][i][m] = rhs[k][j][i][m]*pivot;

			for(n = 0; n < 5; n++){
				if(n != m){
					coeff = lhsZ[n][m][BB][k][i][j];
					for(z = m+1; z < 5; z++){
						lhsZ[n][z][BB][k][i][j] = lhsZ[n][z][BB][k][i][j] - coeff*lhsZ[m][z][BB][k][i][j];
					}
					lhsZ[n][0][CC][k][i][j] = lhsZ[n][0][CC][k][i][j] - coeff*lhsZ[m][0][CC][k][i][j];
					lhsZ[n][1][CC][k][i][j] = lhsZ[n][1][CC][k][i][j] - coeff*lhsZ[m][1][CC][k][i][j];
					lhsZ[n][2][CC][k][i][j] = lhsZ[n][2][CC][k][i][j] - coeff*lhsZ[m][2][CC][k][i][j];
					lhsZ[n][3][CC][k][i][j] = lhsZ[n][3][CC][k][i][j] - coeff*lhsZ[m][3][CC][k][i][j];
					lhsZ[n][4][CC][k][i][j] = lhsZ[n][4][CC][k][i][j] - coeff*lhsZ[m][4][CC][k][i][j];
					rhs[k][j][i][n] = rhs[k][j][i][n] - coeff*rhs[k][j][i][m];
				}
			}
	  	}
		*/
  pivot = 1.00/lhsZ[0][0][BB][k][i][j];
  lhsZ[0][1][BB][k][i][j] = lhsZ[0][1][BB][k][i][j]*pivot;
  lhsZ[0][2][BB][k][i][j] = lhsZ[0][2][BB][k][i][j]*pivot;
  lhsZ[0][3][BB][k][i][j] = lhsZ[0][3][BB][k][i][j]*pivot;
  lhsZ[0][4][BB][k][i][j] = lhsZ[0][4][BB][k][i][j]*pivot;
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j]*pivot;
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j]*pivot;
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j]*pivot;
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j]*pivot;
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j]*pivot;
  rhs[k][j][i][0]   = rhs[k][j][i][0]  *pivot;

  coeff = lhsZ[1][0][BB][k][i][j];
  lhsZ[1][1][BB][k][i][j]= lhsZ[1][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[1][2][BB][k][i][j]= lhsZ[1][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][0];

  coeff = lhsZ[2][0][BB][k][i][j];
  lhsZ[2][1][BB][k][i][j]= lhsZ[2][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][0];

  coeff = lhsZ[3][0][BB][k][i][j];
  lhsZ[3][1][BB][k][i][j]= lhsZ[3][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][0];

  coeff = lhsZ[4][0][BB][k][i][j];
  lhsZ[4][1][BB][k][i][j]= lhsZ[4][1][BB][k][i][j] - coeff*lhsZ[0][1][BB][k][i][j];
  lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[0][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[0][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[0][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[0][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[0][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[0][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[0][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[0][4][CC][k][i][j];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][0];


  pivot = 1.00/lhsZ[1][1][BB][k][i][j];
  lhsZ[1][2][BB][k][i][j] = lhsZ[1][2][BB][k][i][j]*pivot;
  lhsZ[1][3][BB][k][i][j] = lhsZ[1][3][BB][k][i][j]*pivot;
  lhsZ[1][4][BB][k][i][j] = lhsZ[1][4][BB][k][i][j]*pivot;
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j]*pivot;
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j]*pivot;
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j]*pivot;
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j]*pivot;
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j]*pivot;
  rhs[k][j][i][1]   = rhs[k][j][i][1]  *pivot;

  coeff = lhsZ[0][1][BB][k][i][j];
  lhsZ[0][2][BB][k][i][j]= lhsZ[0][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][1];

  coeff = lhsZ[2][1][BB][k][i][j];
  lhsZ[2][2][BB][k][i][j]= lhsZ[2][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j]= lhsZ[2][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][1];

  coeff = lhsZ[3][1][BB][k][i][j];
  lhsZ[3][2][BB][k][i][j]= lhsZ[3][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][1];

  coeff = lhsZ[4][1][BB][k][i][j];
  lhsZ[4][2][BB][k][i][j]= lhsZ[4][2][BB][k][i][j] - coeff*lhsZ[1][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[1][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[1][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[1][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[1][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[1][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[1][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[1][4][CC][k][i][j];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][1];


  pivot = 1.00/lhsZ[2][2][BB][k][i][j];
  lhsZ[2][3][BB][k][i][j] = lhsZ[2][3][BB][k][i][j]*pivot;
  lhsZ[2][4][BB][k][i][j] = lhsZ[2][4][BB][k][i][j]*pivot;
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j]*pivot;
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j]*pivot;
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j]*pivot;
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j]*pivot;
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j]*pivot;
  rhs[k][j][i][2]   = rhs[k][j][i][2]  *pivot;

  coeff = lhsZ[0][2][BB][k][i][j];
  lhsZ[0][3][BB][k][i][j]= lhsZ[0][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][2];

  coeff = lhsZ[1][2][BB][k][i][j];
  lhsZ[1][3][BB][k][i][j]= lhsZ[1][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][2];

  coeff = lhsZ[3][2][BB][k][i][j];
  lhsZ[3][3][BB][k][i][j]= lhsZ[3][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j]= lhsZ[3][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][2];

  coeff = lhsZ[4][2][BB][k][i][j];
  lhsZ[4][3][BB][k][i][j]= lhsZ[4][3][BB][k][i][j] - coeff*lhsZ[2][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[2][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[2][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[2][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[2][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[2][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[2][4][CC][k][i][j];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][2];


  pivot = 1.00/lhsZ[3][3][BB][k][i][j];
  lhsZ[3][4][BB][k][i][j] = lhsZ[3][4][BB][k][i][j]*pivot;
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j]*pivot;
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j]*pivot;
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j]*pivot;
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j]*pivot;
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j]*pivot;
  rhs[k][j][i][3]   = rhs[k][j][i][3]  *pivot;

  coeff = lhsZ[0][3][BB][k][i][j];
  lhsZ[0][4][BB][k][i][j]= lhsZ[0][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][3];

  coeff = lhsZ[1][3][BB][k][i][j];
  lhsZ[1][4][BB][k][i][j]= lhsZ[1][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][3];

  coeff = lhsZ[2][3][BB][k][i][j];
  lhsZ[2][4][BB][k][i][j]= lhsZ[2][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][3];

  coeff = lhsZ[4][3][BB][k][i][j];
  lhsZ[4][4][BB][k][i][j]= lhsZ[4][4][BB][k][i][j] - coeff*lhsZ[3][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j] - coeff*lhsZ[3][0][CC][k][i][j];
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j] - coeff*lhsZ[3][1][CC][k][i][j];
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j] - coeff*lhsZ[3][2][CC][k][i][j];
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j] - coeff*lhsZ[3][3][CC][k][i][j];
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j] - coeff*lhsZ[3][4][CC][k][i][j];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][3];


  pivot = 1.00/lhsZ[4][4][BB][k][i][j];
  lhsZ[4][0][CC][k][i][j] = lhsZ[4][0][CC][k][i][j]*pivot;
  lhsZ[4][1][CC][k][i][j] = lhsZ[4][1][CC][k][i][j]*pivot;
  lhsZ[4][2][CC][k][i][j] = lhsZ[4][2][CC][k][i][j]*pivot;
  lhsZ[4][3][CC][k][i][j] = lhsZ[4][3][CC][k][i][j]*pivot;
  lhsZ[4][4][CC][k][i][j] = lhsZ[4][4][CC][k][i][j]*pivot;
  rhs[k][j][i][4]   = rhs[k][j][i][4]  *pivot;

  coeff = lhsZ[0][4][BB][k][i][j];
  lhsZ[0][0][CC][k][i][j] = lhsZ[0][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[0][1][CC][k][i][j] = lhsZ[0][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[0][2][CC][k][i][j] = lhsZ[0][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[0][3][CC][k][i][j] = lhsZ[0][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[0][4][CC][k][i][j] = lhsZ[0][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][4];

  coeff = lhsZ[1][4][BB][k][i][j];
  lhsZ[1][0][CC][k][i][j] = lhsZ[1][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[1][1][CC][k][i][j] = lhsZ[1][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[1][2][CC][k][i][j] = lhsZ[1][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[1][3][CC][k][i][j] = lhsZ[1][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[1][4][CC][k][i][j] = lhsZ[1][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][4];

  coeff = lhsZ[2][4][BB][k][i][j];
  lhsZ[2][0][CC][k][i][j] = lhsZ[2][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[2][1][CC][k][i][j] = lhsZ[2][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[2][2][CC][k][i][j] = lhsZ[2][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[2][3][CC][k][i][j] = lhsZ[2][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[2][4][CC][k][i][j] = lhsZ[2][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][4];

  coeff = lhsZ[3][4][BB][k][i][j];
  lhsZ[3][0][CC][k][i][j] = lhsZ[3][0][CC][k][i][j] - coeff*lhsZ[4][0][CC][k][i][j];
  lhsZ[3][1][CC][k][i][j] = lhsZ[3][1][CC][k][i][j] - coeff*lhsZ[4][1][CC][k][i][j];
  lhsZ[3][2][CC][k][i][j] = lhsZ[3][2][CC][k][i][j] - coeff*lhsZ[4][2][CC][k][i][j];
  lhsZ[3][3][CC][k][i][j] = lhsZ[3][3][CC][k][i][j] - coeff*lhsZ[4][3][CC][k][i][j];
  lhsZ[3][4][CC][k][i][j] = lhsZ[3][4][CC][k][i][j] - coeff*lhsZ[4][4][CC][k][i][j];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][4];



      }/*end loop k*/
	}/*end loop i*/
  }/*end loop j*/
      //---------------------------------------------------------------------
      // Now finish up special cases for last cell
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // rhs(ksize) = rhs(ksize) - A*rhs(ksize-1)
      //---------------------------------------------------------------------
      //matvec_sub(lhsZ[i][j][AA], rhs[ksize-1][ksize][i][j], rhs[ksize][j][i]);
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
	/*
		for(m = 0; m < 5; m++){
			rhs[ksize][j][i][m] = rhs[ksize][j][i][m] - lhsZ[m][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
											          - lhsZ[m][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
											          - lhsZ[m][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
											          - lhsZ[m][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
											          - lhsZ[m][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
		}
	*/
  rhs[ksize][j][i][0] = rhs[ksize][j][i][0] - lhsZ[0][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
                    - lhsZ[0][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
                    - lhsZ[0][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
                    - lhsZ[0][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
                    - lhsZ[0][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
  rhs[ksize][j][i][1] = rhs[ksize][j][i][1] - lhsZ[1][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
                    - lhsZ[1][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
                    - lhsZ[1][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
                    - lhsZ[1][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
                    - lhsZ[1][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
  rhs[ksize][j][i][2] = rhs[ksize][j][i][2] - lhsZ[2][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
                    - lhsZ[2][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
                    - lhsZ[2][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
                    - lhsZ[2][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
                    - lhsZ[2][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
  rhs[ksize][j][i][3] = rhs[ksize][j][i][3] - lhsZ[3][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
                    - lhsZ[3][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
                    - lhsZ[3][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
                    - lhsZ[3][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
                    - lhsZ[3][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
  rhs[ksize][j][i][4] = rhs[ksize][j][i][4] - lhsZ[4][0][AA][ksize][i][j]*rhs[ksize-1][j][i][0]
                    - lhsZ[4][1][AA][ksize][i][j]*rhs[ksize-1][j][i][1]
                    - lhsZ[4][2][AA][ksize][i][j]*rhs[ksize-1][j][i][2]
                    - lhsZ[4][3][AA][ksize][i][j]*rhs[ksize-1][j][i][3]
                    - lhsZ[4][4][AA][ksize][i][j]*rhs[ksize-1][j][i][4];
	}
  }
      //---------------------------------------------------------------------
      // B(ksize) = B(ksize) - C(ksize-1)*A(ksize)
      // matmul_sub(AA,i,j,ksize,c,
      // $              CC,i,j,ksize-1,c,BB,i,j,ksize)
      //---------------------------------------------------------------------
      //matmul_sub(lhsZ[ksize-1][i][AA], lhsZ[j][ksize][i][j][CC], lhsZ[j][i][ksize][BB]);
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
	/*
	  for(m = 0; m < 5; m++){
	  	for(n = 0; n < 5; n++){
			lhsZ[n][m][BB][ksize][i][j] = lhsZ[n][m][BB][ksize][i][j] - lhsZ[n][0][AA][ksize][i][j]*lhsZ[0][m][CC][ksize-1][i][j]
														- lhsZ[n][1][AA][ksize][i][j]*lhsZ[1][m][CC][ksize-1][i][j]
														- lhsZ[n][2][AA][ksize][i][j]*lhsZ[2][m][CC][ksize-1][i][j]
														- lhsZ[n][3][AA][ksize][i][j]*lhsZ[3][m][CC][ksize-1][i][j]
														- lhsZ[n][4][AA][ksize][i][j]*lhsZ[4][m][CC][ksize-1][i][j];
		}
	  }
	*/
  lhsZ[0][0][BB][ksize][i][j] = lhsZ[0][0][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[1][0][BB][ksize][i][j] = lhsZ[1][0][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[2][0][BB][ksize][i][j] = lhsZ[2][0][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[3][0][BB][ksize][i][j] = lhsZ[3][0][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[4][0][BB][ksize][i][j] = lhsZ[4][0][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][0][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][0][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][0][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][0][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][0][CC][ksize-1][i][j];
  lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[1][1][BB][ksize][i][j] = lhsZ[1][1][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[2][1][BB][ksize][i][j] = lhsZ[2][1][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[3][1][BB][ksize][i][j] = lhsZ[3][1][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[4][1][BB][ksize][i][j] = lhsZ[4][1][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][1][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][1][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][1][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][1][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][1][CC][ksize-1][i][j];
  lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[2][2][BB][ksize][i][j] = lhsZ[2][2][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[3][2][BB][ksize][i][j] = lhsZ[3][2][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[4][2][BB][ksize][i][j] = lhsZ[4][2][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][2][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][2][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][2][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][2][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][2][CC][ksize-1][i][j];
  lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[3][3][BB][ksize][i][j] = lhsZ[3][3][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[4][3][BB][ksize][i][j] = lhsZ[4][3][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][3][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][3][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][3][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][3][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][3][CC][ksize-1][i][j];
  lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j] - lhsZ[0][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[0][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[0][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[0][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[0][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j] - lhsZ[1][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[1][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[1][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[1][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[1][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j] - lhsZ[2][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[2][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[2][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[2][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[2][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j] - lhsZ[3][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[3][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[3][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[3][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[3][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];
  lhsZ[4][4][BB][ksize][i][j] = lhsZ[4][4][BB][ksize][i][j] - lhsZ[4][0][AA][ksize][i][j]*lhsZ[0][4][CC][ksize-1][i][j]
                              - lhsZ[4][1][AA][ksize][i][j]*lhsZ[1][4][CC][ksize-1][i][j]
                              - lhsZ[4][2][AA][ksize][i][j]*lhsZ[2][4][CC][ksize-1][i][j]
                              - lhsZ[4][3][AA][ksize][i][j]*lhsZ[3][4][CC][ksize-1][i][j]
                              - lhsZ[4][4][AA][ksize][i][j]*lhsZ[4][4][CC][ksize-1][i][j];

	}
  }
      //---------------------------------------------------------------------
      // multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
      //---------------------------------------------------------------------
      //binvrhs( lhsZ[i][j][BB], rhs[ksize][ksize][i][j] );
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
	/*
	  for(m = 0; m < 5; m++){
	  	pivot = 1.00/lhsZ[m][m][BB][ksize][i][j];
		for(n = m+1; n < 5; n++){
			lhsZ[m][n][BB][ksize][i][j] = lhsZ[m][n][BB][ksize][i][j]*pivot;
		}
		rhs[ksize][j][i][m] = rhs[ksize][j][i][m]*pivot;
		
		for(n = 0; n < 5; n++){
			if(n != m){
				coeff = lhsZ[n][m][BB][ksize][i][j];
				for(z = m+1; z < 5; z++){
					lhsZ[n][z][BB][ksize][i][j] = lhsZ[n][z][BB][ksize][i][j] - coeff*lhsZ[m][z][BB][ksize][i][j];
				}
				rhs[ksize][j][i][n] = rhs[ksize][j][i][n] - coeff*rhs[ksize][j][i][m];
			}
		}
	  }
	*/

  pivot = 1.00/lhsZ[0][0][BB][ksize][i][j];
  lhsZ[0][1][BB][ksize][i][j] = lhsZ[0][1][BB][ksize][i][j]*pivot;
  lhsZ[0][2][BB][ksize][i][j] = lhsZ[0][2][BB][ksize][i][j]*pivot;
  lhsZ[0][3][BB][ksize][i][j] = lhsZ[0][3][BB][ksize][i][j]*pivot;
  lhsZ[0][4][BB][ksize][i][j] = lhsZ[0][4][BB][ksize][i][j]*pivot;
  rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]  *pivot;

  coeff = lhsZ[1][0][BB][ksize][i][j];
  lhsZ[1][1][BB][ksize][i][j]= lhsZ[1][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[1][2][BB][ksize][i][j]= lhsZ[1][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][0];

  coeff = lhsZ[2][0][BB][ksize][i][j];
  lhsZ[2][1][BB][ksize][i][j]= lhsZ[2][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][0];

  coeff = lhsZ[3][0][BB][ksize][i][j];
  lhsZ[3][1][BB][ksize][i][j]= lhsZ[3][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][0];

  coeff = lhsZ[4][0][BB][ksize][i][j];
  lhsZ[4][1][BB][ksize][i][j]= lhsZ[4][1][BB][ksize][i][j] - coeff*lhsZ[0][1][BB][ksize][i][j];
  lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[0][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[0][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[0][4][BB][ksize][i][j];
  rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][0];


  pivot = 1.00/lhsZ[1][1][BB][ksize][i][j];
  lhsZ[1][2][BB][ksize][i][j] = lhsZ[1][2][BB][ksize][i][j]*pivot;
  lhsZ[1][3][BB][ksize][i][j] = lhsZ[1][3][BB][ksize][i][j]*pivot;
  lhsZ[1][4][BB][ksize][i][j] = lhsZ[1][4][BB][ksize][i][j]*pivot;
  rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]  *pivot;

  coeff = lhsZ[0][1][BB][ksize][i][j];
  lhsZ[0][2][BB][ksize][i][j]= lhsZ[0][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][1];

  coeff = lhsZ[2][1][BB][ksize][i][j];
  lhsZ[2][2][BB][ksize][i][j]= lhsZ[2][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j]= lhsZ[2][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][1];

  coeff = lhsZ[3][1][BB][ksize][i][j];
  lhsZ[3][2][BB][ksize][i][j]= lhsZ[3][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][1];

  coeff = lhsZ[4][1][BB][ksize][i][j];
  lhsZ[4][2][BB][ksize][i][j]= lhsZ[4][2][BB][ksize][i][j] - coeff*lhsZ[1][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[1][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[1][4][BB][ksize][i][j];
  rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][1];


  pivot = 1.00/lhsZ[2][2][BB][ksize][i][j];
  lhsZ[2][3][BB][ksize][i][j] = lhsZ[2][3][BB][ksize][i][j]*pivot;
  lhsZ[2][4][BB][ksize][i][j] = lhsZ[2][4][BB][ksize][i][j]*pivot;
  rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]  *pivot;

  coeff = lhsZ[0][2][BB][ksize][i][j];
  lhsZ[0][3][BB][ksize][i][j]= lhsZ[0][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][2];

  coeff = lhsZ[1][2][BB][ksize][i][j];
  lhsZ[1][3][BB][ksize][i][j]= lhsZ[1][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][2];

  coeff = lhsZ[3][2][BB][ksize][i][j];
  lhsZ[3][3][BB][ksize][i][j]= lhsZ[3][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j]= lhsZ[3][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][2];

  coeff = lhsZ[4][2][BB][ksize][i][j];
  lhsZ[4][3][BB][ksize][i][j]= lhsZ[4][3][BB][ksize][i][j] - coeff*lhsZ[2][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[2][4][BB][ksize][i][j];
  rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][2];


  pivot = 1.00/lhsZ[3][3][BB][ksize][i][j];
  lhsZ[3][4][BB][ksize][i][j] = lhsZ[3][4][BB][ksize][i][j]*pivot;
  rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]  *pivot;

  coeff = lhsZ[0][3][BB][ksize][i][j];
  lhsZ[0][4][BB][ksize][i][j]= lhsZ[0][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][3];

  coeff = lhsZ[1][3][BB][ksize][i][j];
  lhsZ[1][4][BB][ksize][i][j]= lhsZ[1][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][3];

  coeff = lhsZ[2][3][BB][ksize][i][j];
  lhsZ[2][4][BB][ksize][i][j]= lhsZ[2][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][3];

  coeff = lhsZ[4][3][BB][ksize][i][j];
  lhsZ[4][4][BB][ksize][i][j]= lhsZ[4][4][BB][ksize][i][j] - coeff*lhsZ[3][4][BB][ksize][i][j];
  rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]   - coeff*rhs[ksize][j][i][3];


  pivot = 1.00/lhsZ[4][4][BB][ksize][i][j];
  rhs[ksize][j][i][4]   = rhs[ksize][j][i][4]  *pivot;

  coeff = lhsZ[0][4][BB][ksize][i][j];
  rhs[ksize][j][i][0]   = rhs[ksize][j][i][0]   - coeff*rhs[ksize][j][i][4];

  coeff = lhsZ[1][4][BB][ksize][i][j];
  rhs[ksize][j][i][1]   = rhs[ksize][j][i][1]   - coeff*rhs[ksize][j][i][4];

  coeff = lhsZ[2][4][BB][ksize][i][j];
  rhs[ksize][j][i][2]   = rhs[ksize][j][i][2]   - coeff*rhs[ksize][j][i][4];

  coeff = lhsZ[3][4][BB][ksize][i][j];
  rhs[ksize][j][i][3]   = rhs[ksize][j][i][3]   - coeff*rhs[ksize][j][i][4];


	}
  }
      //---------------------------------------------------------------------
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // back solve: if last cell, then generate U(ksize)=rhs(ksize)
      // else assume U(ksize) is loaded in un pack backsub_info
      // so just use it
      // after u(kstart) will be sent to next cell
      //---------------------------------------------------------------------

#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
      for (k = ksize-1; k >= 0; k--) {
        for (m = 0; m < BLOCK_SIZE; m++) {
          for (n = 0; n < BLOCK_SIZE; n++) {
            rhs[k][j][i][m] = rhs[k][j][i][m] 
              - lhsZ[m][n][CC][k][i][j]*rhs[k+1][j][i][n];
          }
        }
      }
    }
  }
#ifdef USE_ASYNC
  #pragma acc wait(0)
#endif
}/*end acc data*/
#ifdef USE_UNIFIEDMEM
  acc_delete_unified(fjacZ,0);
  acc_delete_unified(njacZ,0);
  acc_delete_unified(lhsZ,0);
#else
  free(fjacZ);
  free(njacZ);
  free(lhsZ);
#endif
}

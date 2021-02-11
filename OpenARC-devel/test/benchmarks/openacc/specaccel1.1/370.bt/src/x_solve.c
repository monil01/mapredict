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
// 
// Performs line solves in X direction by first factoring
// the block-tridiagonal matrix into an upper triangular matrix, 
// and then performing back substitution to solve for the unknow
// vectors of each line.  
// 
// Make sure we treat elements zero to cell_size in the direction
// of the sweep.
// 
//---------------------------------------------------------------------
void x_solve()
{
  int i, j, k, m, n, isize, z;
//  double pivot, coeff;
  int gp22, gp12;
//  double temp1, temp2, temp3;
/*
  double fjacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double njacX[5][5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double lhsX[5][5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1];
*/
  double temp1,temp2,temp3,pivot,coeff;
  double (*fjacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double (*njacX)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1];
  double (*lhsX)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1];
#ifdef USE_UNIFIEDMEM
  fjacX = (double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1])acc_create_unified(NULL, sizeof(double)*5*5*(PROBLEM_SIZE+1)*(JMAXP-1)*(KMAX-1));
  njacX = (double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1])acc_create_unified(NULL, sizeof(double)*5*5*(PROBLEM_SIZE+1)*(JMAXP-1)*(KMAX-1));
  lhsX = (double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1])acc_create_unified(NULL, sizeof(double)*5*5*3*(PROBLEM_SIZE)*(JMAXP-1)*(KMAX-1));
#else
  fjacX = (double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1])malloc(sizeof(double)*5*5*(PROBLEM_SIZE+1)*(JMAXP-1)*(KMAX-1));
  njacX = (double (*)[5][PROBLEM_SIZE+1][JMAXP-1][KMAX-1])malloc(sizeof(double)*5*5*(PROBLEM_SIZE+1)*(JMAXP-1)*(KMAX-1));
  lhsX = (double (*)[5][3][PROBLEM_SIZE][JMAXP-1][KMAX-1])malloc(sizeof(double)*5*5*3*(PROBLEM_SIZE)*(JMAXP-1)*(KMAX-1));
#endif

  
  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;

  //---------------------------------------------------------------------
  // This function computes the left hand side in the xi-direction
  //---------------------------------------------------------------------

  isize = grid_points[0]-1;

  //---------------------------------------------------------------------
  // determine a (labeled f) and n jacobians
  //---------------------------------------------------------------------
#pragma acc data present(rho_i,u,qs,rhs,square) pcreate(njacX[0:5][0:5][0:PROBLEM_SIZE+1][0:JMAXP-1][0:KMAX-1],fjacX[0:5][0:5][0:PROBLEM_SIZE+1][0:JMAXP-1][0:KMAX-1],lhsX[0:5][0:5][0:3][0:PROBLEM_SIZE][0:JMAXP-1][0:KMAX-1]) 
{
// #pragma acc parallel loop present(rho_i,u,qs,square,fjacX,njacX) 
#ifdef USE_ASYNC
 #pragma acc kernels loop independent collapse(3) async(0)
#else
 #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 0; i <= isize; i++) {
        temp1 = rho_i[k][j][i];
        temp2 = temp1 * temp1;
        temp3 = temp1 * temp2;
        //-------------------------------------------------------------------
        // 
        //-------------------------------------------------------------------
        fjacX[0][0][i][j][k] = 0.0;
        fjacX[0][1][i][j][k] = 1.0;
        fjacX[0][2][i][j][k] = 0.0;
        fjacX[0][3][i][j][k] = 0.0;
        fjacX[0][4][i][j][k] = 0.0;

        fjacX[1][0][i][j][k] = -(u[k][j][i][1] * temp2 * u[k][j][i][1])
          + c2 * qs[k][j][i];
        fjacX[1][1][i][j][k] = ( 2.0 - c2 ) * ( u[k][j][i][1] / u[k][j][i][0] );
        fjacX[1][2][i][j][k] = - c2 * ( u[k][j][i][2] * temp1 );
        fjacX[1][3][i][j][k] = - c2 * ( u[k][j][i][3] * temp1 );
        fjacX[1][4][i][j][k] = c2;

        fjacX[2][0][i][j][k] = - ( u[k][j][i][1]*u[k][j][i][2] ) * temp2;
        fjacX[2][1][i][j][k] = u[k][j][i][2] * temp1;
        fjacX[2][2][i][j][k] = u[k][j][i][1] * temp1;
        fjacX[2][3][i][j][k] = 0.0;
        fjacX[2][4][i][j][k] = 0.0;

        fjacX[3][0][i][j][k] = - ( u[k][j][i][1]*u[k][j][i][3] ) * temp2;
        fjacX[3][1][i][j][k] = u[k][j][i][3] * temp1;
        fjacX[3][2][i][j][k] = 0.0;
        fjacX[3][3][i][j][k] = u[k][j][i][1] * temp1;
        fjacX[3][4][i][j][k] = 0.0;

        fjacX[4][0][i][j][k] = ( c2 * 2.0 * square[k][j][i] - c1 * u[k][j][i][4] )
          * ( u[k][j][i][1] * temp2 );
        fjacX[4][1][i][j][k] = c1 *  u[k][j][i][4] * temp1 
          - c2 * ( u[k][j][i][1]*u[k][j][i][1] * temp2 + qs[k][j][i] );
        fjacX[4][2][i][j][k] = - c2 * ( u[k][j][i][2]*u[k][j][i][1] ) * temp2;
        fjacX[4][3][i][j][k] = - c2 * ( u[k][j][i][3]*u[k][j][i][1] ) * temp2;
        fjacX[4][4][i][j][k] = c1 * ( u[k][j][i][1] * temp1 );

        njacX[0][0][i][j][k] = 0.0;
        njacX[0][1][i][j][k] = 0.0;
        njacX[0][2][i][j][k] = 0.0;
        njacX[0][3][i][j][k] = 0.0;
        njacX[0][4][i][j][k] = 0.0;

        njacX[1][0][i][j][k] = - con43 * c3c4 * temp2 * u[k][j][i][1];
        njacX[1][1][i][j][k] =   con43 * c3c4 * temp1;
        njacX[1][2][i][j][k] =   0.0;
        njacX[1][3][i][j][k] =   0.0;
        njacX[1][4][i][j][k] =   0.0;

        njacX[2][0][i][j][k] = - c3c4 * temp2 * u[k][j][i][2];
        njacX[2][1][i][j][k] =   0.0;
        njacX[2][2][i][j][k] =   c3c4 * temp1;
        njacX[2][3][i][j][k] =   0.0;
        njacX[2][4][i][j][k] =   0.0;

        njacX[3][0][i][j][k] = - c3c4 * temp2 * u[k][j][i][3];
        njacX[3][1][i][j][k] =   0.0;
        njacX[3][2][i][j][k] =   0.0;
        njacX[3][3][i][j][k] =   c3c4 * temp1;
        njacX[3][4][i][j][k] =   0.0;

        njacX[4][0][i][j][k] = - ( con43 * c3c4
            - c1345 ) * temp3 * (u[k][j][i][1]*u[k][j][i][1])
          - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][2]*u[k][j][i][2])
          - ( c3c4 - c1345 ) * temp3 * (u[k][j][i][3]*u[k][j][i][3])
          - c1345 * temp2 * u[k][j][i][4];

        njacX[4][1][i][j][k] = ( con43 * c3c4
            - c1345 ) * temp2 * u[k][j][i][1];
        njacX[4][2][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[k][j][i][2];
        njacX[4][3][i][j][k] = ( c3c4 - c1345 ) * temp2 * u[k][j][i][3];
        njacX[4][4][i][j][k] = ( c1345 ) * temp1;
      }
	}
  }

      //---------------------------------------------------------------------
      // now jacobians set, so form left hand side in x direction
      //---------------------------------------------------------------------
  //    lhsX[k][j]init(lhsX[k][j], isize);
  // zero the whole left hand side for starters
#ifdef USE_ASYNC
  #pragma acc kernels loop  independent collapse(4) async(0)
#else
  #pragma acc kernels loop  independent collapse(4)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
  		for (n = 0; n < 5; n++) {
		   for (m = 0; m < 5; m++){
      			lhsX[m][n][0][0][j][k] = 0.0;
      			lhsX[m][n][1][0][j][k] = 0.0;
      			lhsX[m][n][2][0][j][k] = 0.0;
      			lhsX[m][n][0][isize][j][k] = 0.0;
      			lhsX[m][n][1][isize][j][k] = 0.0;
      			lhsX[m][n][2][isize][j][k] = 0.0;
      		}
  		}
	}
  }

  // next, set all diagonal values to 1. This is overkill, but convenient
#ifdef USE_ASYNC
  #pragma acc kernels loop  independent collapse(2) async(0)
#else
  #pragma acc kernels loop  independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
    		lhsX[0][0][1][0][j][k] = 1.0;
    		lhsX[0][0][1][isize][j][k] = 1.0;
    		lhsX[1][1][1][0][j][k] = 1.0;
    		lhsX[1][1][1][isize][j][k] = 1.0;
    		lhsX[2][2][1][0][j][k] = 1.0;
    		lhsX[2][2][1][isize][j][k] = 1.0;
    		lhsX[3][3][1][0][j][k] = 1.0;
    		lhsX[3][3][1][isize][j][k] = 1.0;
    		lhsX[4][4][1][0][j][k] = 1.0;
    		lhsX[4][4][1][isize][j][k] = 1.0;
	}
  }
  
#ifdef USE_ASYNC
  #pragma acc kernels loop  independent collapse(3) async(0)
#else
  #pragma acc kernels loop  independent collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
 	   for (i = 1; i <= isize-1; i++) {
        temp1 = dt * tx1;
        temp2 = dt * tx2;

        lhsX[0][0][AA][i][j][k] = - temp2 * fjacX[0][0][i-1][j][k]
          - temp1 * njacX[0][0][i-1][j][k]
          - temp1 * dx1; 
        lhsX[0][1][AA][i][j][k] = - temp2 * fjacX[0][1][i-1][j][k]
          - temp1 * njacX[0][1][i-1][j][k];
        lhsX[0][2][AA][i][j][k] = - temp2 * fjacX[0][2][i-1][j][k]
          - temp1 * njacX[0][2][i-1][j][k];
        lhsX[0][3][AA][i][j][k] = - temp2 * fjacX[0][3][i-1][j][k]
          - temp1 * njacX[0][3][i-1][j][k];
        lhsX[0][4][AA][i][j][k] = - temp2 * fjacX[0][4][i-1][j][k]
          - temp1 * njacX[0][4][i-1][j][k];

        lhsX[1][0][AA][i][j][k] = - temp2 * fjacX[1][0][i-1][j][k]
          - temp1 * njacX[1][0][i-1][j][k];
        lhsX[1][1][AA][i][j][k] = - temp2 * fjacX[1][1][i-1][j][k]
          - temp1 * njacX[1][1][i-1][j][k]
          - temp1 * dx2;
        lhsX[1][2][AA][i][j][k] = - temp2 * fjacX[1][2][i-1][j][k]
          - temp1 * njacX[1][2][i-1][j][k];
        lhsX[1][3][AA][i][j][k] = - temp2 * fjacX[1][3][i-1][j][k]
          - temp1 * njacX[1][3][i-1][j][k];
        lhsX[1][4][AA][i][j][k] = - temp2 * fjacX[1][4][i-1][j][k]
          - temp1 * njacX[1][4][i-1][j][k];

        lhsX[2][0][AA][i][j][k] = - temp2 * fjacX[2][0][i-1][j][k]
          - temp1 * njacX[2][0][i-1][j][k];
        lhsX[2][1][AA][i][j][k] = - temp2 * fjacX[2][1][i-1][j][k]
          - temp1 * njacX[2][1][i-1][j][k];
        lhsX[2][2][AA][i][j][k] = - temp2 * fjacX[2][2][i-1][j][k]
          - temp1 * njacX[2][2][i-1][j][k]
          - temp1 * dx3;
        lhsX[2][3][AA][i][j][k] = - temp2 * fjacX[2][3][i-1][j][k]
          - temp1 * njacX[2][3][i-1][j][k];
        lhsX[2][4][AA][i][j][k] = - temp2 * fjacX[2][4][i-1][j][k]
          - temp1 * njacX[2][4][i-1][j][k];

        lhsX[3][0][AA][i][j][k] = - temp2 * fjacX[3][0][i-1][j][k]
          - temp1 * njacX[3][0][i-1][j][k];
        lhsX[3][1][AA][i][j][k] = - temp2 * fjacX[3][1][i-1][j][k]
          - temp1 * njacX[3][1][i-1][j][k];
        lhsX[3][2][AA][i][j][k] = - temp2 * fjacX[3][2][i-1][j][k]
          - temp1 * njacX[3][2][i-1][j][k];
        lhsX[3][3][AA][i][j][k] = - temp2 * fjacX[3][3][i-1][j][k]
          - temp1 * njacX[3][3][i-1][j][k]
          - temp1 * dx4;
        lhsX[3][4][AA][i][j][k] = - temp2 * fjacX[3][4][i-1][j][k]
          - temp1 * njacX[3][4][i-1][j][k];

        lhsX[4][0][AA][i][j][k] = - temp2 * fjacX[4][0][i-1][j][k]
          - temp1 * njacX[4][0][i-1][j][k];
        lhsX[4][1][AA][i][j][k] = - temp2 * fjacX[4][1][i-1][j][k]
          - temp1 * njacX[4][1][i-1][j][k];
        lhsX[4][2][AA][i][j][k] = - temp2 * fjacX[4][2][i-1][j][k]
          - temp1 * njacX[4][2][i-1][j][k];
        lhsX[4][3][AA][i][j][k] = - temp2 * fjacX[4][3][i-1][j][k]
          - temp1 * njacX[4][3][i-1][j][k];
        lhsX[4][4][AA][i][j][k] = - temp2 * fjacX[4][4][i-1][j][k]
          - temp1 * njacX[4][4][i-1][j][k]
          - temp1 * dx5;

        lhsX[0][0][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[0][0][i][j][k]
          + temp1 * 2.0 * dx1;
        lhsX[0][1][BB][i][j][k] = temp1 * 2.0 * njacX[0][1][i][j][k];
        lhsX[0][2][BB][i][j][k] = temp1 * 2.0 * njacX[0][2][i][j][k];
        lhsX[0][3][BB][i][j][k] = temp1 * 2.0 * njacX[0][3][i][j][k];
        lhsX[0][4][BB][i][j][k] = temp1 * 2.0 * njacX[0][4][i][j][k];

        lhsX[1][0][BB][i][j][k] = temp1 * 2.0 * njacX[1][0][i][j][k];
        lhsX[1][1][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[1][1][i][j][k]
          + temp1 * 2.0 * dx2;
        lhsX[1][2][BB][i][j][k] = temp1 * 2.0 * njacX[1][2][i][j][k];
        lhsX[1][3][BB][i][j][k] = temp1 * 2.0 * njacX[1][3][i][j][k];
        lhsX[1][4][BB][i][j][k] = temp1 * 2.0 * njacX[1][4][i][j][k];

        lhsX[2][0][BB][i][j][k] = temp1 * 2.0 * njacX[2][0][i][j][k];
        lhsX[2][1][BB][i][j][k] = temp1 * 2.0 * njacX[2][1][i][j][k];
        lhsX[2][2][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[2][2][i][j][k]
          + temp1 * 2.0 * dx3;
        lhsX[2][3][BB][i][j][k] = temp1 * 2.0 * njacX[2][3][i][j][k];
        lhsX[2][4][BB][i][j][k] = temp1 * 2.0 * njacX[2][4][i][j][k];

        lhsX[3][0][BB][i][j][k] = temp1 * 2.0 * njacX[3][0][i][j][k];
        lhsX[3][1][BB][i][j][k] = temp1 * 2.0 * njacX[3][1][i][j][k];
        lhsX[3][2][BB][i][j][k] = temp1 * 2.0 * njacX[3][2][i][j][k];
        lhsX[3][3][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[3][3][i][j][k]
          + temp1 * 2.0 * dx4;
        lhsX[3][4][BB][i][j][k] = temp1 * 2.0 * njacX[3][4][i][j][k];

        lhsX[4][0][BB][i][j][k] = temp1 * 2.0 * njacX[4][0][i][j][k];
        lhsX[4][1][BB][i][j][k] = temp1 * 2.0 * njacX[4][1][i][j][k];
        lhsX[4][2][BB][i][j][k] = temp1 * 2.0 * njacX[4][2][i][j][k];
        lhsX[4][3][BB][i][j][k] = temp1 * 2.0 * njacX[4][3][i][j][k];
        lhsX[4][4][BB][i][j][k] = 1.0
          + temp1 * 2.0 * njacX[4][4][i][j][k]
          + temp1 * 2.0 * dx5;

        lhsX[0][0][CC][i][j][k] =  temp2 * fjacX[0][0][i+1][j][k]
          - temp1 * njacX[0][0][i+1][j][k]
          - temp1 * dx1;
        lhsX[0][1][CC][i][j][k] =  temp2 * fjacX[0][1][i+1][j][k]
          - temp1 * njacX[0][1][i+1][j][k];
        lhsX[0][2][CC][i][j][k] =  temp2 * fjacX[0][2][i+1][j][k]
          - temp1 * njacX[0][2][i+1][j][k];
        lhsX[0][3][CC][i][j][k] =  temp2 * fjacX[0][3][i+1][j][k]
          - temp1 * njacX[0][3][i+1][j][k];
        lhsX[0][4][CC][i][j][k] =  temp2 * fjacX[0][4][i+1][j][k]
          - temp1 * njacX[0][4][i+1][j][k];

        lhsX[1][0][CC][i][j][k] =  temp2 * fjacX[1][0][i+1][j][k]
          - temp1 * njacX[1][0][i+1][j][k];
        lhsX[1][1][CC][i][j][k] =  temp2 * fjacX[1][1][i+1][j][k]
          - temp1 * njacX[1][1][i+1][j][k]
          - temp1 * dx2;
        lhsX[1][2][CC][i][j][k] =  temp2 * fjacX[1][2][i+1][j][k]
          - temp1 * njacX[1][2][i+1][j][k];
        lhsX[1][3][CC][i][j][k] =  temp2 * fjacX[1][3][i+1][j][k]
          - temp1 * njacX[1][3][i+1][j][k];
        lhsX[1][4][CC][i][j][k] =  temp2 * fjacX[1][4][i+1][j][k]
          - temp1 * njacX[1][4][i+1][j][k];

        lhsX[2][0][CC][i][j][k] =  temp2 * fjacX[2][0][i+1][j][k]
          - temp1 * njacX[2][0][i+1][j][k];
        lhsX[2][1][CC][i][j][k] =  temp2 * fjacX[2][1][i+1][j][k]
          - temp1 * njacX[2][1][i+1][j][k];
        lhsX[2][2][CC][i][j][k] =  temp2 * fjacX[2][2][i+1][j][k]
          - temp1 * njacX[2][2][i+1][j][k]
          - temp1 * dx3;
        lhsX[2][3][CC][i][j][k] =  temp2 * fjacX[2][3][i+1][j][k]
          - temp1 * njacX[2][3][i+1][j][k];
        lhsX[2][4][CC][i][j][k] =  temp2 * fjacX[2][4][i+1][j][k]
          - temp1 * njacX[2][4][i+1][j][k];

        lhsX[3][0][CC][i][j][k] =  temp2 * fjacX[3][0][i+1][j][k]
          - temp1 * njacX[3][0][i+1][j][k];
        lhsX[3][1][CC][i][j][k] =  temp2 * fjacX[3][1][i+1][j][k]
          - temp1 * njacX[3][1][i+1][j][k];
        lhsX[3][2][CC][i][j][k] =  temp2 * fjacX[3][2][i+1][j][k]
          - temp1 * njacX[3][2][i+1][j][k];
        lhsX[3][3][CC][i][j][k] =  temp2 * fjacX[3][3][i+1][j][k]
          - temp1 * njacX[3][3][i+1][j][k]
          - temp1 * dx4;
        lhsX[3][4][CC][i][j][k] =  temp2 * fjacX[3][4][i+1][j][k]
          - temp1 * njacX[3][4][i+1][j][k];

        lhsX[4][0][CC][i][j][k] =  temp2 * fjacX[4][0][i+1][j][k]
          - temp1 * njacX[4][0][i+1][j][k];
        lhsX[4][1][CC][i][j][k] =  temp2 * fjacX[4][1][i+1][j][k]
          - temp1 * njacX[4][1][i+1][j][k];
        lhsX[4][2][CC][i][j][k] =  temp2 * fjacX[4][2][i+1][j][k]
          - temp1 * njacX[4][2][i+1][j][k];
        lhsX[4][3][CC][i][j][k] =  temp2 * fjacX[4][3][i+1][j][k]
          - temp1 * njacX[4][3][i+1][j][k];
        lhsX[4][4][CC][i][j][k] =  temp2 * fjacX[4][4][i+1][j][k]
          - temp1 * njacX[4][4][i+1][j][k]
          - temp1 * dx5;
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
      // c'(IMAX) and rhs'(IMAX) will be sent to next cell
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // outer most do loops - sweeping in i direction
      //---------------------------------------------------------------------

      //---------------------------------------------------------------------
      // multiply c[k][j][0] by b_inverse and copy back to c
      // multiply rhs(0) by b_inverse(0) and copy to rhs
      //---------------------------------------------------------------------
      //binvcrhs( lhsX[0][j][BB], lhsX[k][0][j][k][CC], rhs[k][j][0] );
#ifdef USE_ASYNC
  #pragma acc kernels loop  independent collapse(2) async(0)
#else
  #pragma acc kernels loop  independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
/*
	  for(m = 0; m < 5; m++){
	  	pivot = 1.00/lhsX[m][m][BB][0][j][k];
		for(n = m+1; n < 5; n++){
			lhsX[m][n][BB][0][j][k] = lhsX[m][n][BB][0][j][k]*pivot;
		}
		lhsX[m][0][CC][0][j][k] = lhsX[m][0][CC][0][j][k]*pivot;
		lhsX[m][1][CC][0][j][k] = lhsX[m][1][CC][0][j][k]*pivot;
		lhsX[m][2][CC][0][j][k] = lhsX[m][2][CC][0][j][k]*pivot;
		lhsX[m][3][CC][0][j][k] = lhsX[m][3][CC][0][j][k]*pivot;
		lhsX[m][4][CC][0][j][k] = lhsX[m][4][CC][0][j][k]*pivot;
		rhs[k][j][0][m] = rhs[k][j][0][m]*pivot;
		for(n = 0; n < 5; n++){
			if(n != m){
				coeff = lhsX[n][m][BB][0][j][k];
				for(z = m+1; z < 5; z++){
					lhsX[n][z][BB][0][j][k] = lhsX[n][z][BB][0][j][k] - coeff*lhsX[m][z][BB][0][j][k];
				}
				lhsX[n][0][CC][0][j][k] = lhsX[n][0][CC][0][j][k] - coeff*lhsX[m][0][CC][0][j][k];
				lhsX[n][1][CC][0][j][k] = lhsX[n][1][CC][0][j][k] - coeff*lhsX[m][1][CC][0][j][k];
				lhsX[n][2][CC][0][j][k] = lhsX[n][2][CC][0][j][k] - coeff*lhsX[m][2][CC][0][j][k];
				lhsX[n][3][CC][0][j][k] = lhsX[n][3][CC][0][j][k] - coeff*lhsX[m][3][CC][0][j][k];
				lhsX[n][4][CC][0][j][k] = lhsX[n][4][CC][0][j][k] - coeff*lhsX[m][4][CC][0][j][k];
				rhs[k][j][0][n] = rhs[k][j][0][n] - coeff*rhs[k][j][0][m];
			}
		}
	  }
*/
  pivot = 1.00/lhsX[0][0][BB][0][j][k];
  lhsX[0][1][BB][0][j][k] = lhsX[0][1][BB][0][j][k]*pivot;
  lhsX[0][2][BB][0][j][k] = lhsX[0][2][BB][0][j][k]*pivot;
  lhsX[0][3][BB][0][j][k] = lhsX[0][3][BB][0][j][k]*pivot;
  lhsX[0][4][BB][0][j][k] = lhsX[0][4][BB][0][j][k]*pivot;
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k]*pivot;
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k]*pivot;
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k]*pivot;
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k]*pivot;
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k]*pivot;
  rhs[k][j][0][0]   = rhs[k][j][0][0]  *pivot;

  coeff = lhsX[1][0][BB][0][j][k];
  lhsX[1][1][BB][0][j][k]= lhsX[1][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[1][2][BB][0][j][k]= lhsX[1][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][0];

  coeff = lhsX[2][0][BB][0][j][k];
  lhsX[2][1][BB][0][j][k]= lhsX[2][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][0];

  coeff = lhsX[3][0][BB][0][j][k];
  lhsX[3][1][BB][0][j][k]= lhsX[3][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][0];

  coeff = lhsX[4][0][BB][0][j][k];
  lhsX[4][1][BB][0][j][k]= lhsX[4][1][BB][0][j][k] - coeff*lhsX[0][1][BB][0][j][k];
  lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[0][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[0][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[0][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[0][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[0][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[0][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[0][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[0][4][CC][0][j][k];
  rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][0];


  pivot = 1.00/lhsX[1][1][BB][0][j][k];
  lhsX[1][2][BB][0][j][k] = lhsX[1][2][BB][0][j][k]*pivot;
  lhsX[1][3][BB][0][j][k] = lhsX[1][3][BB][0][j][k]*pivot;
  lhsX[1][4][BB][0][j][k] = lhsX[1][4][BB][0][j][k]*pivot;
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k]*pivot;
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k]*pivot;
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k]*pivot;
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k]*pivot;
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k]*pivot;
  rhs[k][j][0][1]   = rhs[k][j][0][1]  *pivot;

  coeff = lhsX[0][1][BB][0][j][k];
  lhsX[0][2][BB][0][j][k]= lhsX[0][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][1];

  coeff = lhsX[2][1][BB][0][j][k];
  lhsX[2][2][BB][0][j][k]= lhsX[2][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k]= lhsX[2][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][1];

  coeff = lhsX[3][1][BB][0][j][k];
  lhsX[3][2][BB][0][j][k]= lhsX[3][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][1];

  coeff = lhsX[4][1][BB][0][j][k];
  lhsX[4][2][BB][0][j][k]= lhsX[4][2][BB][0][j][k] - coeff*lhsX[1][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[1][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[1][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[1][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[1][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[1][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[1][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[1][4][CC][0][j][k];
  rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][1];


  pivot = 1.00/lhsX[2][2][BB][0][j][k];
  lhsX[2][3][BB][0][j][k] = lhsX[2][3][BB][0][j][k]*pivot;
  lhsX[2][4][BB][0][j][k] = lhsX[2][4][BB][0][j][k]*pivot;
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k]*pivot;
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k]*pivot;
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k]*pivot;
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k]*pivot;
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k]*pivot;
  rhs[k][j][0][2]   = rhs[k][j][0][2]  *pivot;

  coeff = lhsX[0][2][BB][0][j][k];
  lhsX[0][3][BB][0][j][k]= lhsX[0][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][2];

  coeff = lhsX[1][2][BB][0][j][k];
  lhsX[1][3][BB][0][j][k]= lhsX[1][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][2];

  coeff = lhsX[3][2][BB][0][j][k];
  lhsX[3][3][BB][0][j][k]= lhsX[3][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k]= lhsX[3][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][2];

  coeff = lhsX[4][2][BB][0][j][k];
  lhsX[4][3][BB][0][j][k]= lhsX[4][3][BB][0][j][k] - coeff*lhsX[2][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[2][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[2][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[2][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[2][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[2][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[2][4][CC][0][j][k];
  rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][2];


  pivot = 1.00/lhsX[3][3][BB][0][j][k];
  lhsX[3][4][BB][0][j][k] = lhsX[3][4][BB][0][j][k]*pivot;
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k]*pivot;
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k]*pivot;
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k]*pivot;
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k]*pivot;
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k]*pivot;
  rhs[k][j][0][3]   = rhs[k][j][0][3]  *pivot;

  coeff = lhsX[0][3][BB][0][j][k];
  lhsX[0][4][BB][0][j][k]= lhsX[0][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][3];

  coeff = lhsX[1][3][BB][0][j][k];
  lhsX[1][4][BB][0][j][k]= lhsX[1][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][3];

  coeff = lhsX[2][3][BB][0][j][k];
  lhsX[2][4][BB][0][j][k]= lhsX[2][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][3];

  coeff = lhsX[4][3][BB][0][j][k];
  lhsX[4][4][BB][0][j][k]= lhsX[4][4][BB][0][j][k] - coeff*lhsX[3][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k] - coeff*lhsX[3][0][CC][0][j][k];
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k] - coeff*lhsX[3][1][CC][0][j][k];
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k] - coeff*lhsX[3][2][CC][0][j][k];
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k] - coeff*lhsX[3][3][CC][0][j][k];
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k] - coeff*lhsX[3][4][CC][0][j][k];
  rhs[k][j][0][4]   = rhs[k][j][0][4]   - coeff*rhs[k][j][0][3];


  pivot = 1.00/lhsX[4][4][BB][0][j][k];
  lhsX[4][0][CC][0][j][k] = lhsX[4][0][CC][0][j][k]*pivot;
  lhsX[4][1][CC][0][j][k] = lhsX[4][1][CC][0][j][k]*pivot;
  lhsX[4][2][CC][0][j][k] = lhsX[4][2][CC][0][j][k]*pivot;
  lhsX[4][3][CC][0][j][k] = lhsX[4][3][CC][0][j][k]*pivot;
  lhsX[4][4][CC][0][j][k] = lhsX[4][4][CC][0][j][k]*pivot;
  rhs[k][j][0][4]   = rhs[k][j][0][4]  *pivot;

  coeff = lhsX[0][4][BB][0][j][k];
  lhsX[0][0][CC][0][j][k] = lhsX[0][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[0][1][CC][0][j][k] = lhsX[0][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[0][2][CC][0][j][k] = lhsX[0][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[0][3][CC][0][j][k] = lhsX[0][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[0][4][CC][0][j][k] = lhsX[0][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[k][j][0][0]   = rhs[k][j][0][0]   - coeff*rhs[k][j][0][4];

  coeff = lhsX[1][4][BB][0][j][k];
  lhsX[1][0][CC][0][j][k] = lhsX[1][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[1][1][CC][0][j][k] = lhsX[1][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[1][2][CC][0][j][k] = lhsX[1][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[1][3][CC][0][j][k] = lhsX[1][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[1][4][CC][0][j][k] = lhsX[1][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[k][j][0][1]   = rhs[k][j][0][1]   - coeff*rhs[k][j][0][4];

  coeff = lhsX[2][4][BB][0][j][k];
  lhsX[2][0][CC][0][j][k] = lhsX[2][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[2][1][CC][0][j][k] = lhsX[2][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[2][2][CC][0][j][k] = lhsX[2][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[2][3][CC][0][j][k] = lhsX[2][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[2][4][CC][0][j][k] = lhsX[2][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[k][j][0][2]   = rhs[k][j][0][2]   - coeff*rhs[k][j][0][4];

  coeff = lhsX[3][4][BB][0][j][k];
  lhsX[3][0][CC][0][j][k] = lhsX[3][0][CC][0][j][k] - coeff*lhsX[4][0][CC][0][j][k];
  lhsX[3][1][CC][0][j][k] = lhsX[3][1][CC][0][j][k] - coeff*lhsX[4][1][CC][0][j][k];
  lhsX[3][2][CC][0][j][k] = lhsX[3][2][CC][0][j][k] - coeff*lhsX[4][2][CC][0][j][k];
  lhsX[3][3][CC][0][j][k] = lhsX[3][3][CC][0][j][k] - coeff*lhsX[4][3][CC][0][j][k];
  lhsX[3][4][CC][0][j][k] = lhsX[3][4][CC][0][j][k] - coeff*lhsX[4][4][CC][0][j][k];
  rhs[k][j][0][3]   = rhs[k][j][0][3]   - coeff*rhs[k][j][0][4];
	

	}/*end j*/
  }/*end k*/

      //---------------------------------------------------------------------
      // begin inner most do loop
      // do all the elements of the cell unless last 
      //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= isize-1; i++) {
        //-------------------------------------------------------------------
        // rhs(i) = rhs(i) - A*rhs(i-1)
        //-------------------------------------------------------------------
        //matvec_sub(lhsX[i-1][j][AA], rhs[k][i][j][k], rhs[k][j][i]);
		/*
		for(m = 0; m < 5; m++){
			rhs[k][j][i][m] = rhs[k][j][i][m] - lhsX[m][0][AA][i][j][k]*rhs[k][j][i-1][0]
											  - lhsX[m][1][AA][i][j][k]*rhs[k][j][i-1][1]
											  - lhsX[m][2][AA][i][j][k]*rhs[k][j][i-1][2]
											  - lhsX[m][3][AA][i][j][k]*rhs[k][j][i-1][3]
											  - lhsX[m][4][AA][i][j][k]*rhs[k][j][i-1][4];
		}
		*/
  rhs[k][j][i][0] = rhs[k][j][i][0] - lhsX[0][0][AA][i][j][k]*rhs[k][j][i-1][0]
                    - lhsX[0][1][AA][i][j][k]*rhs[k][j][i-1][1]
                    - lhsX[0][2][AA][i][j][k]*rhs[k][j][i-1][2]
                    - lhsX[0][3][AA][i][j][k]*rhs[k][j][i-1][3]
                    - lhsX[0][4][AA][i][j][k]*rhs[k][j][i-1][4];
  rhs[k][j][i][1] = rhs[k][j][i][1] - lhsX[1][0][AA][i][j][k]*rhs[k][j][i-1][0]
                    - lhsX[1][1][AA][i][j][k]*rhs[k][j][i-1][1]
                    - lhsX[1][2][AA][i][j][k]*rhs[k][j][i-1][2]
                    - lhsX[1][3][AA][i][j][k]*rhs[k][j][i-1][3]
                    - lhsX[1][4][AA][i][j][k]*rhs[k][j][i-1][4];
  rhs[k][j][i][2] = rhs[k][j][i][2] - lhsX[2][0][AA][i][j][k]*rhs[k][j][i-1][0]
                    - lhsX[2][1][AA][i][j][k]*rhs[k][j][i-1][1]
                    - lhsX[2][2][AA][i][j][k]*rhs[k][j][i-1][2]
                    - lhsX[2][3][AA][i][j][k]*rhs[k][j][i-1][3]
                    - lhsX[2][4][AA][i][j][k]*rhs[k][j][i-1][4];
  rhs[k][j][i][3] = rhs[k][j][i][3] - lhsX[3][0][AA][i][j][k]*rhs[k][j][i-1][0]
                    - lhsX[3][1][AA][i][j][k]*rhs[k][j][i-1][1]
                    - lhsX[3][2][AA][i][j][k]*rhs[k][j][i-1][2]
                    - lhsX[3][3][AA][i][j][k]*rhs[k][j][i-1][3]
                    - lhsX[3][4][AA][i][j][k]*rhs[k][j][i-1][4];
  rhs[k][j][i][4] = rhs[k][j][i][4] - lhsX[4][0][AA][i][j][k]*rhs[k][j][i-1][0]
                    - lhsX[4][1][AA][i][j][k]*rhs[k][j][i-1][1]
                    - lhsX[4][2][AA][i][j][k]*rhs[k][j][i-1][2]
                    - lhsX[4][3][AA][i][j][k]*rhs[k][j][i-1][3]
                    - lhsX[4][4][AA][i][j][k]*rhs[k][j][i-1][4];
		

        //-------------------------------------------------------------------
        // B(i) = B(i) - C(i-1)*A(i)
        //-------------------------------------------------------------------
      //  matmul_sub(lhsX[i-1][j][AA], lhsX[k][i][j][k][CC], lhsX[k][j][i][BB]);
	  /*
	  	for(m = 0; m < 5; m++){
	  		for(n = 0; n < 5; n++){
			lhsX[n][m][BB][i][j][k] = lhsX[n][m][BB][i][j][k] - lhsX[n][0][AA][i][j][k]*lhsX[0][m][CC][i-1][j][k]
												- lhsX[n][1][AA][i][j][k]*lhsX[1][m][CC][i-1][j][k]
												- lhsX[n][2][AA][i][j][k]*lhsX[2][m][CC][i-1][j][k]
												- lhsX[n][3][AA][i][j][k]*lhsX[3][m][CC][i-1][j][k]
												- lhsX[n][4][AA][i][j][k]*lhsX[4][m][CC][i-1][j][k];
		}
	  }
		*/
  lhsX[0][0][BB][i][j][k] = lhsX[0][0][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[1][0][BB][i][j][k] = lhsX[1][0][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[2][0][BB][i][j][k] = lhsX[2][0][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[3][0][BB][i][j][k] = lhsX[3][0][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[4][0][BB][i][j][k] = lhsX[4][0][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][0][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][0][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][0][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][0][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][0][CC][i-1][j][k];
  lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[1][1][BB][i][j][k] = lhsX[1][1][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[2][1][BB][i][j][k] = lhsX[2][1][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[3][1][BB][i][j][k] = lhsX[3][1][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[4][1][BB][i][j][k] = lhsX[4][1][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][1][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][1][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][1][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][1][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][1][CC][i-1][j][k];
  lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[2][2][BB][i][j][k] = lhsX[2][2][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[3][2][BB][i][j][k] = lhsX[3][2][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[4][2][BB][i][j][k] = lhsX[4][2][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][2][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][2][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][2][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][2][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][2][CC][i-1][j][k];
  lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[3][3][BB][i][j][k] = lhsX[3][3][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[4][3][BB][i][j][k] = lhsX[4][3][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][3][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][3][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][3][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][3][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][3][CC][i-1][j][k];
  lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k] - lhsX[0][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[0][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[0][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[0][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[0][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k] - lhsX[1][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[1][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[1][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[1][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[1][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k] - lhsX[2][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[2][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[2][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[2][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[2][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k] - lhsX[3][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[3][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[3][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[3][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[3][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];
  lhsX[4][4][BB][i][j][k] = lhsX[4][4][BB][i][j][k] - lhsX[4][0][AA][i][j][k]*lhsX[0][4][CC][i-1][j][k]
                              - lhsX[4][1][AA][i][j][k]*lhsX[1][4][CC][i-1][j][k]
                              - lhsX[4][2][AA][i][j][k]*lhsX[2][4][CC][i-1][j][k]
                              - lhsX[4][3][AA][i][j][k]*lhsX[3][4][CC][i-1][j][k]
                              - lhsX[4][4][AA][i][j][k]*lhsX[4][4][CC][i-1][j][k];

        //-------------------------------------------------------------------
        // multiply c[k][j][i] by b_inverse and copy back to c
        // multiply rhs[k][j][0] by b_inverse[k][j][0] and copy to rhs
        //-------------------------------------------------------------------
        //binvcrhs( lhsX[i][j][BB], lhsX[k][i][j][k][CC], rhs[k][j][i] );
		/*
	  	for(m = 0; m < 5; m++){
	  		pivot = 1.00/lhsX[m][m][BB][i][j][k];
			for(n = m+1; n < 5; n++){
				lhsX[m][n][BB][i][j][k] = lhsX[m][n][BB][i][j][k]*pivot;
			}
			lhsX[m][0][CC][i][j][k] = lhsX[m][0][CC][i][j][k]*pivot;
			lhsX[m][1][CC][i][j][k] = lhsX[m][1][CC][i][j][k]*pivot;
			lhsX[m][2][CC][i][j][k] = lhsX[m][2][CC][i][j][k]*pivot;
			lhsX[m][3][CC][i][j][k] = lhsX[m][3][CC][i][j][k]*pivot;
			lhsX[m][4][CC][i][j][k] = lhsX[m][4][CC][i][j][k]*pivot;
			rhs[k][j][i][m] = rhs[k][j][i][m]*pivot;

			for(n = 0; n < 5; n++){
			   if(n != m){
					coeff = lhsX[n][m][BB][i][j][k];
					for(z = m+1; z < 5; z++){
						lhsX[n][z][BB][i][j][k] = lhsX[n][z][BB][i][j][k] - coeff*lhsX[m][z][BB][i][j][k];
					}
					lhsX[n][0][CC][i][j][k] = lhsX[n][0][CC][i][j][k] - coeff*lhsX[m][0][CC][i][j][k];
					lhsX[n][1][CC][i][j][k] = lhsX[n][1][CC][i][j][k] - coeff*lhsX[m][1][CC][i][j][k];
					lhsX[n][2][CC][i][j][k] = lhsX[n][2][CC][i][j][k] - coeff*lhsX[m][2][CC][i][j][k];
					lhsX[n][3][CC][i][j][k] = lhsX[n][3][CC][i][j][k] - coeff*lhsX[m][3][CC][i][j][k];
					lhsX[n][4][CC][i][j][k] = lhsX[n][4][CC][i][j][k] - coeff*lhsX[m][4][CC][i][j][k];
					rhs[k][j][i][n] = rhs[k][j][i][n] - coeff*rhs[k][j][i][m];
				}
			}
	  	}
		*/
  pivot = 1.00/lhsX[0][0][BB][i][j][k];
  lhsX[0][1][BB][i][j][k] = lhsX[0][1][BB][i][j][k]*pivot;
  lhsX[0][2][BB][i][j][k] = lhsX[0][2][BB][i][j][k]*pivot;
  lhsX[0][3][BB][i][j][k] = lhsX[0][3][BB][i][j][k]*pivot;
  lhsX[0][4][BB][i][j][k] = lhsX[0][4][BB][i][j][k]*pivot;
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k]*pivot;
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k]*pivot;
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k]*pivot;
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k]*pivot;
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k]*pivot;
  rhs[k][j][i][0]   = rhs[k][j][i][0]  *pivot;

  coeff = lhsX[1][0][BB][i][j][k];
  lhsX[1][1][BB][i][j][k]= lhsX[1][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[1][2][BB][i][j][k]= lhsX[1][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][0];

  coeff = lhsX[2][0][BB][i][j][k];
  lhsX[2][1][BB][i][j][k]= lhsX[2][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][0];

  coeff = lhsX[3][0][BB][i][j][k];
  lhsX[3][1][BB][i][j][k]= lhsX[3][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][0];

  coeff = lhsX[4][0][BB][i][j][k];
  lhsX[4][1][BB][i][j][k]= lhsX[4][1][BB][i][j][k] - coeff*lhsX[0][1][BB][i][j][k];
  lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[0][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[0][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[0][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[0][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[0][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[0][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[0][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[0][4][CC][i][j][k];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][0];


  pivot = 1.00/lhsX[1][1][BB][i][j][k];
  lhsX[1][2][BB][i][j][k] = lhsX[1][2][BB][i][j][k]*pivot;
  lhsX[1][3][BB][i][j][k] = lhsX[1][3][BB][i][j][k]*pivot;
  lhsX[1][4][BB][i][j][k] = lhsX[1][4][BB][i][j][k]*pivot;
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k]*pivot;
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k]*pivot;
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k]*pivot;
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k]*pivot;
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k]*pivot;
  rhs[k][j][i][1]   = rhs[k][j][i][1]  *pivot;

  coeff = lhsX[0][1][BB][i][j][k];
  lhsX[0][2][BB][i][j][k]= lhsX[0][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][1];

  coeff = lhsX[2][1][BB][i][j][k];
  lhsX[2][2][BB][i][j][k]= lhsX[2][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k]= lhsX[2][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][1];

  coeff = lhsX[3][1][BB][i][j][k];
  lhsX[3][2][BB][i][j][k]= lhsX[3][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][1];

  coeff = lhsX[4][1][BB][i][j][k];
  lhsX[4][2][BB][i][j][k]= lhsX[4][2][BB][i][j][k] - coeff*lhsX[1][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[1][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[1][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[1][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[1][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[1][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[1][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[1][4][CC][i][j][k];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][1];


  pivot = 1.00/lhsX[2][2][BB][i][j][k];
  lhsX[2][3][BB][i][j][k] = lhsX[2][3][BB][i][j][k]*pivot;
  lhsX[2][4][BB][i][j][k] = lhsX[2][4][BB][i][j][k]*pivot;
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k]*pivot;
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k]*pivot;
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k]*pivot;
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k]*pivot;
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k]*pivot;
  rhs[k][j][i][2]   = rhs[k][j][i][2]  *pivot;

  coeff = lhsX[0][2][BB][i][j][k];
  lhsX[0][3][BB][i][j][k]= lhsX[0][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][2];

  coeff = lhsX[1][2][BB][i][j][k];
  lhsX[1][3][BB][i][j][k]= lhsX[1][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][2];

  coeff = lhsX[3][2][BB][i][j][k];
  lhsX[3][3][BB][i][j][k]= lhsX[3][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k]= lhsX[3][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][2];

  coeff = lhsX[4][2][BB][i][j][k];
  lhsX[4][3][BB][i][j][k]= lhsX[4][3][BB][i][j][k] - coeff*lhsX[2][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[2][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[2][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[2][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[2][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[2][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[2][4][CC][i][j][k];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][2];


  pivot = 1.00/lhsX[3][3][BB][i][j][k];
  lhsX[3][4][BB][i][j][k] = lhsX[3][4][BB][i][j][k]*pivot;
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k]*pivot;
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k]*pivot;
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k]*pivot;
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k]*pivot;
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k]*pivot;
  rhs[k][j][i][3]   = rhs[k][j][i][3]  *pivot;

  coeff = lhsX[0][3][BB][i][j][k];
  lhsX[0][4][BB][i][j][k]= lhsX[0][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][3];

  coeff = lhsX[1][3][BB][i][j][k];
  lhsX[1][4][BB][i][j][k]= lhsX[1][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][3];

  coeff = lhsX[2][3][BB][i][j][k];
  lhsX[2][4][BB][i][j][k]= lhsX[2][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][3];

  coeff = lhsX[4][3][BB][i][j][k];
  lhsX[4][4][BB][i][j][k]= lhsX[4][4][BB][i][j][k] - coeff*lhsX[3][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k] - coeff*lhsX[3][0][CC][i][j][k];
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k] - coeff*lhsX[3][1][CC][i][j][k];
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k] - coeff*lhsX[3][2][CC][i][j][k];
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k] - coeff*lhsX[3][3][CC][i][j][k];
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k] - coeff*lhsX[3][4][CC][i][j][k];
  rhs[k][j][i][4]   = rhs[k][j][i][4]   - coeff*rhs[k][j][i][3];


  pivot = 1.00/lhsX[4][4][BB][i][j][k];
  lhsX[4][0][CC][i][j][k] = lhsX[4][0][CC][i][j][k]*pivot;
  lhsX[4][1][CC][i][j][k] = lhsX[4][1][CC][i][j][k]*pivot;
  lhsX[4][2][CC][i][j][k] = lhsX[4][2][CC][i][j][k]*pivot;
  lhsX[4][3][CC][i][j][k] = lhsX[4][3][CC][i][j][k]*pivot;
  lhsX[4][4][CC][i][j][k] = lhsX[4][4][CC][i][j][k]*pivot;
  rhs[k][j][i][4]   = rhs[k][j][i][4]  *pivot;

  coeff = lhsX[0][4][BB][i][j][k];
  lhsX[0][0][CC][i][j][k] = lhsX[0][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[0][1][CC][i][j][k] = lhsX[0][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[0][2][CC][i][j][k] = lhsX[0][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[0][3][CC][i][j][k] = lhsX[0][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[0][4][CC][i][j][k] = lhsX[0][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[k][j][i][0]   = rhs[k][j][i][0]   - coeff*rhs[k][j][i][4];

  coeff = lhsX[1][4][BB][i][j][k];
  lhsX[1][0][CC][i][j][k] = lhsX[1][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[1][1][CC][i][j][k] = lhsX[1][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[1][2][CC][i][j][k] = lhsX[1][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[1][3][CC][i][j][k] = lhsX[1][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[1][4][CC][i][j][k] = lhsX[1][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[k][j][i][1]   = rhs[k][j][i][1]   - coeff*rhs[k][j][i][4];

  coeff = lhsX[2][4][BB][i][j][k];
  lhsX[2][0][CC][i][j][k] = lhsX[2][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[2][1][CC][i][j][k] = lhsX[2][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[2][2][CC][i][j][k] = lhsX[2][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[2][3][CC][i][j][k] = lhsX[2][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[2][4][CC][i][j][k] = lhsX[2][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[k][j][i][2]   = rhs[k][j][i][2]   - coeff*rhs[k][j][i][4];

  coeff = lhsX[3][4][BB][i][j][k];
  lhsX[3][0][CC][i][j][k] = lhsX[3][0][CC][i][j][k] - coeff*lhsX[4][0][CC][i][j][k];
  lhsX[3][1][CC][i][j][k] = lhsX[3][1][CC][i][j][k] - coeff*lhsX[4][1][CC][i][j][k];
  lhsX[3][2][CC][i][j][k] = lhsX[3][2][CC][i][j][k] - coeff*lhsX[4][2][CC][i][j][k];
  lhsX[3][3][CC][i][j][k] = lhsX[3][3][CC][i][j][k] - coeff*lhsX[4][3][CC][i][j][k];
  lhsX[3][4][CC][i][j][k] = lhsX[3][4][CC][i][j][k] - coeff*lhsX[4][4][CC][i][j][k];
  rhs[k][j][i][3]   = rhs[k][j][i][3]   - coeff*rhs[k][j][i][4];


      }/*end i*/
    }
  }
      //---------------------------------------------------------------------
      // rhs(isize) = rhs(isize) - A*rhs(isize-1)
      //---------------------------------------------------------------------
      //matvec_sub(lhsX[isize-1][j][AA], rhs[k][isize][j][k], rhs[k][j][isize]);
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
	/*
		for(m = 0; m < 5; m++){
			rhs[k][j][isize][m] = rhs[k][j][isize][m] - lhsX[m][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
											          - lhsX[m][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
											          - lhsX[m][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
											          - lhsX[m][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
											          - lhsX[m][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
		}
	*/
  rhs[k][j][isize][0] = rhs[k][j][isize][0] - lhsX[0][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
                    - lhsX[0][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
                    - lhsX[0][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
                    - lhsX[0][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
                    - lhsX[0][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
  rhs[k][j][isize][1] = rhs[k][j][isize][1] - lhsX[1][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
                    - lhsX[1][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
                    - lhsX[1][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
                    - lhsX[1][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
                    - lhsX[1][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
  rhs[k][j][isize][2] = rhs[k][j][isize][2] - lhsX[2][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
                    - lhsX[2][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
                    - lhsX[2][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
                    - lhsX[2][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
                    - lhsX[2][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
  rhs[k][j][isize][3] = rhs[k][j][isize][3] - lhsX[3][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
                    - lhsX[3][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
                    - lhsX[3][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
                    - lhsX[3][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
                    - lhsX[3][4][AA][isize][j][k]*rhs[k][j][isize-1][4];
  rhs[k][j][isize][4] = rhs[k][j][isize][4] - lhsX[4][0][AA][isize][j][k]*rhs[k][j][isize-1][0]
                    - lhsX[4][1][AA][isize][j][k]*rhs[k][j][isize-1][1]
                    - lhsX[4][2][AA][isize][j][k]*rhs[k][j][isize-1][2]
                    - lhsX[4][3][AA][isize][j][k]*rhs[k][j][isize-1][3]
                    - lhsX[4][4][AA][isize][j][k]*rhs[k][j][isize-1][4];


	 }
  }
      //---------------------------------------------------------------------
      // B(isize) = B(isize) - C(isize-1)*A(isize)
      //---------------------------------------------------------------------
      //matmul_sub(lhsX[isize-1][j][AA], lhsX[k][isize][j][k][CC], lhsX[k][j][isize][BB]);
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
	/*
	  for(m = 0; m < 5; m++){
	  	for(n = 0; n < 5; n++){
			lhsX[n][m][BB][isize][j][k] = lhsX[n][m][BB][isize][j][k] - lhsX[n][0][AA][isize][j][k]*lhsX[0][m][CC][isize-1][j][k]
														- lhsX[n][1][AA][isize][j][k]*lhsX[1][m][CC][isize-1][j][k]
														- lhsX[n][2][AA][isize][j][k]*lhsX[2][m][CC][isize-1][j][k]
														- lhsX[n][3][AA][isize][j][k]*lhsX[3][m][CC][isize-1][j][k]
														- lhsX[n][4][AA][isize][j][k]*lhsX[4][m][CC][isize-1][j][k];
		}
	  }
	 */
  lhsX[0][0][BB][isize][j][k] = lhsX[0][0][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[1][0][BB][isize][j][k] = lhsX[1][0][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[2][0][BB][isize][j][k] = lhsX[2][0][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[3][0][BB][isize][j][k] = lhsX[3][0][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[4][0][BB][isize][j][k] = lhsX[4][0][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][0][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][0][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][0][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][0][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][0][CC][isize-1][j][k];
  lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[1][1][BB][isize][j][k] = lhsX[1][1][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[2][1][BB][isize][j][k] = lhsX[2][1][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[3][1][BB][isize][j][k] = lhsX[3][1][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[4][1][BB][isize][j][k] = lhsX[4][1][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][1][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][1][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][1][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][1][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][1][CC][isize-1][j][k];
  lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[2][2][BB][isize][j][k] = lhsX[2][2][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[3][2][BB][isize][j][k] = lhsX[3][2][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[4][2][BB][isize][j][k] = lhsX[4][2][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][2][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][2][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][2][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][2][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][2][CC][isize-1][j][k];
  lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[3][3][BB][isize][j][k] = lhsX[3][3][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[4][3][BB][isize][j][k] = lhsX[4][3][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][3][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][3][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][3][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][3][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][3][CC][isize-1][j][k];
  lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k] - lhsX[0][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[0][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[0][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[0][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[0][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k] - lhsX[1][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[1][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[1][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[1][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[1][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k] - lhsX[2][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[2][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[2][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[2][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[2][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k] - lhsX[3][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[3][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[3][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[3][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[3][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];
  lhsX[4][4][BB][isize][j][k] = lhsX[4][4][BB][isize][j][k] - lhsX[4][0][AA][isize][j][k]*lhsX[0][4][CC][isize-1][j][k]
                              - lhsX[4][1][AA][isize][j][k]*lhsX[1][4][CC][isize-1][j][k]
                              - lhsX[4][2][AA][isize][j][k]*lhsX[2][4][CC][isize-1][j][k]
                              - lhsX[4][3][AA][isize][j][k]*lhsX[3][4][CC][isize-1][j][k]
                              - lhsX[4][4][AA][isize][j][k]*lhsX[4][4][CC][isize-1][j][k];


    }
  }
      //---------------------------------------------------------------------
      // multiply rhs() by b_inverse() and copy to rhs
      //---------------------------------------------------------------------
      //binvrhs( lhsX[isize][j][BB], rhs[k][isize][j][k] );
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
	/*
	  for(m = 0; m < 5; m++){
	  	pivot = 1.00/lhsX[m][m][BB][isize][j][k];
		for(n = m+1; n < 5; n++){
			lhsX[m][n][BB][isize][j][k] = lhsX[m][n][BB][isize][j][k]*pivot;
		}
		rhs[k][j][isize][m] = rhs[k][j][isize][m]*pivot;
		
		for(n = 0; n < 5; n++){
			if(n != m){
				coeff = lhsX[n][m][BB][isize][j][k];
				for(z = m+1; z < 5; z++){
					lhsX[n][z][BB][isize][j][k] = lhsX[n][z][BB][isize][j][k] - coeff*lhsX[m][z][BB][isize][j][k];
				}
				rhs[k][j][isize][n] = rhs[k][j][isize][n] - coeff*rhs[k][j][isize][m];
			}
		}
	  }
	 */
  pivot = 1.00/lhsX[0][0][BB][isize][j][k];
  lhsX[0][1][BB][isize][j][k] = lhsX[0][1][BB][isize][j][k]*pivot;
  lhsX[0][2][BB][isize][j][k] = lhsX[0][2][BB][isize][j][k]*pivot;
  lhsX[0][3][BB][isize][j][k] = lhsX[0][3][BB][isize][j][k]*pivot;
  lhsX[0][4][BB][isize][j][k] = lhsX[0][4][BB][isize][j][k]*pivot;
  rhs[k][j][isize][0]   = rhs[k][j][isize][0]  *pivot;

  coeff = lhsX[1][0][BB][isize][j][k];
  lhsX[1][1][BB][isize][j][k]= lhsX[1][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[1][2][BB][isize][j][k]= lhsX[1][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][0];

  coeff = lhsX[2][0][BB][isize][j][k];
  lhsX[2][1][BB][isize][j][k]= lhsX[2][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][0];

  coeff = lhsX[3][0][BB][isize][j][k];
  lhsX[3][1][BB][isize][j][k]= lhsX[3][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][0];

  coeff = lhsX[4][0][BB][isize][j][k];
  lhsX[4][1][BB][isize][j][k]= lhsX[4][1][BB][isize][j][k] - coeff*lhsX[0][1][BB][isize][j][k];
  lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[0][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[0][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[0][4][BB][isize][j][k];
  rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][0];


  pivot = 1.00/lhsX[1][1][BB][isize][j][k];
  lhsX[1][2][BB][isize][j][k] = lhsX[1][2][BB][isize][j][k]*pivot;
  lhsX[1][3][BB][isize][j][k] = lhsX[1][3][BB][isize][j][k]*pivot;
  lhsX[1][4][BB][isize][j][k] = lhsX[1][4][BB][isize][j][k]*pivot;
  rhs[k][j][isize][1]   = rhs[k][j][isize][1]  *pivot;

  coeff = lhsX[0][1][BB][isize][j][k];
  lhsX[0][2][BB][isize][j][k]= lhsX[0][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][1];

  coeff = lhsX[2][1][BB][isize][j][k];
  lhsX[2][2][BB][isize][j][k]= lhsX[2][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k]= lhsX[2][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][1];

  coeff = lhsX[3][1][BB][isize][j][k];
  lhsX[3][2][BB][isize][j][k]= lhsX[3][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][1];

  coeff = lhsX[4][1][BB][isize][j][k];
  lhsX[4][2][BB][isize][j][k]= lhsX[4][2][BB][isize][j][k] - coeff*lhsX[1][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[1][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[1][4][BB][isize][j][k];
  rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][1];


  pivot = 1.00/lhsX[2][2][BB][isize][j][k];
  lhsX[2][3][BB][isize][j][k] = lhsX[2][3][BB][isize][j][k]*pivot;
  lhsX[2][4][BB][isize][j][k] = lhsX[2][4][BB][isize][j][k]*pivot;
  rhs[k][j][isize][2]   = rhs[k][j][isize][2]  *pivot;

  coeff = lhsX[0][2][BB][isize][j][k];
  lhsX[0][3][BB][isize][j][k]= lhsX[0][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][2];

  coeff = lhsX[1][2][BB][isize][j][k];
  lhsX[1][3][BB][isize][j][k]= lhsX[1][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][2];

  coeff = lhsX[3][2][BB][isize][j][k];
  lhsX[3][3][BB][isize][j][k]= lhsX[3][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k]= lhsX[3][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][2];

  coeff = lhsX[4][2][BB][isize][j][k];
  lhsX[4][3][BB][isize][j][k]= lhsX[4][3][BB][isize][j][k] - coeff*lhsX[2][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[2][4][BB][isize][j][k];
  rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][2];


  pivot = 1.00/lhsX[3][3][BB][isize][j][k];
  lhsX[3][4][BB][isize][j][k] = lhsX[3][4][BB][isize][j][k]*pivot;
  rhs[k][j][isize][3]   = rhs[k][j][isize][3]  *pivot;

  coeff = lhsX[0][3][BB][isize][j][k];
  lhsX[0][4][BB][isize][j][k]= lhsX[0][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][3];

  coeff = lhsX[1][3][BB][isize][j][k];
  lhsX[1][4][BB][isize][j][k]= lhsX[1][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][3];

  coeff = lhsX[2][3][BB][isize][j][k];
  lhsX[2][4][BB][isize][j][k]= lhsX[2][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][3];

  coeff = lhsX[4][3][BB][isize][j][k];
  lhsX[4][4][BB][isize][j][k]= lhsX[4][4][BB][isize][j][k] - coeff*lhsX[3][4][BB][isize][j][k];
  rhs[k][j][isize][4]   = rhs[k][j][isize][4]   - coeff*rhs[k][j][isize][3];


  pivot = 1.00/lhsX[4][4][BB][isize][j][k];
  rhs[k][j][isize][4]   = rhs[k][j][isize][4]  *pivot;

  coeff = lhsX[0][4][BB][isize][j][k];
  rhs[k][j][isize][0]   = rhs[k][j][isize][0]   - coeff*rhs[k][j][isize][4];

  coeff = lhsX[1][4][BB][isize][j][k];
  rhs[k][j][isize][1]   = rhs[k][j][isize][1]   - coeff*rhs[k][j][isize][4];

  coeff = lhsX[2][4][BB][isize][j][k];
  rhs[k][j][isize][2]   = rhs[k][j][isize][2]   - coeff*rhs[k][j][isize][4];

  coeff = lhsX[3][4][BB][isize][j][k];
  rhs[k][j][isize][3]   = rhs[k][j][isize][3]   - coeff*rhs[k][j][isize][4];


	}
  }

      //---------------------------------------------------------------------
      // back solve: if last cell, then generate U(isize)=rhs(isize)
      // else assume U(isize) is loaded in un pack backsub_info
      // so just use it
      // after u(istart) will be sent to next cell
      //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = isize-1; i >=0; i--) {
        for (m = 0; m < BLOCK_SIZE; m++) {
          for (n = 0; n < BLOCK_SIZE; n++) {
            rhs[k][j][i][m] = rhs[k][j][i][m] 
              - lhsX[m][n][CC][i][j][k]*rhs[k][j][i+1][n];
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
  acc_delete_unified(fjacX, 0);
  acc_delete_unified(njacX, 0);
  acc_delete_unified(lhsX, 0);
#else
  free(fjacX);
  free(njacX);
  free(lhsX);
#endif
}

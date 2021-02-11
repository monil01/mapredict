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
//#include "timers.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;
  int gp22, gp12, gp02;

  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

#ifdef USE_ASYNC
#pragma acc kernels loop independent present(u,rhs) async(0) //pcopy(rhs)
#else
#pragma acc kernels loop independent present(u,rhs) //pcopy(rhs)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
	  /*
        for (m = 0; m < 5; m++) {
          u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
        }
	  */
          u[k][j][i][0] = u[k][j][i][0] + rhs[k][j][i][0];
          u[k][j][i][1] = u[k][j][i][1] + rhs[k][j][i][1];
          u[k][j][i][2] = u[k][j][i][2] + rhs[k][j][i][2];
          u[k][j][i][3] = u[k][j][i][3] + rhs[k][j][i][3];
          u[k][j][i][4] = u[k][j][i][4] + rhs[k][j][i][4];
      }
    }
  }
#ifdef USE_ASYNC
#pragma acc wait(0)
#endif
}

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

void compute_rhs()
{
  int i, j, k, m;
  double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;
  int gp01,gp11,gp21;
  int gp02,gp12,gp22;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];
  gp01 = grid_points[0]-1;
  gp11 = grid_points[1]-1;
  gp21 = grid_points[2]-1;
  gp02 = grid_points[0]-2;
  gp12 = grid_points[1]-2;
  gp22 = grid_points[2]-2;

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound.
  //---------------------------------------------------------------------
#pragma acc data present(forcing,rho_i,u,us,vs,ws,square,qs,rhs) //pcopy(rhs) present(rho_i,u,forcing)
{
#ifdef USE_ASYNC
 #pragma acc kernels loop independent collapse(3) async(0)
#else
 #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 0; k <= gp21; k++) {
    for (j = 0; j <= gp11; j++) {
      for (i = 0; i <= gp01; i++) {
        rho_inv = 1.0/u[k][j][i][0];
        rho_i[k][j][i] = rho_inv;
        us[k][j][i] = u[k][j][i][1] * rho_inv;
        vs[k][j][i] = u[k][j][i][2] * rho_inv;
        ws[k][j][i] = u[k][j][i][3] * rho_inv;
        square[k][j][i] = 0.5* (
            u[k][j][i][1]*u[k][j][i][1] + 
            u[k][j][i][2]*u[k][j][i][2] +
            u[k][j][i][3]*u[k][j][i][3] ) * rho_inv;
        qs[k][j][i] = square[k][j][i] * rho_inv;
      }
    }
  }

  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 0; k <= gp21; k++) {
    for (j = 0; j <= gp11; j++) {
      for (i = 0; i <= gp01; i++) {
          rhs[k][j][i][0] = forcing[k][j][i][0];
          rhs[k][j][i][1] = forcing[k][j][i][1];
          rhs[k][j][i][2] = forcing[k][j][i][2];
          rhs[k][j][i][3] = forcing[k][j][i][3];
          rhs[k][j][i][4] = forcing[k][j][i][4];
      }
    }
  }

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent async(0)
#else
  #pragma acc kernels loop independent
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[k][j][i][0] = rhs[k][j][i][0] + dx1tx1 * 
          (u[k][j][i+1][0] - 2.0*u[k][j][i][0] + 
           u[k][j][i-1][0]) -
          tx2 * (u[k][j][i+1][1] - u[k][j][i-1][1]);

        rhs[k][j][i][1] = rhs[k][j][i][1] + dx2tx1 * 
          (u[k][j][i+1][1] - 2.0*u[k][j][i][1] + 
           u[k][j][i-1][1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[k][j][i+1][1]*up1 - 
              u[k][j][i-1][1]*um1 +
              (u[k][j][i+1][4]- square[k][j][i+1]-
               u[k][j][i-1][4]+ square[k][j][i-1])*
              c2);

        rhs[k][j][i][2] = rhs[k][j][i][2] + dx3tx1 * 
          (u[k][j][i+1][2] - 2.0*u[k][j][i][2] +
           u[k][j][i-1][2]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
              vs[k][j][i-1]) -
          tx2 * (u[k][j][i+1][2]*up1 - 
              u[k][j][i-1][2]*um1);

        rhs[k][j][i][3] = rhs[k][j][i][3] + dx4tx1 * 
          (u[k][j][i+1][3] - 2.0*u[k][j][i][3] +
           u[k][j][i-1][3]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
              ws[k][j][i-1]) -
          tx2 * (u[k][j][i+1][3]*up1 - 
              u[k][j][i-1][3]*um1);

        rhs[k][j][i][4] = rhs[k][j][i][4] + dx5tx1 * 
          (u[k][j][i+1][4] - 2.0*u[k][j][i][4] +
           u[k][j][i-1][4]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
              qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + 
              um1*um1) +
          xxcon5 * (u[k][j][i+1][4]*rho_i[k][j][i+1] - 
              2.0*u[k][j][i][4]*rho_i[k][j][i] +
              u[k][j][i-1][4]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[k][j][i+1][4] - 
                c2*square[k][j][i+1])*up1 -
              (c1*u[k][j][i-1][4] - 
               c2*square[k][j][i-1])*um1 );
      }
    }

    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
	#pragma acc loop independent
    for (j = 1; j <= gp12; j++) {
      i = 1;
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp * 
          ( 5.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] +
            u[k][j][i+2][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp * 
          ( 5.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] +
            u[k][j][i+2][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp * 
          ( 5.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] +
            u[k][j][i+2][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp * 
          ( 5.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] +
            u[k][j][i+2][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp * 
          ( 5.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] +
            u[k][j][i+2][4]);

      i = 2;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp * 
          (-4.0*u[k][j][i-1][0] + 6.0*u[k][j][i][0] -
           4.0*u[k][j][i+1][0] + u[k][j][i+2][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp * 
          (-4.0*u[k][j][i-1][1] + 6.0*u[k][j][i][1] -
           4.0*u[k][j][i+1][1] + u[k][j][i+2][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp * 
          (-4.0*u[k][j][i-1][2] + 6.0*u[k][j][i][2] -
           4.0*u[k][j][i+1][2] + u[k][j][i+2][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp * 
          (-4.0*u[k][j][i-1][3] + 6.0*u[k][j][i][3] -
           4.0*u[k][j][i+1][3] + u[k][j][i+2][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp * 
          (-4.0*u[k][j][i-1][4] + 6.0*u[k][j][i][4] -
           4.0*u[k][j][i+1][4] + u[k][j][i+2][4]);
    }

    #pragma acc loop independent collapse(2)
    for (j = 1; j <= gp12; j++) {
      for (i = 3; i <= gp02-2; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp*
		    (  u[k][j][i-2][0] - 4.0*u[k][j][i-1][0] + 
               6.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] + 
               u[k][j][i+2][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp*
		    (  u[k][j][i-2][1] - 4.0*u[k][j][i-1][1] + 
               6.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] + 
               u[k][j][i+2][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp*
		    (  u[k][j][i-2][2] - 4.0*u[k][j][i-1][2] + 
               6.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] + 
               u[k][j][i+2][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp*
		    (  u[k][j][i-2][3] - 4.0*u[k][j][i-1][3] + 
               6.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] + 
               u[k][j][i+2][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp*
		    (  u[k][j][i-2][4] - 4.0*u[k][j][i-1][4] + 
               6.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] + 
               u[k][j][i+2][4] );
      }
    }

    #pragma acc loop  independent
    for (j = 1; j <= gp12; j++) {
      i = gp0-3;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j][i-2][0] - 4.0*u[k][j][i-1][0] + 
            6.0*u[k][j][i][0] - 4.0*u[k][j][i+1][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j][i-2][1] - 4.0*u[k][j][i-1][1] + 
            6.0*u[k][j][i][1] - 4.0*u[k][j][i+1][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j][i-2][2] - 4.0*u[k][j][i-1][2] + 
            6.0*u[k][j][i][2] - 4.0*u[k][j][i+1][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j][i-2][3] - 4.0*u[k][j][i-1][3] + 
            6.0*u[k][j][i][3] - 4.0*u[k][j][i+1][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j][i-2][4] - 4.0*u[k][j][i-1][4] + 
            6.0*u[k][j][i][4] - 4.0*u[k][j][i+1][4] );

      i = gp02;
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j][i-2][0] - 4.*u[k][j][i-1][0] +
            5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j][i-2][1] - 4.*u[k][j][i-1][1] +
            5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j][i-2][2] - 4.*u[k][j][i-1][2] +
            5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j][i-2][3] - 4.*u[k][j][i-1][3] +
            5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j][i-2][4] - 4.*u[k][j][i-1][4] +
            5.*u[k][j][i][4] );
    }
  }

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent async(0)
#else
  #pragma acc kernels loop independent
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];
        rhs[k][j][i][0] = rhs[k][j][i][0] + dy1ty1 * 
          (u[k][j+1][i][0] - 2.0*u[k][j][i][0] + 
           u[k][j-1][i][0]) -
          ty2 * (u[k][j+1][i][2] - u[k][j-1][i][2]);
        rhs[k][j][i][1] = rhs[k][j][i][1] + dy2ty1 * 
          (u[k][j+1][i][1] - 2.0*u[k][j][i][1] + 
           u[k][j-1][i][1]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + 
              us[k][j-1][i]) -
          ty2 * (u[k][j+1][i][1]*vp1 - 
              u[k][j-1][i][1]*vm1);
        rhs[k][j][i][2] = rhs[k][j][i][2] + dy3ty1 * 
          (u[k][j+1][i][2] - 2.0*u[k][j][i][2] + 
           u[k][j-1][i][2]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[k][j+1][i][2]*vp1 - 
              u[k][j-1][i][2]*vm1 +
              (u[k][j+1][i][4] - square[k][j+1][i] - 
               u[k][j-1][i][4] + square[k][j-1][i])
              *c2);
        rhs[k][j][i][3] = rhs[k][j][i][3] + dy4ty1 * 
          (u[k][j+1][i][3] - 2.0*u[k][j][i][3] + 
           u[k][j-1][i][3]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + 
              ws[k][j-1][i]) -
          ty2 * (u[k][j+1][i][3]*vp1 - 
              u[k][j-1][i][3]*vm1);
        rhs[k][j][i][4] = rhs[k][j][i][4] + dy5ty1 * 
          (u[k][j+1][i][4] - 2.0*u[k][j][i][4] + 
           u[k][j-1][i][4]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + 
              qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + 
              vm1*vm1) +
          yycon5 * (u[k][j+1][i][4]*rho_i[k][j+1][i] - 
              2.0*u[k][j][i][4]*rho_i[k][j][i] +
              u[k][j-1][i][4]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[k][j+1][i][4] - 
                c2*square[k][j+1][i]) * vp1 -
              (c1*u[k][j-1][i][4] - 
               c2*square[k][j-1][i]) * vm1);
      }
    }

    //---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
    j = 1;
	#pragma acc loop independent
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp * 
          ( 5.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] +
            u[k][j+2][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp * 
          ( 5.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] +
            u[k][j+2][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp * 
          ( 5.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] +
            u[k][j+2][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp * 
          ( 5.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] +
            u[k][j+2][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp * 
          ( 5.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] +
            u[k][j+2][i][4]);
    }

    j = 2;
	#pragma acc loop  independent
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp * 
          (-4.0*u[k][j-1][i][0] + 6.0*u[k][j][i][0] -
           4.0*u[k][j+1][i][0] + u[k][j+2][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp * 
          (-4.0*u[k][j-1][i][1] + 6.0*u[k][j][i][1] -
           4.0*u[k][j+1][i][1] + u[k][j+2][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp * 
          (-4.0*u[k][j-1][i][2] + 6.0*u[k][j][i][2] -
           4.0*u[k][j+1][i][2] + u[k][j+2][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp * 
          (-4.0*u[k][j-1][i][3] + 6.0*u[k][j][i][3] -
           4.0*u[k][j+1][i][3] + u[k][j+2][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp * 
          (-4.0*u[k][j-1][i][4] + 6.0*u[k][j][i][4] -
           4.0*u[k][j+1][i][4] + u[k][j+2][i][4]);
    }

    #pragma acc loop independent collapse(2)
    for (j = 3; j <= gp1-4; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp * 
            (  u[k][j-2][i][0] - 4.0*u[k][j-1][i][0] + 
               6.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] + 
               u[k][j+2][i][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp * 
            (  u[k][j-2][i][1] - 4.0*u[k][j-1][i][1] + 
               6.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] + 
               u[k][j+2][i][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp * 
            (  u[k][j-2][i][2] - 4.0*u[k][j-1][i][2] + 
               6.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] + 
               u[k][j+2][i][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp * 
            (  u[k][j-2][i][3] - 4.0*u[k][j-1][i][3] + 
               6.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] + 
               u[k][j+2][i][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp * 
            (  u[k][j-2][i][4] - 4.0*u[k][j-1][i][4] + 
               6.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] + 
               u[k][j+2][i][4] );
      }
    }

    j = gp1-3;
	#pragma acc loop independent
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j-2][i][0] - 4.0*u[k][j-1][i][0] + 
            6.0*u[k][j][i][0] - 4.0*u[k][j+1][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j-2][i][1] - 4.0*u[k][j-1][i][1] + 
            6.0*u[k][j][i][1] - 4.0*u[k][j+1][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j-2][i][2] - 4.0*u[k][j-1][i][2] + 
            6.0*u[k][j][i][2] - 4.0*u[k][j+1][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j-2][i][3] - 4.0*u[k][j-1][i][3] + 
            6.0*u[k][j][i][3] - 4.0*u[k][j+1][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j-2][i][4] - 4.0*u[k][j-1][i][4] + 
            6.0*u[k][j][i][4] - 4.0*u[k][j+1][i][4] );
    }

    j = gp12;
	#pragma acc loop independent
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k][j-2][i][0] - 4.*u[k][j-1][i][0] +
            5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k][j-2][i][1] - 4.*u[k][j-1][i][1] +
            5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k][j-2][i][2] - 4.*u[k][j-1][i][2] +
            5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k][j-2][i][3] - 4.*u[k][j-1][i][3] +
            5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k][j-2][i][4] - 4.*u[k][j-1][i][4] +
            5.*u[k][j][i][4] );
    }
  }
  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[k][j][i][0] = rhs[k][j][i][0] + dz1tz1 * 
          (u[k+1][j][i][0] - 2.0*u[k][j][i][0] + 
           u[k-1][j][i][0]) -
          tz2 * (u[k+1][j][i][3] - u[k-1][j][i][3]);
        rhs[k][j][i][1] = rhs[k][j][i][1] + dz2tz1 * 
          (u[k+1][j][i][1] - 2.0*u[k][j][i][1] + 
           u[k-1][j][i][1]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + 
              us[k-1][j][i]) -
          tz2 * (u[k+1][j][i][1]*wp1 - 
              u[k-1][j][i][1]*wm1);
        rhs[k][j][i][2] = rhs[k][j][i][2] + dz3tz1 * 
          (u[k+1][j][i][2] - 2.0*u[k][j][i][2] + 
           u[k-1][j][i][2]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + 
              vs[k-1][j][i]) -
          tz2 * (u[k+1][j][i][2]*wp1 - 
              u[k-1][j][i][2]*wm1);
        rhs[k][j][i][3] = rhs[k][j][i][3] + dz4tz1 * 
          (u[k+1][j][i][3] - 2.0*u[k][j][i][3] + 
           u[k-1][j][i][3]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[k+1][j][i][3]*wp1 - 
              u[k-1][j][i][3]*wm1 +
              (u[k+1][j][i][4] - square[k+1][j][i] - 
               u[k-1][j][i][4] + square[k-1][j][i])
              *c2);
        rhs[k][j][i][4] = rhs[k][j][i][4] + dz5tz1 * 
          (u[k+1][j][i][4] - 2.0*u[k][j][i][4] + 
           u[k-1][j][i][4]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + 
              qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + 
              wm1*wm1) +
          zzcon5 * (u[k+1][j][i][4]*rho_i[k+1][j][i] - 
              2.0*u[k][j][i][4]*rho_i[k][j][i] +
              u[k-1][j][i][4]*rho_i[k-1][j][i]) -
          tz2 * ( (c1*u[k+1][j][i][4] - 
                c2*square[k+1][j][i])*wp1 -
              (c1*u[k-1][j][i][4] - 
               c2*square[k-1][j][i])*wm1);
      }
    }
  }
  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0]- dssp * 
          ( 5.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] +
            u[k+2][j][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1]- dssp * 
          ( 5.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] +
            u[k+2][j][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2]- dssp * 
          ( 5.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] +
            u[k+2][j][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3]- dssp * 
          ( 5.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] +
            u[k+2][j][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4]- dssp * 
          ( 5.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] +
            u[k+2][j][i][4]);
    }
  }

  k = 2;
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++){ 
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp * 
          (-4.0*u[k-1][j][i][0] + 6.0*u[k][j][i][0] -
           4.0*u[k+1][j][i][0] + u[k+2][j][i][0]);
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp * 
          (-4.0*u[k-1][j][i][1] + 6.0*u[k][j][i][1] -
           4.0*u[k+1][j][i][1] + u[k+2][j][i][1]);
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp * 
          (-4.0*u[k-1][j][i][2] + 6.0*u[k][j][i][2] -
           4.0*u[k+1][j][i][2] + u[k+2][j][i][2]);
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp * 
          (-4.0*u[k-1][j][i][3] + 6.0*u[k][j][i][3] -
           4.0*u[k+1][j][i][3] + u[k+2][j][i][3]);
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp * 
          (-4.0*u[k-1][j][i][4] + 6.0*u[k][j][i][4] -
           4.0*u[k+1][j][i][4] + u[k+2][j][i][4]);
    }
  }

#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 3; k <= gp2-4; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] - dssp * 
            (  u[k-2][j][i][0] - 4.0*u[k-1][j][i][0] + 
               6.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] + 
               u[k+2][j][i][0] );
          rhs[k][j][i][1] = rhs[k][j][i][1] - dssp * 
            (  u[k-2][j][i][1] - 4.0*u[k-1][j][i][1] + 
               6.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] + 
               u[k+2][j][i][1] );
          rhs[k][j][i][2] = rhs[k][j][i][2] - dssp * 
            (  u[k-2][j][i][2] - 4.0*u[k-1][j][i][2] + 
               6.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] + 
               u[k+2][j][i][2] );
          rhs[k][j][i][3] = rhs[k][j][i][3] - dssp * 
            (  u[k-2][j][i][3] - 4.0*u[k-1][j][i][3] + 
               6.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] + 
               u[k+2][j][i][3] );
          rhs[k][j][i][4] = rhs[k][j][i][4] - dssp * 
            (  u[k-2][j][i][4] - 4.0*u[k-1][j][i][4] + 
               6.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] + 
               u[k+2][j][i][4] );
      }
    }
  }

  k = gp2-3;
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k-2][j][i][0] - 4.0*u[k-1][j][i][0] + 
            6.0*u[k][j][i][0] - 4.0*u[k+1][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k-2][j][i][1] - 4.0*u[k-1][j][i][1] + 
            6.0*u[k][j][i][1] - 4.0*u[k+1][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k-2][j][i][2] - 4.0*u[k-1][j][i][2] + 
            6.0*u[k][j][i][2] - 4.0*u[k+1][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k-2][j][i][3] - 4.0*u[k-1][j][i][3] + 
            6.0*u[k][j][i][3] - 4.0*u[k+1][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k-2][j][i][4] - 4.0*u[k-1][j][i][4] + 
            6.0*u[k][j][i][4] - 4.0*u[k+1][j][i][4] );
    }
  }

  k = gp22;
#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(2) async(0)
#else
  #pragma acc kernels loop independent collapse(2)
#endif
  for (j = 1; j <= gp12; j++) {
    for (i = 1; i <= gp02; i++) {
        rhs[k][j][i][0] = rhs[k][j][i][0] - dssp *
          ( u[k-2][j][i][0] - 4.*u[k-1][j][i][0] +
            5.*u[k][j][i][0] );
        rhs[k][j][i][1] = rhs[k][j][i][1] - dssp *
          ( u[k-2][j][i][1] - 4.*u[k-1][j][i][1] +
            5.*u[k][j][i][1] );
        rhs[k][j][i][2] = rhs[k][j][i][2] - dssp *
          ( u[k-2][j][i][2] - 4.*u[k-1][j][i][2] +
            5.*u[k][j][i][2] );
        rhs[k][j][i][3] = rhs[k][j][i][3] - dssp *
          ( u[k-2][j][i][3] - 4.*u[k-1][j][i][3] +
            5.*u[k][j][i][3] );
        rhs[k][j][i][4] = rhs[k][j][i][4] - dssp *
          ( u[k-2][j][i][4] - 4.*u[k-1][j][i][4] +
            5.*u[k][j][i][4] );
    }
  }

#ifdef USE_ASYNC
  #pragma acc kernels loop independent collapse(3) async(0)
#else
  #pragma acc kernels loop independent collapse(3)
#endif
  for (k = 1; k <= gp22; k++) {
    for (j = 1; j <= gp12; j++) {
      for (i = 1; i <= gp02; i++) {
          rhs[k][j][i][0] = rhs[k][j][i][0] * dt;
          rhs[k][j][i][1] = rhs[k][j][i][1] * dt;
          rhs[k][j][i][2] = rhs[k][j][i][2] * dt;
          rhs[k][j][i][3] = rhs[k][j][i][3] * dt;
          rhs[k][j][i][4] = rhs[k][j][i][4] * dt;
      }
    }
  }
#ifdef USE_ASYNC
#pragma acc wait(0)
#endif

}/*end acc data*/
}

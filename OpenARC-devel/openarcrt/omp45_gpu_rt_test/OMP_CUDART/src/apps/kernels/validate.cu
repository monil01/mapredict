#include "pragmatic.h"
#include <stdio.h>

static int check_is_failed(const int index, double value, double expected)
{
  double m = max(fabs(value), fabs(expected));
  m = (m == 0.0) ? 0.0 : fabs(value - expected) / m;

  if(m > TOLERANCE)
  {
    printf("\x1B[31mFailed\x1B[0m at %d exp %.12e act %.12e\n", index, expected, value);
    return 1;
  }
  return 0;
}

void validate_vec_add_2d(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = (double)ii + 1.0;
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_vec_add_2d_elemental(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = (double)ii + 1.0;
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_vec_add_reverse_indirection(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = (double)ii + 1.0;
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_vec_add_column_indirection(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = (double)ii + 1.0;
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_two_pt_stencil(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y - 1; ++ii)
  {
    const double expected = (ii + ii + 1.0);
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_two_pt_stencil_dist_10(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y - 10; ++ii)
  {
    const double expected = (ii + ii + 10.0);
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_vec_add_sqrt(const int x, const int y, double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = ii + 1.0;
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_vec_add_and_mul(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = (ii + ii);
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_compute_bound(
    const int x, const int y, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  if(x != X || y != Y)
  {
    printf("\t*Cannot validate this problem size.");
    return;
  }

  for(int ii = 0; ii < x*y; ++ii)
  {
    if(check_is_failed(ii, a[ii], COMPUTE_BOUND_VALIDATE))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_five_pt_stencil_2d(
    const int x, const int y, const double delta, double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for(int ii = 1; ii < y - 1; ++ii)
  {
    for(int jj = 1; jj < x - 1; ++jj)
    {
      const int index = ii*x + jj;
      const double expected = (double)(4.0 - delta)*index;
      if(check_is_failed(index, a[index], expected))
      {
        printf("\t*\x1B[31mFailed\x1B[0m validation\n");
        return;
      }
    }
  }
  printf("\t*Passed validation\n");
}

void validate_nine_pt_stencil_2d(
    const int x, const int y, const double alpha, const double beta, 
    const double delta, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 1; ii < y - 1; ++ii)
  {
    for(int jj = 1; jj < x - 1; ++jj)
    {
      const int index = ii*x + jj;
      const double expected = (4*alpha + 4*beta - delta)*index;
      if(check_is_failed(index, a[index], expected))
      {
        printf("\t*\x1B[31mFailed\x1B[0m validation\n");
        return;
      }
    }
  }
  printf("\t*Passed validation\n");
}

void validate_seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, const double beta, 
    double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y*z);
  cudaMemcpy(a, d_a, sizeof(double)*x*y*z, cudaMemcpyDeviceToHost);

  for(int ii = 1; ii < z - 1; ++ii)
  {
    for(int jj = 1; jj < y - 1; ++jj)
    {
      for(int kk = 1; kk < x - 1; ++kk)
      {
        const int index = ii*x*y + jj*x + kk;
        const double expected = (6*beta - alpha)*index;
        if(check_is_failed(index, a[index], expected))
        {
          printf("\t*\x1B[31mFailed\x1B[0m validation\n");
          return;
        }
      }
    }
  }
  printf("\t*Passed validation\n");
}

void validate_twenty_seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, const double beta,
    const double delta, const double gamma, const double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y*z);
  cudaMemcpy(a, d_a, sizeof(double)*x*y*z, cudaMemcpyDeviceToHost);

  for(int ii = 1; ii < z - 1; ++ii)
  {
    for(int jj = 1; jj < y - 1; ++jj)
    {
      for(int kk = 1; kk < x - 1; ++kk)
      {
        const int index = ii*x*y + jj*x + kk;
        const double expected = (8*delta + 12*gamma + 6*beta - alpha)*index;
        if(check_is_failed(index, a[index], expected))
        {
          printf("\t*\x1B[31mFailed\x1B[0m validation\n");
          return;
        }
      }
    }
  }
  printf("\t*Passed validation\n");
}

void validate_five_pt_wavefront(
    const int x, const int y, const int z, double* d_a)
{
  double* a = (double*)malloc(sizeof(double)*x*y*z);
  cudaMemcpy(a, d_a, sizeof(double)*x*y*z, cudaMemcpyDeviceToHost);

  for(int ii = 1; ii < z - 1; ++ii)
  {
    for(int jj = 1; jj < y - 1; ++jj)
    {
      const double expected = (double)(ii*y + jj);
      for(int dd = 0; dd < x; ++dd)
      {
        const int index = ii*x*y + jj*x + dd;
        if(check_is_failed(index, a[index], expected))
        {
          printf("\t*\x1B[31mFailed\x1B[0m validation\n");
          return;
        }
      }
    }
  }
  printf("\t*Passed validation\n");
}

void validate_even_odd_divergence(
    const int x, const int y, const double* d_a, const double* d_d)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  double* d = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);
  cudaMemcpy(d, d_d, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = 2.0*ii;
    const double actual = a[ii] + d[ii];
    if(check_is_failed(ii, actual, expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_tealeaf_cheby_iter(
    const int x, const int y, const int halo_depth, const double* d_g)
{
  double* g = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(g, d_g, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  if(x != X || y != Y)
  {
    printf("\t*Cannot validate when non-default problem size.");
    return;
  }

  double gval = 0.0;
  for(int ii = halo_depth; ii < y-halo_depth; ++ii)
  {
    for(int jj = halo_depth; jj < x-halo_depth; ++jj)
    {	
      const int index = ii*x + jj;
      gval += g[index];
    }
  }

  if(check_is_failed(0, gval, TEALEAF_VALIDATE))
    printf("\t*\x1B[31mFailed\x1B[0m validation\n");
  else
    printf("\t*Passed validation\n");
}

void validate_cloverleaf_energy_flux(
    const int x, const int y, const int halo_depth, const double* d_f)
{
  double* f = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(f, d_f, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  double fval = 0.0;
  for(int ii = halo_depth; ii < y; ++ii)
  {
    for(int jj = halo_depth; jj < x-halo_depth; ++jj)
    {	
      const int index = ii*x + jj;
      fval += f[index];
    }
  }

  if(check_is_failed(0, fval, CLOVERLEAF_VALIDATE))
    printf("\t*\x1B[31mFailed\x1B[0m validation\n");
  else
    printf("\t*Passed validation\n");
}

void validate_dense_mat_vec(
    const int x, const int y, const double* d_f)
{
  double* f = (double*)malloc(sizeof(double)*y);
  cudaMemcpy(f, d_f, sizeof(double)*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < y; ++ii)
  {
    const double expected = 5*x*0.1;
    if(check_is_failed(ii, f[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}

void validate_snap_sweep(
    const int x, const int y, const double* d_g)
{
  double* g = (double*)malloc(sizeof(double)*G_LEN);
  cudaMemcpy(g, d_g, sizeof(double)*G_LEN, cudaMemcpyDeviceToHost);

  /// Requires some validation procedure
  double gval = 0.0;

  for(int ii = 1; ii < y-1; ++ii)
    for(int jj = 1; jj < x-1; ++jj)
      for(int kk = 0; kk < NGROUPS; ++kk)
        for(int ll = 0; ll < NANG; ++ll)
          gval += g[ii*NANG*NGROUPS*x + jj*NGROUPS*NANG + kk*NANG + ll];

  if(check_is_failed(0, gval, SNAP_VALIDATE))
    printf("\t*\x1B[31mFailed\x1B[0m validation\n");
  else
    printf("\t*Passed validation\n");
}

void validate_matrix_multiply(
    const int x, const int y, double* d_a, const double* d_b)
{
  double* a = (double*)malloc(sizeof(double)*x*y);
  double* b = (double*)malloc(sizeof(double)*x*y);
  cudaMemcpy(a, d_a, sizeof(double)*x*y, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, d_b, sizeof(double)*x*y, cudaMemcpyDeviceToHost);

  for(int ii = 0; ii < x*y; ++ii)
  {
    const double expected = b[ii];
    if(check_is_failed(ii, a[ii], expected))
    {
      printf("\t*\x1B[31mFailed\x1B[0m validation\n");
      return;
    }
  }
  printf("\t*Passed validation\n");
}


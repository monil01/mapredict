#include "pragmatic.h"

/// Initialise the 2d vector addition
__global__ void init_vec_add_2d(
    const int x, const int y, double* a, double* b, double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = 1.0;
  c[gid] = (double)gid;
}

/// Initialise the 2d vector addition using elemental function
__global__ void init_vec_add_2d_elemental(
    const int x, const int y, double* a, double* b, double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = 1.0;
  c[gid] = (double)gid;
}

__global__ void init_vec_add_reverse_indirection(
    const int x, const int y, double* a, double* b, double* c, int* r_indirection)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = 1.0;
  c[gid] = (double)gid;
}

__global__ void init_reverse_indirection(
    const int x, int* r_indirection)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;

  if(gid >= x)
    return;

  r_indirection[gid] = ((x-1)-gid);
}

__global__ void init_vec_add_column_indirection(
    const int x, const int y, double* a, double* b, double* c, int* c_indirection)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = 1.0;
  c[gid] = (double)gid;
  if(gid < y)
    c_indirection[gid] = x;
}

__global__ void init_two_pt_stencil(
    const int x, const int y, double* a, double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_two_pt_stencil_dist_10(
    const int x, const int y, double* a, double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_vec_add_sqrt(
    const int x, const int y, double* a, double* b, double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
  c[gid] = 1.0;
}

__global__ void init_vec_add_and_mul(
    const int x, const int y, double* a, double* b, double* c, double* d)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
  c[gid] = 1.0;
  d[gid] = (double)gid;
}

__global__ void init_five_pt_stencil_2d(
    const int x, const int y, double* a, double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_compute_bound(
    const int x, const int y, double* a)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.1;
}

__global__ void init_nine_pt_stencil_2d(
    const int x, const int y, double* a, double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_seven_pt_stencil_3d(
    const int x, const int y, const int z, double* a, double* b) {
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y*z)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_twenty_seven_pt_stencil_3d(
    const int x, const int y, const int z, double* a, double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y*z)
    return;
  a[gid] = 0.0;
  b[gid] = (double)gid;
}

__global__ void init_five_pt_wavefront(
    const int x, const int y, const int z, double* a)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;

  const int ii = gid / x;
  const int jj = gid % x;

  for(int dd = 0; dd < x; ++dd)
    a[ii*x*y + jj*x + dd] = (double)(ii*y + jj);
}

__global__ void init_tealeaf_cheby_iter(
    const int x, const int y, double* a, double* b, double* c, double* d,
    double* e, double* f, double* g)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;
  a[gid] = 0.1;
  b[gid] = 0.1;
  c[gid] = 0.1;
  d[gid] = 0.1;
  e[gid] = 0.1;
  f[gid] = 0.1;
  g[gid] = 0.1;
}

__global__ void init_cloverleaf_energy_flux(
    const int x, const int y, double* a, double* b, double* c, double* d,
    double* e, double* f, double* a_s, double* b_s, double* c_s)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;

  if(gid < y)
  {
    a_s[gid] = 0.1;
    b_s[gid] = 0.1;
    c_s[gid] = 0.1;
  }

  a[gid] = 0.1;
  b[gid] = 0.1;
  c[gid] = 0.1;
  d[gid] = 0.1;
  e[gid] = 0.1;
  f[gid] = 0.1;
}

__global__ void init_snap_sweep(
    const int nplanes, const int x, const int y, double* a, 
    double* b, double* c, double* d, double* e, double* f, double* g, 
    double* a_s, double* b_s, double* c_s)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;

  if(gid < A_LEN)
    a[gid] = 0.1;
  if(gid < B_LEN)
    b[gid] = 0.1;
  if(gid < C_LEN)
    c[gid] = 0.1;
  if(gid < D_LEN)
    d[gid] = 0.1;
  if(gid < E_LEN)
    e[gid] = 0.1;
  if(gid < F_LEN)
    f[gid] = 0.1;
  if(gid < G_LEN)
    g[gid] = 0.1;

  if(gid < x)
  {
    a_s[gid] = 0.1;
    b_s[gid] = 0.1;
    c_s[gid] = 0.1;
  }
}

__global__ void init_dense_mat_vec(
    const int x, const int y, double* a, double* f, double* g)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid < 5*x*y)
    a[gid] = 0.1;
  if(gid < y)
    f[gid] = 0.0;
  if(gid < 5*x)
    g[gid] = 1.0;
}

__global__ void init_matrix_multiply(
    const int x, const int y, double* a, double* b, double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;

  const int ii = gid / x;
  const int jj = gid % x;

  a[gid] = 0.0;

  /// Generate identity matrix
  b[gid] = (ii == jj) ? 1.0 : 0.0;

  c[ii] = (double)gid;
}


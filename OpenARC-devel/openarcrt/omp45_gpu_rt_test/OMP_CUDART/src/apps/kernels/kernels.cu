#include "pragmatic.h"

__global__ void warmup(
    const int buffer_length, const double value, double* a, 
    double* b, double* c, double* d, double* e, double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= buffer_length) 
    return;

  a[gid] = value;
  b[gid] = value;
  c[gid] = value;
  d[gid] = value;
  e[gid] = value;
  f[gid] = value;
}

__global__ void vec_add_2d(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] + c[gid];
}

__global__ void vec_add_2d_work(
    const int x, const int y, double* a, const double* b, const double* c)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    a[gid] = b[gid] + c[gid];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void dense_mat_vec(
    const int x, const int y, const double*  a_l, 
    double*  f, const double*  g)
{
  const int ii = blockIdx.x*blockDim.x + threadIdx.x;
  if(ii >= y) 
    return;

  // Width of the matrix increased for larger data load
#pragma unroll 5
  for(int jj = 0; jj < 5*x; ++jj)
  {
    f[ii] += a_l[ii*(5*x) + jj]*g[jj];
  }
}

__global__ void dense_mat_vec_work(
    const int x, const int y, const double*  a_l, 
    double*  f, const double*  g)
{
  int ii = blockIdx.x*blockDim.x + threadIdx.x;

  while(ii < y) {
    // Width of the matrix increased for larger data load
#pragma unroll 5
    for(int jj = 0; jj < 5*x; ++jj)
    {
      f[ii] += a_l[ii*(5*x) + jj]*g[jj];
    }
    ii += gridDim.x*blockDim.x;
  }
}

__global__ void vec_add_reverse_indirect(
    const int x, const int y, int* indirection, double* a, 
    const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  // Artificially introduce second dimension
  const int ii = gid / x;
  const int jj = gid % x;
  const int index = ii*x + indirection[jj];
  a[index] = b[index] + c[index];
}

__global__ void vec_add_reverse_indirect_work(
    const int x, const int y, int* indirection, double* a, 
    const double* b, const double* c)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    // Artificially introduce second dimension
    const int ii = gid / x;
    const int jj = gid % x;
    const int index = ii*x + indirection[jj];
    a[index] = b[index] + c[index];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void vec_add_column_indirect(
    const int x, const int y, const int*  indirection, 
    double*  a, const double*  b, const double*  c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(gid >= x*y || jj >= indirection[jj]) 
    return;

  const int index = ii*x + jj;
  a[index] = b[index] + c[index];
}

__global__ void vec_add_column_indirect_work(
    const int x, const int y, const int*  indirection, 
    double*  a, const double*  b, const double*  c)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    const int ii = gid / x;
    const int jj = gid % x;
    const int index = ii*x + jj;

    if(jj < indirection[jj]) 
      a[index] = b[index] + c[index];

    gid += blockDim.x*gridDim.x;
  }
}

__global__ void two_pt_stencil(
    const int x, const int y, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y - 1) 
    return;

  a[gid] = b[gid] + b[gid + 1];
}

__global__ void two_pt_stencil_work(
    const int x, const int y, double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    a[gid] = b[gid] + b[gid + 1];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void two_pt_stencil_dist_10(
    const int x, const int y, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y - 10) 
    return;

  a[gid] = b[gid] + b[gid + 10];
}

__global__ void two_pt_stencil_dist_10_work(
    const int x, const int y, double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y - 10) {
    a[gid] = b[gid] + b[gid + 10];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void vec_sqrt(
    const int x, const int y, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = sqrt(b[gid]);
}

__global__ void vec_sqrt_work(
    const int x, const int y, double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    a[gid] = sqrt(b[gid]);
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void vec_add_sqrt(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] + sqrt(c[gid]);
}

__global__ void vec_add_sqrt_work(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  for(int ii = gid; ii < x*y; ii += blockDim.x*gridDim.x) 
    a[ii] = b[ii] + sqrt(c[ii]);
}

__global__ void vec_add_and_mul(
    const int x, const int y, double* a, const double* b, const double* c, 
    const double* d)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] + c[gid] * d[gid];
}

__global__ void vec_add_and_mul_work(
    const int x, const int y, double* a, const double* b, const double* c, 
    const double* d)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  while(gid < x*y) {
    a[gid] = b[gid] + c[gid] * d[gid];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void compute_bound(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  double m = a[gid];
  double n = 0.1;

  MAD_64(m, n); MAD_64(m, n);

  a[gid] = m + n;
}

__global__ void compute_bound_work(
    const int x, const int y, double* a, const double* b, const double* c)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    double m = a[gid];
    double n = 0.1;

    MAD_64(m, n); MAD_64(m, n);

    a[gid] = m + n;
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void five_pt_stencil_2d(
    const int x, const int y, const double delta, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
    a[gid] = b[gid - x] + b[gid - 1] - delta*b[gid] + 
      b[gid + 1] + b[gid + x];
}

__global__ void five_pt_stencil_2d_work(
    const int x, const int y, const double delta, double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    const int ii = gid / x;
    const int jj = gid % x;

    if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
      a[gid] = b[gid - x] + b[gid - 1] - delta*b[gid] + 
        b[gid + 1] + b[gid + x];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void nine_pt_stencil_2d(
    const int x, const int y, const double alpha, 
    const double beta, const double delta, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
    a[gid] =
      alpha  * b[gid - x - 1] +
      beta   * b[gid - x] +
      alpha  * b[gid - x + 1] +
      beta   * b[gid - 1] -
      delta  * b[gid] +
      beta   * b[gid + 1] +
      alpha  * b[gid + x - 1] +
      beta   * b[gid + x] +
      alpha  * b[gid + x + 1];
}

__global__ void nine_pt_stencil_2d_work(
    const int x, const int y, const double alpha, 
    const double beta, const double delta, double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    const int ii = gid / x;
    const int jj = gid % x;

    if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
      a[gid] =
        alpha  * b[gid - x - 1] +
        beta   * b[gid - x] +
        alpha  * b[gid - x + 1] +
        beta   * b[gid - 1] -
        delta  * b[gid] +
        beta   * b[gid + 1] +
        alpha  * b[gid + x - 1] +
        beta   * b[gid + x] +
        alpha  * b[gid + x + 1];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, 
    const double beta, double* a, const double* b)
{
  const int gid   = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii    = gid / (x*y);
  const int jj    = (gid / x) % y;
  const int kk    = gid % x;

  if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1 && kk > 0 && kk < x - 1)
    a[gid] =
      beta  * b[gid - x*y] +
      beta  * b[gid - x] +
      beta  * b[gid - 1] -
      alpha * b[gid] +
      beta  * b[gid + 1] +
      beta  * b[gid + x] +
      beta  * b[gid + x*y];
}

__global__ void seven_pt_stencil_3d_work(
    const int x, const int y, const int z, const double alpha, 
    const double beta, double* a, const double* b)
{
  int gid   = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y*z) {
    const int ii    = gid / (x*y);
    const int jj    = (gid / x) % y;
    const int kk    = gid % x;
    if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1 && kk > 0 && kk < x - 1)
      a[gid] =
        beta  * b[gid - x*y] +
        beta  * b[gid - x] +
        beta  * b[gid - 1] -
        alpha * b[gid] +
        beta  * b[gid + 1] +
        beta  * b[gid + x] +
        beta  * b[gid + x*y];

    gid += gridDim.x*blockDim.x;
  }
}

__global__ void twenty_seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, 
    const double beta, const double delta, const double gamma, 
    double* a, const double* b)
{
  const int gid   = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii    = gid / (x*y);
  const int jj    = (gid / x) % y;
  const int kk    = gid % x;

  if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1 && kk > 0 && kk < x - 1)
    a[gid] =
      delta  *b[gid - x*y - x - 1] +
      gamma  *b[gid - x*y - x] +
      delta  *b[gid - x*y - x + 1] +
      gamma  *b[gid - x*y - 1] +
      beta   *b[gid - x*y] +
      gamma  *b[gid - x*y + 1] +
      delta  *b[gid - x*y + x - 1] +
      gamma  *b[gid - x*y + x] +
      delta  *b[gid - x*y + x + 1] +
      gamma  *b[gid - x - 1] +
      beta   *b[gid - x] +
      gamma  *b[gid - x + 1] +
      beta   *b[gid - 1] -
      alpha  *b[gid] +
      beta   *b[gid + 1] +
      gamma  *b[gid + x - 1] +
      beta   *b[gid + x] +
      gamma  *b[gid + x + 1] +
      delta  *b[gid + x*y - x - 1] +
      gamma  *b[gid + x*y - x] +
      delta  *b[gid + x*y - x + 1] +
      gamma  *b[gid + x*y - 1] +
      beta   *b[gid + x*y] +
      gamma  *b[gid + x*y + 1] +
      delta  *b[gid + x*y + x - 1] +
      gamma  *b[gid + x*y + x] +
      delta  *b[gid + x*y + x + 1];
}

__global__ void twenty_seven_pt_stencil_3d_work(
    const int x, const int y, const int z, const double alpha, 
    const double beta, const double delta, const double gamma, 
    double* a, const double* b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y*z) {
    const int ii = gid / (x*y);
    const int jj = (gid / x) % y;
    const int kk = gid % x;
    if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1 && kk > 0 && kk < x - 1)
      a[gid] =
        delta  *b[gid - x*y - x - 1] +
        gamma  *b[gid - x*y - x] +
        delta  *b[gid - x*y - x + 1] +
        gamma  *b[gid - x*y - 1] +
        beta   *b[gid - x*y] +
        gamma  *b[gid - x*y + 1] +
        delta  *b[gid - x*y + x - 1] +
        gamma  *b[gid - x*y + x] +
        delta  *b[gid - x*y + x + 1] +
        gamma  *b[gid - x - 1] +
        beta   *b[gid - x] +
        gamma  *b[gid - x + 1] +
        beta   *b[gid - 1] -
        alpha  *b[gid] +
        beta   *b[gid + 1] +
        gamma  *b[gid + x - 1] +
        beta   *b[gid + x] +
        gamma  *b[gid + x + 1] +
        delta  *b[gid + x*y - x - 1] +
        gamma  *b[gid + x*y - x] +
        delta  *b[gid + x*y - x + 1] +
        gamma  *b[gid + x*y - 1] +
        beta   *b[gid + x*y] +
        gamma  *b[gid + x*y + 1] +
        delta  *b[gid + x*y + x - 1] +
        gamma  *b[gid + x*y + x] +
        delta  *b[gid + x*y + x + 1];
    gid += gridDim.x*blockDim.x;
  }
}

__global__ void twenty_seven_pt_stencil_3d_work2(
    const int x, const int y, const int z, const double alpha, 
    const double beta, const double delta, const double gamma, 
    double*  a, const double*  b)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  const int left = 1;
  const int bottom = 1;
  const int back = 1;
  const int right = 1;
  const int top = 1;
  const int front = 1;
  const int new_x = (x - left - right);
  const int new_y = (y - top - bottom);
  const int new_z = (z - front - back);

  while(gid < new_x*new_y*new_z) {
    const int ii = (gid / (new_x*new_y)) + front;
    const int jj = ((gid / new_x) % new_y) + bottom;
    const int kk = (gid % new_x) + left;
    const int index = ii*x*y + jj*x + kk;
    a[index] =
      delta  *b[index - x*y - x - 1] +
      gamma  *b[index - x*y - x] +
      delta  *b[index - x*y - x + 1] +
      gamma  *b[index - x*y - 1] +
      beta   *b[index - x*y] +
      gamma  *b[index - x*y + 1] +
      delta  *b[index - x*y + x - 1] +
      gamma  *b[index - x*y + x] +
      delta  *b[index - x*y + x + 1] +
      gamma  *b[index - x - 1] +
      beta   *b[index - x] +
      gamma  *b[index - x + 1] +
      beta   *b[index - 1] -
      alpha  *b[index] +
      beta   *b[index + 1] +
      gamma  *b[index + x - 1] +
      beta   *b[index + x] +
      gamma  *b[index + x + 1] +
      delta  *b[index + x*y - x - 1] +
      gamma  *b[index + x*y - x] +
      delta  *b[index + x*y - x + 1] +
      gamma  *b[index + x*y - 1] +
      beta   *b[index + x*y] +
      gamma  *b[index + x*y + 1] +
      delta  *b[index + x*y + x - 1] +
      gamma  *b[index + x*y + x] +
      delta  *b[index + x*y + x + 1];
    gid += gridDim.x*blockDim.x;
  }
}

__global__ void five_pt_wavefront(
    const int x, const int y, const int z, double* a)
{
  const int gid   = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii    = gid / (x*y);
  const int jj    = (gid / x) % y;
  const int dd    = gid % x;
  const int jjs   = jj - ii;

  if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1)
    a[ii*x*y + jjs*x + dd] = 
      (a[(ii - 1)*x*y + jjs*x + dd] + 
       a[ii*x*y + (jjs - 1)*x + dd] +
       a[ii*x*y + jjs*x + dd] + 
       a[ii*x*y + (jjs + 1)*x + dd] +
       a[(ii + 1)*x*y + jjs*x + dd])*0.2; 
}

__global__ void five_pt_wavefront_work(
    const int x, const int y, const int z, double* a)
{
  int gid   = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    const int ii    = gid / (x*y);
    const int jj    = (gid / x) % y;
    const int dd    = gid % x;
    const int jjs   = jj - ii;

    if(ii > 0 && ii < z - 1 && jj > 0 && jj < y - 1)
      a[ii*x*y + jjs*x + dd] = 
        (a[(ii - 1)*x*y + jjs*x + dd] + 
         a[ii*x*y + (jjs - 1)*x + dd] +
         a[ii*x*y + jjs*x + dd] + 
         a[ii*x*y + (jjs + 1)*x + dd] +
         a[(ii + 1)*x*y + jjs*x + dd])*0.2; 
    gid += blockDim.x*gridDim.x;
  }
}

/// A chebyshev iteration of the TeaLeaf application
__global__ void tealeaf_cheby_iter(
    const int x, const int y, const int halo_depth, double alpha, double beta,
    const double* a, const double* b, const double* c, double* d, double* e,
    const double* f, double* g)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;
  const int index = ii*x + jj;

  if(ii >= halo_depth && ii < y - halo_depth &&
      jj >= halo_depth && jj < x - halo_depth)
  {
    const double smvp = 
      (1.0 + (a[index+1]+a[index])
       + (b[index+x]+b[index]))*c[index]
      - (a[index+1]*c[index+1]+a[index]*c[index-1])
      - (b[index+x]*c[index+x]+b[index]*c[index-x]);
    d[index] = smvp;
    e[index] = f[index]-d[index];
    g[index] = alpha*g[index] + beta*e[index];
  }
}

/// A chebyshev iteration of the TeaLeaf application
__global__ void tealeaf_cheby_iter_work(
    const int x, const int y, const int halo_depth, double alpha, double beta,
    const double* a, const double* b, const double* c, double* d, double* e,
    const double* f, double* g)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    const int ii = gid / x;
    const int jj = gid % x;
    const int index = ii*x + jj;

    if(ii >= halo_depth && ii < y - halo_depth &&
        jj >= halo_depth && jj < x - halo_depth)
    {
      const double smvp = 
        (1.0 + (a[index+1]+a[index])
         + (b[index+x]+b[index]))*c[index]
        - (a[index+1]*c[index+1]+a[index]*c[index-1])
        - (b[index+x]*c[index+x]+b[index]*c[index-x]);
      d[index] = smvp;
      e[index] = f[index]-d[index];
      g[index] = alpha*g[index] + beta*e[index];
    }
    gid += gridDim.x*blockDim.x;
  }
}

__global__ void cloverleaf_energy_flux(
    const int x, const int y, const int halo_depth, const double* a, const double* b, 
    const double* c, const double* d, double* e, double* f, const double* a_s)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(ii >= halo_depth && ii < y && jj >= halo_depth && jj < x - halo_depth) {
    const int dif = ii - 1;
    const int rflux = a[gid] > 0.0;
    const int upwind = (rflux) ? (ii-2)*x + jj : min(ii + 1, y + 2)*x + jj;
    const int donor = (rflux) ? (ii-1)*x + jj : ii*x + jj;
    const int downwind = (rflux) ? ii*x + jj : (ii - 1)*x + jj;
    const double sigmat = fabs(a[gid] / b[donor]);
    const double sigma3 = (1.0+sigmat)*(a_s[ii] / a_s[dif]);
    const double sigma4 = 2.0 - sigmat;
    const double sigmav=sigmat;
    double diffuw = c[donor] - c[upwind];
    double diffdw = c[downwind] - c[donor];

    double limiter = 0.0;
    if(diffuw*diffdw > 0.0)
    {
      limiter = (1.0 - sigmav)*((diffdw > 0) - (diffdw < 0))*
        min(fabs(diffuw), min(fabs(diffdw), (1.0/6.0)*
              (sigma3*fabs(diffuw) + sigma4*fabs(diffdw))));
    }

    e[gid] = a[gid]*(c[donor] + limiter);

    const double sigmam = fabs(e[gid]) / (c[donor]*b[donor]);
    diffuw = d[donor] - d[upwind];
    diffdw = d[downwind] - d[donor];

    limiter = 0.0;
    if(diffuw*diffdw > 0.0)
    {
      limiter = (1.0-sigmam)*((diffdw > 0) - (diffdw < 0))* 
        min(fabs(diffuw),min(fabs(diffdw), (1.0/6.0)*
              (sigma3*fabs(diffuw)+sigma4*fabs(diffdw))));
    }

    f[gid] = e[gid]*(d[donor] + limiter);
  }
}

__global__ void cloverleaf_energy_flux_work(
    const int x, const int y, const int halo_depth, const double* a, const double* b, 
    const double* c, const double* d, double* e, double* f, const double* a_s)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < x*y) {
    const int ii = gid / x;
    const int jj = gid % x;

    if(ii >= halo_depth && ii < y && jj >= halo_depth && jj < x - halo_depth) {
      const int dif = ii - 1;

      const int rflux = a[gid] > 0.0;
      const int upwind = (rflux) ? (ii-2)*x + jj : min(ii + 1, y + 2)*x + jj;
      const int donor = (rflux) ? (ii-1)*x + jj : ii*x + jj;
      const int downwind = (rflux) ? ii*x + jj : (ii - 1)*x + jj;
      const double sigmat = fabs(a[gid] / b[donor]);
      const double sigma3 = (1.0+sigmat)*(a_s[ii] / a_s[dif]);
      const double sigma4 = 2.0 - sigmat;
      const double sigmav=sigmat;
      double diffuw = c[donor] - c[upwind];
      double diffdw = c[downwind] - c[donor];

      double limiter = 0.0;
      if(diffuw*diffdw > 0.0)
      {
        limiter = (1.0 - sigmav)*((diffdw > 0) - (diffdw < 0))*
          min(fabs(diffuw), min(fabs(diffdw), (1.0/6.0)*
                (sigma3*fabs(diffuw) + sigma4*fabs(diffdw))));
      }

      e[gid] = a[gid]*(c[donor] + limiter);

      const double sigmam = fabs(e[gid]) / (c[donor]*b[donor]);
      diffuw = d[donor] - d[upwind];
      diffdw = d[downwind] - d[donor];

      limiter = 0.0;
      if(diffuw*diffdw > 0.0)
      {
        limiter = (1.0-sigmam)*((diffdw > 0) - (diffdw < 0))* 
          min(fabs(diffuw),min(fabs(diffdw), (1.0/6.0)*
                (sigma3*fabs(diffuw)+sigma4*fabs(diffdw))));
      }

      f[gid] = e[gid]*(d[donor] + limiter);
    }

    gid += gridDim.x*blockDim.x;
  }
}

__global__ void snap_sweep(
    const int x, const int y, const int plane_length, const int ss,
    const int offset, const int sweep_mod_ii, const int sweep_mod_jj, 
    const int* sweep_indices, const double alpha, const int nmom, 
    const int nang, const int ngroups, const double* a, const double* b, 
    double* c, double* d, const double* e, const double* f, double* g, 
    double* a_s, double* b_s, double* c_s)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= plane_length*nang*ngroups) 
    return;

  // Fetch the indices from the plane indirections
  const int spatial_offset = gid/(nang*ngroups);
  const int tot_offset = offset + spatial_offset*2;
  const int ii = (sweep_mod_ii)
    ? sweep_indices[tot_offset] 
    : (y - sweep_indices[tot_offset] - 1);
  const int jj = (sweep_mod_jj) 
    ? sweep_indices[tot_offset + 1] 
    : (x - sweep_indices[tot_offset + 1] - 1);

  // Get group 
  const int gg = (gid/nang)%ngroups;

  // Get angle
  const int aa = gid%nang;

  double source_term = a[0 + jj*nmom + ii*nmom*x + gg*nmom*x*y];

  // Add in the anisotropic scattering source moments
  for (int ll = 1; ll < nmom; ll++)
  {
    source_term += b[ll + aa*nmom + ss*nmom*nang]*
      a[ll + jj*nmom + ii*nmom*x + gg*nmom*x*y];
  }

  double psi = (source_term + 
      c[aa + gg*nang + ii*nang*ngroups]*a_s[aa]*alpha + 
      d[aa + gg*nang + jj*nang*ngroups]*b_s[aa] + 
      e[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x]*c_s[gg])*
    f[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x];

  c[aa + gg*nang + ii*nang*ngroups] = 
    2.0*psi - c[aa + gg*nang + ii*nang*ngroups];
  d[aa + gg*nang + jj*nang*ngroups] =
    2.0*psi - d[aa + gg*nang + jj*nang*ngroups];
  g[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x] = 
    2.0*psi - e[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x];
}

__global__ void snap_sweep_work(
    const int x, const int y, const int plane_length, const int ss,
    const int offset, const int sweep_mod_ii, const int sweep_mod_jj, 
    const int* sweep_indices, const double alpha, const int nmom, 
    const int nang, const int ngroups, const double* a, const double* b, 
    double* c, double* d, const double* e, const double* f, double* g, 
    double* a_s, double* b_s, double* c_s)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  while(gid < plane_length*nang*ngroups) {
    // Fetch the indices from the plane indirections
    const int spatial_offset = gid/(nang*ngroups);
    const int tot_offset = offset + spatial_offset*2;
    const int ii = (sweep_mod_ii)
      ? sweep_indices[tot_offset] 
      : (y - sweep_indices[tot_offset] - 1);
    const int jj = (sweep_mod_jj) 
      ? sweep_indices[tot_offset + 1] 
      : (x - sweep_indices[tot_offset + 1] - 1);

    // Get group 
    const int gg = (gid/nang)%ngroups;

    // Get angle
    const int aa = gid%nang;

    double source_term = a[0 + jj*nmom + ii*nmom*x + gg*nmom*x*y];

    // Add in the anisotropic scattering source moments
    for (int ll = 1; ll < nmom; ll++)
    {
      source_term += b[ll + aa*nmom + ss*nmom*nang]*
        a[ll + jj*nmom + ii*nmom*x + gg*nmom*x*y];
    }

    double psi = (source_term + 
        c[aa + gg*nang + ii*nang*ngroups]*a_s[aa]*alpha + 
        d[aa + gg*nang + jj*nang*ngroups]*b_s[aa] + 
        e[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x]*c_s[gg])*
      f[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x];

    c[aa + gg*nang + ii*nang*ngroups] = 
      2.0*psi - c[aa + gg*nang + ii*nang*ngroups];
    d[aa + gg*nang + jj*nang*ngroups] =
      2.0*psi - d[aa + gg*nang + jj*nang*ngroups];
    g[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x] = 
      2.0*psi - e[aa + gg*nang + jj*nang*ngroups + ii*nang*ngroups*x];

    gid += blockDim.x*gridDim.x;
  }
}

__global__ void matrix_multiply(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y)
    return;

#if 0
  const int ii = gid / x;
  const int jj = gid % x;
#endif // if 0

  a[gid] = 0.0;

  for(int kk = 0; kk < y; ++kk) {
    for(int ll = 0; ll < x; ++ll) {
      a[gid] += b[kk*x + ll]*c[ll*y + kk];
    }
  }
}

#if 0
__global__ void vec_add_work_increase(
    const int x, const int y, double* a, const double* b, const double* c)
{
  int gid = blockDim.x*blockIdx.x + threadIdx.x;
  while(gid < x*y) {
    a[gid] = b[gid] + c[gid];
    gid += blockDim.x*gridDim.x;
  }
}

__global__ void vec_add_inplace(
    const int x, const int y, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] += b[gid];
}

__global__ void vec_add(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] + c[gid];
}

__global__ void mat_mul_transposed(
    const int x, const int y, const int z, double* a, 
    const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y*z) 
    return;

  const int ii = gid / (x*y);
  const int jj = (gid / x) % y;
  const int kk = gid % x;

  a[ii*x + jj] += b[ii*x + kk]*c[jj*x + kk];
}

__device__ void vec_add_kernel(
    const int gid, double* a, const double* b, const double* c)
{
  a[gid] = b[gid] + c[gid];
}

__global__ void vec_add_elemental(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  vec_add_kernel(gid, a, b, c);
}

__global__ void vec_mul(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] * c[gid];
}

__global__ void vec_div(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] / c[gid];
}

__global__ void vec_add_and_mul_2d(
    const int x, const int y, double* a, const double* b, const double* c, 
    const double* d)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  const int ii = gid / x;
  const int jj = gid % x;
  const int index = ii*x + jj;
  a[index] = b[index] + c[index] * d[index];
}

__global__ void vec_add_twice(
    const int x, const int y, double* a, const double* b, const double* c, 
    double* d, const double* e, const double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid] + c[gid];
  d[gid] = e[gid] + f[gid];
}

__global__ void vec_div_twice(
    const int x, const int y, double* a, const double* b, const double* c, 
    double* d, const double* e, const double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  a[gid] = b[gid]/c[gid];
  d[gid] = e[gid]/f[gid];
}

__device__ void vec_add_and_div_elemental(
    const int gid, double* a, const double* b, const double* c, const double* d)
{
  a[gid] = b[gid] + c[gid]/d[gid];
}

__global__ void vec_add_and_div_twice_elemental(
    const int x, const int y, double* a, const double* b, const double* c, 
    const double* d, double* e, const double* f, const double* g,
    const double* h)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  vec_add_and_div_elemental(gid, a, b, c, d);
  vec_add_and_div_elemental(gid, e, f, g, h);
}

__global__ void three_pt_stencil(
    const int x, const int y, const double alpha, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid < 1 || gid >= x*y - 1) 
    return;

  a[gid] = b[gid - 1] - 2.0*b[gid] + b[gid + 1];
}

__global__ void three_pt_stencil_2d(
    const int x, const int y, const double alpha, double* a, const double* b)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
    a[gid] = b[gid - 1] - 2.0*b[gid] + b[gid + 1];
}

__global__ void five_pt_stencil_2d_twice(
    const int x, const int y, const double alpha, double* a, const double* b, 
    double* c, const double* d, double* e, const double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = gid / x;
  const int jj = gid % x;

  if(ii > 0 && ii < y - 1 && jj > 0 && jj < x - 1)
  {
    a[gid] = b[gid - x] + b[gid - 1] - 
      alpha*b[gid] + b[gid + 1] + b[gid + x];
    c[gid] = d[gid - x] + d[gid - 1] - 
      alpha*d[gid] + d[gid + 1] + d[gid + x];
  }
}

__global__ void five_pt_indirect_sweep(
    const int x, const int y, const int z, const int plane_length, 
    const int offset, const int sweep_mod_ii, const int sweep_mod_jj, 
    int* sweep_indices, double* a)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= plane_length) 
    return;

  const int dd = gid % z;

  // Each spatial pt has z data pts
  const int spatial_offset = gid / z;
  const int tot_offset = offset + spatial_offset*2;
  const int ii = (sweep_mod_ii)
    ? sweep_indices[tot_offset] 
    : (y - sweep_indices[tot_offset] - 1);
  const int jj = (sweep_mod_jj) 
    ? sweep_indices[tot_offset + 1] 
    : (x - sweep_indices[tot_offset + 1] - 1);

  a[ii*x*y + jj*x + dd] = 
    a[(ii - 1)*x*y + jj*x + dd] + 
    a[ii*x*y + (jj - 1)*x + dd] - 
    a[ii*x*y + jj*x + dd] + 
    a[ii*x*y + (jj + 1)*x + dd] + 
    a[(ii + 1)*x*y + jj*x + dd] ;
}

__global__ void even_odd_divergence(
    const int x, const int y, double* a, const double* b, const double* c,
    double* d, const double* e, const double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  const int ii = gid / x;
  const int jj = gid % x;
  const int index = ii*x + jj;

  if(index % 2 == 0)
  {
    a[index] = b[index] + c[index];
  }
  else
  {
    d[index] = e[index] + f[index];
  }
}

__global__ void half_team_divergence(
    const int x, const int y, double* a, const double* b, const double* c,
    double* d, const double* e, const double* f)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  const int ii = gid / x;
  const int jj = gid % x;
  const int index = ii*x + jj;

  if(threadIdx.x < blockDim.x / 2)
  {
    a[index] = b[index] + c[index];
  }
  else
  {
    d[index] = e[index] + f[index];
  }
}

__global__ void triangular_load_imbalance(
    const int x, const int y, double* a, const double* b, const double* c)
{
  const int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid >= x*y) 
    return;

  const int ii = gid / x;
  const int jj = gid % x;

  const int index = ii*x + jj;
  const int num_load_groups = 16;
  const int load_group_size = x*y/num_load_groups;
  const int load_group = index / load_group_size;
  const int load_group_start = (load_group*(load_group+1))/2;

  for(int kk = 0; kk < load_group; ++kk)
  {
    a[index] = b[load_group_start + kk] + c[load_group_start + kk];
  }
}
__global__ void vec_add_tiling_32_by_32(
    const int x, const int y, double* a, const double* b, const double* c)
{
}

#endif // if 0


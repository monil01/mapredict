#pragma once

#include "profiler.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

// Default startup parameters
#define X 4096
#define Y 4096
#define NITERS 100
#define BLOCK_SIZE 128

#define MAD(a, b, c) (a = (a*b) + c);
#define MAD_4(m, n) \
  MAD(m, n, n); MAD(n, m, m); MAD(m, -n, n); MAD(n, m, m);
#define MAD_16(m, n) \
  MAD_4(m, n); MAD_4(m, n); MAD_4(m, n); MAD_4(m, n);
#define MAD_64(m, n) \
  MAD_16(m, n); MAD_16(m, n); MAD_16(m, n); MAD_16(m, n);

// Kernel constants
#define ALPHA 1.0
#define BETA 2.0
#define DELTA 3.0
#define GAMMA 4.0
#define HALO_DEPTH 2
#define SWEEPX 64
#define SWEEPY 64
#define NGROUPS 50
#define NANG 136
#define NMOM 2
#define NDIMS 2
#define NSWEEPS 4

// Pre-computed validation numbers
#define TEALEAF_VALIDATE 1.674446400000e+06
#define CLOVERLEAF_VALIDATE 1.686313200422e+06
#define SNAP_VALIDATE -1.973234358353e+06
#define COMPUTE_BOUND_VALIDATE 1.046113500730e+00
#define TOLERANCE 1.0e-8
#define GB (1024.0*1024.0*1024.0)

// Snap array sizes
#define A_LEN (NGROUPS*NMOM*SWEEPX*SWEEPY)
#define B_LEN (NSWEEPS*NMOM*NANG)
#define C_LEN (NANG*NGROUPS*SWEEPY)
#define D_LEN (NANG*NGROUPS*SWEEPX)
#define E_LEN (NGROUPS*NANG*SWEEPX*SWEEPY)
#define F_LEN (NGROUPS*NANG*SWEEPX*SWEEPY)
#define G_LEN (NGROUPS*NANG*SWEEPX*SWEEPY)

#define cuda_check_errors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define PRE_KERNEL() \
  START_PROFILING(p.profile);

#define POST_FLOP_KERNEL(nflops, nmem_access, func_name) \
  cuda_check_errors(cudaDeviceSynchronize()); \
  STOP_PROFILING(p.profile, func_name); \
  print_time_and_flops(func_name, p.niters, nflops, nmem_access);

#define POST_MEM_KERNEL(nwrites, nreads, func_name) \
  cuda_check_errors(cudaDeviceSynchronize());\
  STOP_PROFILING(p.profile, func_name); \
  print_time_and_mem_bw(func_name, p.niters, nreads, nwrites);

typedef struct
{
  int niters;
  int ndims;
  int x;
  int y;
  int x3;
  int y3;
  int z3;

  double* a_l;
  double* a_s;
  double* b_s;
  double* c_s;
  double* d_s;

  double* a;
  double* b;
  double* c;
  double* d;
  double* e;
  double* f;
  double* g;
  double* h;
  int* sweep_indices;
  int* indirection;
  int* c_indirection;
  int* r_indirection;

  struct Profile* profile;

} KernelParams;

void run_full_micro_suite( 
    double* a, double* b, double* c, double* d, double* e, 
    double* f, double* g, double* a_s, double* b_s, double* c_s, double* d_s, 
    int* c_indirection, int* r_indirection, int* sweep_indices);
void init(KernelParams* p);
void print_time(const char* func_name);
void run_snap_sweep(
    const int x, const int y, const int nsweeps, const int nplanes, const int ngroups, 
    const int nang, const int nmom, const int ndims, const double alpha, const double beta, 
    double* a_s, double* b_s, double* c_s);
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void print_time_and_flops(
    const char* func_name, const int niters,
    const double nflops, const double nmem_access);
void print_time_and_mem_bw(
    const char* func_name, const int niters, 
    const double nreads, const double nwrites);


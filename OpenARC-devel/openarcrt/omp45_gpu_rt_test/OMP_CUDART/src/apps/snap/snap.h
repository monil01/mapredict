#ifndef SNAP_H
#define SNAP_H
// Kernel constants
#define ALPHA 1.0
#define BETA 2.0
#define DELTA 3.0
#define GAMMA 4.0
#define HALO_DEPTH 2
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

double run_snap_global(struct user_parameters* params);
double run_snap_task(struct user_parameters* params);

void validate_snap_sweep(
    const int x, const int y, const double* d_g);

__global__ void snap_sweep_global1(const int x, const int y,
		const int plane_length, const int ss, const int offset,
		const int sweep_mod_ii, const int sweep_mod_jj,
		const int* sweep_indices, const double alpha, const int nmom,
		const int nang, const int ngroups, const double* a, const double* b,
		double* c, double* d, const double* e, const double* f, double* g,
		double* a_s, double* b_s, double* c_s);
__global__ void snap_sweep_global2(const int x, const int y,
		const int plane_length, const int ss, const int offset,
		const int sweep_mod_ii, const int sweep_mod_jj,
		const int* sweep_indices, const double alpha, const int nmom,
		const int nang, const int ngroups, const double* a, const double* b,
		double* c, double* d, const double* e, const double* f, double* g,
		double* a_s, double* b_s, double* c_s);

__global__ void init_snap_sweep(const int nplanes, const int x, const int y,
		double* a, double* b, double* c, double* d, double* e, double* f,
		double* g, double* a_s, double* b_s, double* c_s);

#endif

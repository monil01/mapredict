#include "../../rt.h"
#include "snap.h"

#ifdef APP_SNAP
__device__ inline void snap_sweep_core(const int gid, const int x, const int y,
		const int ss, const int offset, const int sweep_mod_ii,
		const int sweep_mod_jj, const int* sweep_indices, const double alpha,
		const int nmom, const int nang, const int ngroups, const double* a,
		const double* b, double* c, double* d, const double* e, const double* f,
		double* g, double* a_s, double* b_s, double* c_s) {
	// Fetch the indices from the plane indirections
	const int spatial_offset = gid / (nang * ngroups);
	const int tot_offset = offset + spatial_offset * 2;
	const int ii =
			(sweep_mod_ii) ?
					sweep_indices[tot_offset] :
					(y - sweep_indices[tot_offset] - 1);
	const int jj =
			(sweep_mod_jj) ?
					sweep_indices[tot_offset + 1] :
					(x - sweep_indices[tot_offset + 1] - 1);

	// Get group
	const int gg = (gid / nang) % ngroups;

	// Get angle
	const int aa = gid % nang;

	double source_term = a[0 + jj * nmom + ii * nmom * x + gg * nmom * x * y];

	// Add in the anisotropic scattering source moments
	for (int ll = 1; ll < nmom; ll++) {
		source_term += b[ll + aa * nmom + ss * nmom * nang]
				* a[ll + jj * nmom + ii * nmom * x + gg * nmom * x * y];
	}

	double psi = (source_term
			+ c[aa + gg * nang + ii * nang * ngroups] * a_s[aa] * alpha
			+ d[aa + gg * nang + jj * nang * ngroups] * b_s[aa]
			+ e[aa + gg * nang + jj * nang * ngroups + ii * nang * ngroups * x]
					* c_s[gg])
			* f[aa + gg * nang + jj * nang * ngroups + ii * nang * ngroups * x];

	c[aa + gg * nang + ii * nang * ngroups] = 2.0 * psi
			- c[aa + gg * nang + ii * nang * ngroups];
	d[aa + gg * nang + jj * nang * ngroups] = 2.0 * psi
			- d[aa + gg * nang + jj * nang * ngroups];
	g[aa + gg * nang + jj * nang * ngroups + ii * nang * ngroups * x] = 2.0
			* psi
			- e[aa + gg * nang + jj * nang * ngroups + ii * nang * ngroups * x];

}

__global__ void snap_sweep_global1(const int x, const int y,
		const int plane_length, const int ss, const int offset,
		const int sweep_mod_ii, const int sweep_mod_jj,
		const int* sweep_indices, const double alpha, const int nmom,
		const int nang, const int ngroups, const double* a, const double* b,
		double* c, double* d, const double* e, const double* f, double* g,
		double* a_s, double* b_s, double* c_s) {
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid >= plane_length * nang * ngroups)
		return;
	snap_sweep_core(gid, x, y, ss, offset, sweep_mod_ii, sweep_mod_jj,
			sweep_indices, alpha, nmom, nang, ngroups, a, b, c, d, e, f, g, a_s,
			b_s, c_s);
}

__global__ void snap_sweep_global2(const int x, const int y,
		const int plane_length, const int ss, const int offset,
		const int sweep_mod_ii, const int sweep_mod_jj,
		const int* sweep_indices, const double alpha, const int nmom,
		const int nang, const int ngroups, const double* a, const double* b,
		double* c, double* d, const double* e, const double* f, double* g,
		double* a_s, double* b_s, double* c_s) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	while (gid < plane_length * nang * ngroups) {
		snap_sweep_core(gid, x, y, ss, offset, sweep_mod_ii, sweep_mod_jj,
				sweep_indices, alpha, nmom, nang, ngroups, a, b, c, d, e, f, g,
				a_s, b_s, c_s);

		gid += blockDim.x * gridDim.x;
	}
}

__global__ void init_snap_sweep(const int nplanes, const int x, const int y,
		double* a, double* b, double* c, double* d, double* e, double* f,
		double* g, double* a_s, double* b_s, double* c_s) {
	const int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int SWEEPX = x;
	int SWEEPY = y;
	int A_LEN = (NGROUPS * NMOM * x * y);
	int B_LEN = (NSWEEPS * NMOM * NANG);
	int C_LEN = (NANG * NGROUPS * SWEEPY);
	int D_LEN = (NANG * NGROUPS * SWEEPX);
	int E_LEN = (NGROUPS * NANG * SWEEPX * SWEEPY);
	int F_LEN = (NGROUPS * NANG * SWEEPX * SWEEPY);
	int G_LEN = (NGROUPS * NANG * SWEEPX * SWEEPY);

	if (gid < A_LEN)
		a[gid] = 0.1;
	if (gid < B_LEN)
		b[gid] = 0.1;
	if (gid < C_LEN)
		c[gid] = 0.1;
	if (gid < D_LEN)
		d[gid] = 0.1;
	if (gid < E_LEN)
		e[gid] = 0.1;
	if (gid < F_LEN)
		f[gid] = 0.1;
	if (gid < G_LEN)
		g[gid] = 0.1;

	if (gid < x) {
		a_s[gid] = 0.1;
		b_s[gid] = 0.1;
		c_s[gid] = 0.1;
	}
}
__device__ void  app_kernel(TASK* task, APP_CONTEXT* appContext, RT_CONTEXT* dynContext){

}

#endif

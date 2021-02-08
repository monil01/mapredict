#include "../../rt.h"
#include "../main.h"
#include "snap.h"
#include "../timer.h"


#ifdef APP_SNAP
/*
 * Simplified SNAP sweep kernel excluding negative flux fix up
 * and assuming time dependent
 */
double run_snap_global(struct user_parameters* params) {

	double totalTime = 0;
	int x = params->matrix_size;
	int y = params->matrix_size;
	int blockSize = params->blocksize;
	int nsweeps = NSWEEPS;
	int nplanes = x+y-1;
	int ngroups = NGROUPS;
	int nang = NANG;
	int nmom = NMOM;
	int ndims = NDIMS;
	double alpha = ALPHA;
	double beta = BETA;
	const int sweep_mods_ii[NSWEEPS] = { 1, 0, 1, 0 };
	const int sweep_mods_jj[NSWEEPS] = { 1, 1, 0, 0 };
	int SWEEPX = x;
	int SWEEPY = y;

	double* a;
	double* b;
	double* c;
	double* d;
	double* e;
	double* f;
	double* g;
	int* sweep_indices;
	int A_LEN = (NGROUPS*NMOM*SWEEPX*SWEEPY);
	int B_LEN = (NSWEEPS*NMOM*NANG);
	int C_LEN = (NANG*NGROUPS*SWEEPY);
	int D_LEN = (NANG*NGROUPS*SWEEPX);
	int E_LEN = (NGROUPS*NANG*SWEEPX*SWEEPY);
	int F_LEN = (NGROUPS*NANG*SWEEPX*SWEEPY);
	int G_LEN = (NGROUPS*NANG*SWEEPX*SWEEPY);

	double* a_s;
	double* b_s;
	double* c_s;
	const int small_buffer_length = max(x, y);


	// Initialise the 2d snap indirection lists for positive upwind quadrant
	int* h_sweep_indices = (int*) malloc(sizeof(int) * x * y * ndims);
	int sweep_offset = 0;
	for (int jj = 2; jj < nplanes - 2; ++jj)
	//for(int jj = 2; jj < nplanes - 1; ++jj)
			{
		for (int ii = max(1, jj - x + 2); ii < min(y - 1, jj); ++ii) {
			// Skewed jj index
			const int jjs = jj - ii;
			h_sweep_indices[sweep_offset++] = ii;
			h_sweep_indices[sweep_offset++] = jjs;
		}
	}


	cudaMalloc((void**) &a_s, small_buffer_length * sizeof(double));
	cudaMalloc((void**) &b_s, small_buffer_length * sizeof(double));
	cudaMalloc((void**) &c_s, small_buffer_length * sizeof(double));


	cudaMalloc((void**) &a, A_LEN * sizeof(double));
	cudaMalloc((void**) &b, B_LEN * sizeof(double));
	cudaMalloc((void**) &c, C_LEN * sizeof(double));
	cudaMalloc((void**) &d, D_LEN * sizeof(double));
	cudaMalloc((void**) &e, E_LEN * sizeof(double));
	cudaMalloc((void**) &f, F_LEN * sizeof(double));
	cudaMalloc((void**) &g, G_LEN * sizeof(double));
	cudaMalloc((void**) &sweep_indices, sizeof(int) * x * y * ndims);
	cudaDeviceSynchronize();


	cudaMemcpy(sweep_indices, h_sweep_indices, sizeof(int) * x * y * ndims,
			cudaMemcpyHostToDevice);

	const int n2d_blocks = ceil((double) x * y / blockSize);
	init_snap_sweep<<<n2d_blocks, blockSize>>>(nplanes, x, y, a, b, c, d,
			e, f, g, a_s, b_s, c_s);
	//PRE_KERNEL();

	// niter is assumed to be always 1
	for (int tt = 0; tt < params->niter; ++tt) {
		const int ss = tt % nsweeps;
		const int sweep_mod_ii = sweep_mods_ii[ss];
		const int sweep_mod_jj = sweep_mods_jj[ss];

		int offset = 0;
		for (int pp = 2; pp < nplanes - 2; ++pp) {
			const int plane_length = min(pp - 1, nplanes - pp - 2);
			const int data_in_plane = plane_length * nang * ngroups;
			const int nblocks = ceil((double) data_in_plane / blockSize);

			START_TIMER
#if defined(RUN_DYNAMIC_WORK)
			snap_sweep_work<<<512, 1024>>>(
					x, y, plane_length, ss, offset, sweep_mod_ii, sweep_mod_jj,
					sweep_indices, alpha, nmom, nang, ngroups, a, b, c, d, e, f, g,
					a_s, b_s, c_s);
#else
			snap_sweep_global1<<<nblocks, blockSize>>>(x, y, plane_length, ss, offset,
					sweep_mod_ii, sweep_mod_jj, sweep_indices, alpha, nmom,
					nang, ngroups, a, b, c, d, e, f, g, a_s, b_s, c_s);
#endif
			checkCudaErrors(cudaDeviceSynchronize());

			END_TIMER
			offset += (plane_length * ndims);

			totalTime += TIMER;
		}
	}


	if (params->check){
		validate_snap_sweep(x, y, g);
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(f);
	cudaFree(g);

	return totalTime;
}

#endif

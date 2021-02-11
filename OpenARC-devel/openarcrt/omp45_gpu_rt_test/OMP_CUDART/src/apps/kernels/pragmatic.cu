#include <stdlib.h>
#include <cmath>
#include <stdio.h>
#include "initialization.cu"
#include "profiler.h"
#include "pragmatic.h"
#include "kernels.cu"
#include "validate.h"

KernelParams p;
#ifdef APP_KERNELS
double run(struct user_parameters* params)
{
}
#endif

int run(int argc, char** argv) {
	struct Profile profile;

	int length = 0;
	if (argc == 1) {
		length = X * Y;
		p.x = X;
		p.y = Y;
		p.niters = NITERS;
	} else {
		length = atoi(argv[1]);
		p.x = sqrt(length);
		p.y = sqrt(length);

		if (argc == 2)
			p.niters = NITERS;
		else
			p.niters = atoi(argv[2]);
	}

	p.x3 = cbrt((double) length);
	p.y3 = cbrt((double) length);
	p.z3 = cbrt((double) length);

	p.profile = &profile;

	printf("\nRunning 1d problems with dims (%d)\n", length);
	printf("Running 2d problems with dims (%d, %d)\n", p.x, p.y);
	printf("Running 3d problems with dims (%d, %d, %d)\n\n", p.x3, p.y3, p.z3);

#if defined(RUN_DYNAMIC_WORK)
	printf("Running dynamic work kernels to match Clang\n");
#endif

	init(&p);

	run_full_micro_suite(p.a, p.b, p.c, p.d, p.e, p.f, p.g, p.a_s, p.b_s, p.c_s,
			p.d_s, p.c_indirection, p.r_indirection, p.sweep_indices);

	return EXIT_SUCCESS;
}

// Runs the entire micro benchmark suite
void run_full_micro_suite(double* a, double* b, double* c, double* d, double* e,
		double* f, double* g, double* a_s, double* b_s, double* c_s,
		double* d_s, int* c_indirection, int* r_indirection,
		int* sweep_indices) {
	const int n2d_blocks = ceil((double) p.x * p.y / BLOCK_SIZE);
	const int n3d_blocks = ceil((double) p.x3 * p.y3 * p.z3 / BLOCK_SIZE);

	init_vec_add_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		vec_add_2d_work<<<512, 1024>>>(p.x, p.y, a, b, c);
#else
		vec_add_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 2 * p.x * p.y, "vec_add_2d");

#if defined(RUN_VALIDATION_SUITE)
	validate_vec_add_2d(p.x, p.y, a);
#endif

	init_vec_add_sqrt<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		vec_add_sqrt_work<<<512, 1024>>>(p.x, p.y, a, b, c);
#else
		vec_add_sqrt<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 2 * p.x * p.y, "vec_add_sqrt");

#if defined(RUN_VALIDATION_SUITE)
	validate_vec_add_sqrt(p.x, p.y, a);
#endif

	init_vec_add_and_mul<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c, d);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		vec_add_and_mul_work<<<512, 1024>>>(p.x, p.y, a, b, c, d);
#else
		vec_add_and_mul<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c, d);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 3 * p.x * p.y, "vec_add_and_mul");

#if defined(RUN_VALIDATION_SUITE)
	validate_vec_add_and_mul(p.x, p.y, a);
#endif

	init_vec_add_reverse_indirection<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b,
			c, r_indirection);
	init_reverse_indirection<<<n2d_blocks, BLOCK_SIZE>>>(p.x, r_indirection);

	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		vec_add_reverse_indirect_work<<<512, 1024>>>(p.x, p.y, r_indirection, a, b, c);
#else
		vec_add_reverse_indirect<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y,
				r_indirection, a, b, c);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 2 * p.x * p.y, "reverse_indirect");

#if defined(RUN_VALIDATION_SUITE)
	validate_vec_add_reverse_indirection(p.x, p.y, a);
#endif

	init_vec_add_column_indirection<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b,
			c, c_indirection);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
		// Not sure if this is actually representative of the loop nest
#if defined(RUN_DYNAMIC_WORK)
		vec_add_column_indirect_work<<<512, 1024>>>(p.x, p.y, c_indirection, a, b, c);
#else
		vec_add_column_indirect<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y,
				c_indirection, a, b, c);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 2 * p.x * p.y, "column_indirect");

#if defined(RUN_VALIDATION_SUITE)
	validate_vec_add_column_indirection(p.x, p.y, a);
#endif

	init_two_pt_stencil<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		two_pt_stencil_work<<<512, 1024>>>(p.x, p.y, a, b);
#else
		two_pt_stencil<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y, "2pt_stencil");

#if defined(RUN_VALIDATION_SUITE)
	validate_two_pt_stencil(p.x, p.y, a);
#endif

	init_two_pt_stencil_dist_10<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		two_pt_stencil_dist_10_work<<<512, 1024>>>(p.x, p.y, a, b);
#else
		two_pt_stencil_dist_10<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y, "2pt_stencil_10_gap");

#if defined(RUN_VALIDATION_SUITE)
	validate_two_pt_stencil_dist_10(p.x, p.y, a);
#endif

	init_five_pt_stencil_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		five_pt_stencil_2d_work<<<512, 1024>>>(p.x, p.y, DELTA, a, b);
#else
		five_pt_stencil_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, DELTA, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y, "5pt_stencil_2d");

#if defined(RUN_VALIDATION_SUITE)
	validate_five_pt_stencil_2d(p.x, p.y, DELTA, a);
#endif

	init_nine_pt_stencil_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		nine_pt_stencil_2d_work<<<512, 1024>>>(p.x, p.y, ALPHA, BETA, DELTA, a, b);
#else
		nine_pt_stencil_2d<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, ALPHA, BETA,
				DELTA, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y, "9pt_stencil_2d");

#if defined(RUN_VALIDATION_SUITE)
	validate_nine_pt_stencil_2d(p.x, p.y, ALPHA, BETA, DELTA, a);
#endif

	init_seven_pt_stencil_3d<<<n3d_blocks, BLOCK_SIZE>>>(p.x3, p.y3, p.z3, a,
			b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		seven_pt_stencil_3d_work<<<512, 1024>>>(p.x3, p.y3, p.z3, ALPHA, BETA, a, b);
#else
		seven_pt_stencil_3d<<<n3d_blocks, BLOCK_SIZE>>>(p.x3, p.y3, p.z3, ALPHA,
				BETA, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x3 * p.y3 * p.z3, 2 * p.x3 * p.y3 * p.z3,
			"7pt_stencil_3d");

#if defined(RUN_VALIDATION_SUITE)
	validate_seven_pt_stencil_3d(p.x3, p.y3, p.z3, ALPHA, BETA, a);
#endif

	init_twenty_seven_pt_stencil_3d<<<n3d_blocks, BLOCK_SIZE>>>(p.x3, p.y3,
			p.z3, a, b);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		twenty_seven_pt_stencil_3d_work<<<512, 1024>>>(p.x3, p.y3, p.z3, ALPHA, BETA, DELTA, GAMMA, a, b);
#else
		twenty_seven_pt_stencil_3d<<<n3d_blocks, BLOCK_SIZE>>>(p.x3, p.y3, p.z3,
				ALPHA, BETA, DELTA, GAMMA, a, b);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 2 * p.x * p.y, "27pt_stencil_3d");

#if defined(RUN_VALIDATION_SUITE)
	validate_twenty_seven_pt_stencil_3d(
			p.x3, p.y3, p.z3, ALPHA, BETA, DELTA, GAMMA, a);
#endif

	const int nplanes = p.x3 + p.y3 - 1;
	const int nwave_blocks = ceil((double) (p.x3 * p.y3) / BLOCK_SIZE);
	init_five_pt_wavefront<<<nwave_blocks, BLOCK_SIZE>>>(p.x3, p.y3, p.z3, a);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
		for (int jj = 2; jj < nplanes - 2; ++jj) {
			const int offset = max(1, jj - p.y3 + 2);
			const int length = (min(p.z3 - 1, jj) - offset + 1);
			const int nblocks = ceil((double) (length * p.x3) / BLOCK_SIZE);
#if defined(RUN_DYNAMIC_WORK)
			five_pt_wavefront_work<<<512, 128>>>(p.x3, p.y3, p.z3, a);
#else
			five_pt_wavefront<<<nblocks, BLOCK_SIZE>>>(p.x3, p.y3, p.z3, a);
#endif
		}
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y, "5pt_wavefront");

#if defined(RUN_VALIDATION_SUITE)
	validate_five_pt_wavefront(p.x3, p.y3, p.z3, a);
#endif

	init_tealeaf_cheby_iter<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c, d, e,
			f, g);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		tealeaf_cheby_iter_work<<<512, 1024>>>(p.x, p.y, HALO_DEPTH, ALPHA, BETA, a, b, c, d, e, f, g);
#else
		tealeaf_cheby_iter<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, HALO_DEPTH,
				ALPHA, BETA, a, b, c, d, e, f, g);
#endif
	}
	POST_MEM_KERNEL(3 * p.x * p.y, 4 * p.x * p.y, "tealeaf_cheby_iter");

#if defined(RUN_VALIDATION_SUITE)
	validate_tealeaf_cheby_iter(p.x, p.y, HALO_DEPTH, g);
#endif

	init_compute_bound<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		compute_bound_work<<<512, 1024>>>(p.x, p.y, a, b, c);
#else
		compute_bound<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c);
#endif
	}
	POST_FLOP_KERNEL(256.0 * p.x * p.y, 2 * p.x * p.y, "compute_bound");

#if defined(RUN_VALIDATION_SUITE)
	validate_compute_bound(p.x, p.y, a);
#endif

	const int n2d_5wide_blocks = ceil((double) (5.0 * p.x * p.y) / BLOCK_SIZE);
	init_dense_mat_vec<<<n2d_5wide_blocks, BLOCK_SIZE>>>(p.x, p.y, p.a_l, f, g);

	const int ndense_2d_blocks = ceil((double) (p.y) / BLOCK_SIZE);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		dense_mat_vec_work<<<p.y/128, 128>>>(p.x, p.y, p.a_l, f, g);
#else
		dense_mat_vec<<<ndense_2d_blocks, BLOCK_SIZE>>>(p.x, p.y, p.a_l, f, g);
#endif
	}
	POST_MEM_KERNEL(5 * p.y, 5 * p.x * p.y, "dense_mat_vec");

#if defined(RUN_VALIDATION_SUITE)
	validate_dense_mat_vec(p.x, p.y, f);
#endif

	init_cloverleaf_energy_flux<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, a, b, c,
			d, e, f, a_s, b_s, c_s);
	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
#if defined(RUN_DYNAMIC_WORK)
		cloverleaf_energy_flux_work<<<512, 1024>>>(
				p.x, p.y, HALO_DEPTH, a, b, c, d, e, f, a_s);
#else
		cloverleaf_energy_flux<<<n2d_blocks, BLOCK_SIZE>>>(p.x, p.y, HALO_DEPTH,
				a, b, c, d, e, f, a_s);
#endif
	}
	POST_MEM_KERNEL(p.x * p.y, 5*(p.x-2*HALO_DEPTH)*(p.y-HALO_DEPTH),
			"clover_energy_flux");

#if defined(RUN_VALIDATION_SUITE)
	validate_cloverleaf_energy_flux(p.x, p.y, HALO_DEPTH, g);
#endif

	init_matrix_multiply<<<n2d_blocks, BLOCK_SIZE>>>(p.x3, p.y3, a, b, c);

	PRE_KERNEL();
	for (int tt = 0; tt < p.niters; ++tt) {
		matrix_multiply<<<n2d_blocks, BLOCK_SIZE>>>(p.x3, p.y3, a, b, c);
	}
	POST_MEM_KERNEL(p.x * p.y, p.x * p.y * (2 * p.x), "matrix_multiply");

#if defined(RUN_VALIDATION_SUITE)
	validate_matrix_multiply(p.x3, p.y3, a, b);
#endif

	cudaFree(a);
	cudaDeviceSynchronize();
	run_snap_sweep(
	SWEEPX, SWEEPY, NSWEEPS, SWEEPX + SWEEPY - 1, NGROUPS, NANG, NMOM, NDIMS,
	ALPHA, BETA, a_s, b_s, c_s);

}

/*
 * Simplified SNAP sweep kernel excluding negative flux fix up
 * and assuming time dependent
 */
void run_snap_sweep(const int x, const int y, const int nsweeps,
		const int nplanes, const int ngroups, const int nang, const int nmom,
		const int ndims, const double alpha, const double beta, double* a_s,
		double* b_s, double* c_s) {
	const int sweep_mods_ii[NSWEEPS] = { 1, 0, 1, 0 };
	const int sweep_mods_jj[NSWEEPS] = { 1, 1, 0, 0 };

	double* a;
	double* b;
	double* c;
	double* d;
	double* e;
	double* f;
	double* g;
	int* sweep_indices;
	cudaMalloc((void**) &a, A_LEN * sizeof(double));
	cudaMalloc((void**) &b, B_LEN * sizeof(double));
	cudaMalloc((void**) &c, C_LEN * sizeof(double));
	cudaMalloc((void**) &d, D_LEN * sizeof(double));
	cudaMalloc((void**) &e, E_LEN * sizeof(double));
	cudaMalloc((void**) &f, F_LEN * sizeof(double));
	cudaMalloc((void**) &g, G_LEN * sizeof(double));
	cudaMalloc((void**) &sweep_indices, sizeof(int) * x * y * ndims);
	cudaDeviceSynchronize();

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

	cudaMemcpy(sweep_indices, h_sweep_indices, sizeof(int) * x * y * ndims,
			cudaMemcpyHostToDevice);

	const int n2d_blocks = ceil((double) p.x * p.y / BLOCK_SIZE);
	init_snap_sweep<<<n2d_blocks, BLOCK_SIZE>>>(nplanes, p.x, p.y, a, b, c, d,
			e, f, g, a_s, b_s, c_s);
	PRE_KERNEL();

	for (int tt = 0; tt < p.niters; ++tt) {
		const int ss = tt % nsweeps;
		const int sweep_mod_ii = sweep_mods_ii[ss];
		const int sweep_mod_jj = sweep_mods_jj[ss];

		int offset = 0;
		for (int pp = 2; pp < nplanes - 2; ++pp) {
			const int plane_length = min(pp - 1, nplanes - pp - 2);
			const int data_in_plane = plane_length * nang * ngroups;
			const int nblocks = ceil((double) data_in_plane / BLOCK_SIZE);

#if defined(RUN_DYNAMIC_WORK)
			snap_sweep_work<<<512, 1024>>>(
					x, y, plane_length, ss, offset, sweep_mod_ii, sweep_mod_jj,
					sweep_indices, alpha, nmom, nang, ngroups, a, b, c, d, e, f, g,
					a_s, b_s, c_s);
#else
			snap_sweep<<<nblocks, BLOCK_SIZE>>>(x, y, plane_length, ss, offset,
					sweep_mod_ii, sweep_mod_jj, sweep_indices, alpha, nmom,
					nang, ngroups, a, b, c, d, e, f, g, a_s, b_s, c_s);
#endif

			offset += (plane_length * ndims);
		}
	}

	POST_MEM_KERNEL(0,
			(double)(A_LEN + B_LEN + C_LEN + D_LEN + E_LEN + F_LEN + G_LEN),
			"snap_sweep");

#if defined(RUN_VALIDATION_SUITE)
	validate_snap_sweep(x, y, g);
#endif

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	cudaFree(d);
	cudaFree(e);
	cudaFree(f);
	cudaFree(g);
}

// Allocates some device memory
void init(KernelParams* p) {
	const int nplanes = p->x3 + p->y3 - 1;
	p->ndims = NDIMS;

	int ndevices;
	cudaGetDeviceCount(&ndevices);

	int device_id = 0;

	int result = cudaSetDevice(device_id);
	if (result != cudaSuccess) {
		printf("Could not allocate CUDA device %d.\n", device_id);
		exit(1);
	}

	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device_id);

	printf("Using %s device id %d\n", properties.name, device_id);

	int* h_r_indirection = (int*) malloc(sizeof(int) * p->x);
	for (int jj = 0; jj < p->x; ++jj) {
		h_r_indirection[jj] = (p->x - jj - 1);
	}
	int* h_c_indirection = (int*) malloc(sizeof(int) * p->x);
	for (int ii = 0; ii < p->x; ++ii) {
		h_c_indirection[ii] = p->x;
	}

	int* h_sweep_indices = (int*) malloc(
			sizeof(int) * p->x3 * p->y3 * p->ndims);

	// Initialise the 3d sweep indirection lists for positive upwind quadrant
	int offset = 0;
	for (int jj = 2; jj < nplanes - 1; ++jj) {
		for (int ii = max(1, jj - p->y3 + 2); ii < min(p->z3 - 1, jj); ++ii) {
			// Skewed jj index
			h_sweep_indices[offset++] = ii;
			h_sweep_indices[offset++] = jj;
		}
	}

#if defined(RUN_VALIDATION_SUITE)
	p->niters = 1;
#endif

	const int buffer_length = max(p->x * p->y, p->x3 * p->y3 * p->z3);

	const double nbytes = sizeof(double) * 8 * buffer_length
			+ sizeof(int) * p->x * p->y * p->ndims;
	printf("Rough lower bound of total data: %.2lf GB\n", (double) nbytes / GB);

	const int small_buffer_length = max(p->x, p->y);

	cudaMalloc((void**) &p->a_l, 7 * buffer_length * sizeof(double));
	p->a = p->a_l;
	p->b = &(p->a_l[1 * buffer_length]);
	p->c = &(p->a_l[2 * buffer_length]);
	p->d = &(p->a_l[3 * buffer_length]);
	p->e = &(p->a_l[4 * buffer_length]);
	p->f = &(p->a_l[5 * buffer_length]);
	p->g = &(p->a_l[6 * buffer_length]);

	cudaMalloc((void**) &p->a_s, small_buffer_length * sizeof(double));
	cudaMalloc((void**) &p->b_s, small_buffer_length * sizeof(double));
	cudaMalloc((void**) &p->c_s, small_buffer_length * sizeof(double));
	cudaMalloc((void**) &p->d_s, small_buffer_length * sizeof(double));

	cudaMalloc((void**) &p->sweep_indices,
			sizeof(int) * p->x3 * p->y3 * p->ndims);
	cudaMalloc((void**) &p->c_indirection, sizeof(int) * p->x);
	cudaMalloc((void**) &p->r_indirection, sizeof(int) * p->x);
	cuda_check_errors(cudaGetLastError());
	cuda_check_errors(cudaDeviceSynchronize());

	const int nblocks = ceil((float) buffer_length / BLOCK_SIZE);
	const double init_value = 0.1;
	warmup<<<nblocks, BLOCK_SIZE>>>(buffer_length, init_value, p->a, p->b, p->c,
			p->d, p->e, p->f);
	cuda_check_errors(cudaDeviceSynchronize());

	cudaMemcpy(p->sweep_indices, h_sweep_indices,
			sizeof(int) * p->x3 * p->y3 * p->ndims, cudaMemcpyHostToDevice);
	cudaMemcpy(p->c_indirection, h_c_indirection, sizeof(int) * p->x,
			cudaMemcpyHostToDevice);
	cudaMemcpy(p->r_indirection, h_r_indirection, sizeof(int) * p->x,
			cudaMemcpyHostToDevice);
	cuda_check_errors(cudaDeviceSynchronize());
}

inline void gpuAssert(cudaError_t code, const char *file, int line,
		bool abort) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		if (abort)
			exit(code);
	}
}

void print_time_and_flops(const char* func_name, const int niters,
		const double nflops, const double nmem_access) {
	struct ProfileEntry entry = profiler_get_profile_entry(p.profile,
			func_name);
	printf("%s\t%.4lf\n", func_name, entry.time * 1000);
}

void print_time_and_mem_bw(const char* func_name, const int niters,
		const double nreads, const double nwrites) {
	struct ProfileEntry entry = profiler_get_profile_entry(p.profile,
			func_name);
	printf("%s\t%.4lf\n", func_name, entry.time * 1000);
}

void print_time(const char* func_name) {
	struct ProfileEntry entry = profiler_get_profile_entry(p.profile,
			func_name);
	printf("%s\t%.4lf\n", func_name, entry.time * 1000);
}


#include "../../rt.h"
#include "../main.h"
#include "snap.h"
#include "../timer.h"


#ifdef APP_SNAP
/*
 * Simplified SNAP sweep kernel excluding negative flux fix up
 * and assuming time dependent
 */
double run_snap_task(struct user_parameters* params) {

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



	/******************************************************************************/
	/* BEGIN: OPENARC GENERATED CODE ******************************************/
	/******************************************************************************/
	APP_CONTEXT appContext_h;
	APP_CONTEXT appContext_d;
	RT_CONTEXT rtContext_d;

//	// TODO: [OPEANARC] Compiler should inspect the tasks in the nested loops
//	// to come up with nTasks and nEdges values.
//	int nTasks = (itnew-itold) * // outermost loop (time iteration)
//			max_blocks_x * max_blocks_y * // inner loops
//			2; // there are two kernels
//
//	int nEdges = (5 + 1) * // 1st kernel has 1 and 2nd 5 depend "in" clauses
//			nTasks; // each task has those dependencies.
//
//	// Host related application initializations
//	initAppContext_H(appContext_h, nTasks, nEdges);
//	appContext_h.appData.blockSize = block_size;
//	appContext_h.appData.nx = nx;
//	appContext_h.appData.ny = ny;
//	appContext_h.appData.dx = dx;
//	appContext_h.appData.dy = dy;
//	appContext_h.appData.f = f_; // unnecessary assignment, but compiler needs
//						//to pass the arguments to app data if they are in the
//						// kernel scope
//	appContext_h.appData.u = u_;
//	appContext_h.appData.unew = unew_;
//
//	int noDependencies = 0;
//
//	// niter is assumed to be always 1
//	for (int tt = 0; tt < params->niter; ++tt) {
//		const int ss = tt % nsweeps;
//		const int sweep_mod_ii = sweep_mods_ii[ss];
//		const int sweep_mod_jj = sweep_mods_jj[ss];
//
//		int offset = 0;
//		for (int pp = 2; pp < nplanes - 2; ++pp) {
//			const int plane_length = min(pp - 1, nplanes - pp - 2);
//			const int data_in_plane = plane_length * nang * ngroups;
//			const int nblocks = ceil((double) data_in_plane / blockSize);
//
//			//TODO:
//			int taskIndex=addTask(appContext_h, KERNEL_TYPE_COPY_BLOCK);
//			appContext_h.tasks[taskIndex].taskData.block_x = block_x;
//			appContext_h.tasks[taskIndex].taskData.block_y = block_y;
//
//			if (noDependencies) // for debugging purposes
//				continue;
//			// First map "out" dependencies in taskmap to the current
//			// task( and possibly others) to easily back refer
//			// via an "in" dependency later.
//
//			// We create a dependency identifier to store the info in the
//			// OMP clause
//			// Original OMP clause:
//			// depend(out: u[block_x * block_size: block_size]
//            //				[block_y * block_size: block_size])
//
//			processOutDependency(appContext_h, taskIndex,
//					5,
//					(intptr_t)appContext_h.appData.u,
//					block_x * block_size,
//					block_size,
//					block_y * block_size,
//					block_size);
//
//			// Then process "in" dependencies to find the parent(s) of the
//			// current task.
//
//			// Original OMP clause for the in dependency range:
//			// depend(in: unew[block_x * block_size: block_size]
//			//					[block_y * block_size: block_size])
//
//			processInDependency(appContext_h, taskIndex,
//					5,
//					(intptr_t)appContext_h.appData.unew,
//					block_x * block_size,
//					block_size,
//					block_y * block_size,
//					block_size);
//			offset += (plane_length * ndims);
//
//		}
//	}

	START_TIMER
	//TODO:
//	snap_sweep_global1<<<nblocks, blockSize>>>(x, y, plane_length, ss, offset,
//			sweep_mod_ii, sweep_mod_jj, sweep_indices, alpha, nmom,
//			nang, ngroups, a, b, c, d, e, f, g, a_s, b_s, c_s);

	checkCudaErrors(cudaDeviceSynchronize());

	END_TIMER

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

	return TIMER;
}
void validate_snap_sweep(
    const int x, const int y, const double* d_g)
{
//  double* g = (double*)malloc(sizeof(double)*G_LEN);
//  cudaMemcpy(g, d_g, sizeof(double)*G_LEN, cudaMemcpyDeviceToHost);
//
//  /// Requires some validation procedure
//  double gval = 0.0;
//
//  for(int ii = 1; ii < y-1; ++ii)
//    for(int jj = 1; jj < x-1; ++jj)
//      for(int kk = 0; kk < NGROUPS; ++kk)
//        for(int ll = 0; ll < NANG; ++ll)
//          gval += g[ii*NANG*NGROUPS*x + jj*NGROUPS*NANG + kk*NANG + ll];
//
//  if(check_is_failed(0, gval, SNAP_VALIDATE))
//    printf("\t*\x1B[31mFailed\x1B[0m validation\n");
//  else
//    printf("\t*Passed validation\n");
}

#endif

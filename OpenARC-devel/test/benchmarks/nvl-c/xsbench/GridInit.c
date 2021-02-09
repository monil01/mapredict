#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

// Generates randomized energy grid for each nuclide
// Note that this is done as part of initialization (serial), so rand() is used.
void generate_grids(NVL_PREFIX double * nuclide_grids, long n_isotopes, long n_gridpoints)
{
	long i, j;
#if MEM == NVL
	double * nuclide_grids_v = nvl_bare_hack(nuclide_grids);
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 0] = ((double)rand()/(double)RAND_MAX); //energy
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 1] = ((double)rand()/(double)RAND_MAX); //total xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 2] = ((double)rand()/(double)RAND_MAX); //elastic xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 3] = ((double)rand()/(double)RAND_MAX); //absorption xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 4] = ((double)rand()/(double)RAND_MAX); //fission xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 5] = ((double)rand()/(double)RAND_MAX); //nu fission xs
		}
	}
#else
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			nuclide_grids[i*n_gridpoints*6 + j*6 + 0] = ((double)rand()/(double)RAND_MAX); //energy
			nuclide_grids[i*n_gridpoints*6 + j*6 + 1] = ((double)rand()/(double)RAND_MAX); //total xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 2] = ((double)rand()/(double)RAND_MAX); //elastic xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 3] = ((double)rand()/(double)RAND_MAX); //absorption xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 4] = ((double)rand()/(double)RAND_MAX); //fission xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 5] = ((double)rand()/(double)RAND_MAX); //nu fission xs
		}
	}
#endif
}

// Verification version of this function (tighter control over RNG)
void generate_grids_v(NVL_PREFIX double * nuclide_grids, long n_isotopes, long n_gridpoints)
{
	long i, j;
#if MEM == NVL
	double * nuclide_grids_v = nvl_bare_hack(nuclide_grids);
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 0] = rn_v(); //energy
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 1] = rn_v(); //total xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 2] = rn_v(); //elastic xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 3] = rn_v(); //absorption xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 4] = rn_v(); //fission xs
			nuclide_grids_v[i*n_gridpoints*6 + j*6 + 5] = rn_v(); //nu fission xs
		}
	}
#else
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			nuclide_grids[i*n_gridpoints*6 + j*6 + 0] = rn_v(); //energy
			nuclide_grids[i*n_gridpoints*6 + j*6 + 1] = rn_v(); //total xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 2] = rn_v(); //elastic xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 3] = rn_v(); //absorption xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 4] = rn_v(); //fission xs
			nuclide_grids[i*n_gridpoints*6 + j*6 + 5] = rn_v(); //nu fission xs
		}
	}
#endif
}

// Sorts the nuclide grids by energy (lowest -> highest)
void sort_nuclide_grids(NVL_PREFIX double * nuclide_grids, long n_isotopes, long n_gridpoints)
{
	long i, j;
	int (*cmp) (const void *, const void *);
	cmp = d_compare;

	for(i=0; i<n_isotopes; i++){
#if MEM == NVL
		qsort(nvl_bare_hack(nuclide_grids + i*n_gridpoints*6), n_gridpoints, sizeof(double)*6, cmp);
#else
		qsort(&nuclide_grids[i*n_gridpoints*6], n_gridpoints, sizeof(double)*6, cmp);
#endif
	}
}

// Allocates unionized energy grid, and assigns union of energy levels
// from nuclide grids to it.
NVL_PREFIX double * generate_energy_grid(long n_isotopes, long n_gridpoints, NVL_PREFIX double * nuclide_grids)
{
	long i, j, n_unionized_grid_points;
	int mype = 0;
	NVL_PREFIX double * energy_grid = 0;
	int (*cmp) (const void *, const void *);
#if MEM == NVL
	double * energy_grid_v, * nuclide_grids_v;
#endif
	cmp = d_compare;

	#ifdef MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif

	if(mype == 0) printf("Generating Unionized Energy Grid...\n");

	n_unionized_grid_points = n_isotopes*n_gridpoints;
#if MEM == NVL
	energy_grid = nvl_alloc_nv(heap, n_unionized_grid_points, double);
    if(!energy_grid) {
        perror("nvl_alloc_nv failed");
		exit(1);
    }
#elif MEM == VHEAP
	energy_grid = nvl_vmalloc(vheap, n_unionized_grid_points*sizeof(double));
    if(!energy_grid) {
        perror("nvl_vmalloc failed");
		exit(1);
    }
#else
	energy_grid = (double *)malloc(n_unionized_grid_points*sizeof(double));
#endif

	if(mype == 0) printf("Copying and sorting nuclide grid energies...\n");

#if MEM == NVL
	energy_grid_v = nvl_bare_hack(energy_grid);
	nuclide_grids_v = nvl_bare_hack(nuclide_grids);
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			energy_grid_v[i*n_gridpoints + j] = nuclide_grids_v[i*n_gridpoints*6 + j*6];
		}
	}
#else
	for(i=0; i<n_isotopes; i++){
		for(j=0; j<n_gridpoints; j++){
			energy_grid[i*n_gridpoints + j] = nuclide_grids[i*n_gridpoints*6 + j*6];
		}
	}
#endif
#if MEM == NVL
	qsort(nvl_bare_hack(energy_grid), n_unionized_grid_points, sizeof(double), cmp);
#else 
	qsort(energy_grid, n_unionized_grid_points, sizeof(double), cmp);
#endif

	return energy_grid;
}

// Searches each nuclide grid for the closest energy level and assigns
// pointer from unionized grid to the correct spot in the nuclide grid.
// This process is time consuming, as the number of binary searches
// required is:  binary searches = n_gridpoints * n_isotopes^2
NVL_PREFIX int * generate_grid_ptrs(long n_isotopes, long n_gridpoints, NVL_PREFIX double *nuclide_grids, NVL_PREFIX double * energy_grid)
{
	long i, j;
	double quarry;
	int mype = 0;
	NVL_PREFIX int * grid_ptrs = 0;
#if MEM == NVL
	int * grid_ptrs_v;
#endif

	#ifdef MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif

#if MEM == NVL
	grid_ptrs = nvl_alloc_nv(heap, n_isotopes*n_gridpoints*n_isotopes, int);
    if(!grid_ptrs) {
        perror("nvl_alloc_nv failed");
		exit(1);
    }
	grid_ptrs_v = nvl_bare_hack(grid_ptrs);
#elif MEM == VHEAP
	grid_ptrs = nvl_vmalloc(vheap, n_isotopes*n_gridpoints*n_isotopes*sizeof(int));
    if(!grid_ptrs) {
        perror("nvl_vmalloc failed");
		exit(1);
    }
#else
	grid_ptrs = (int *) malloc(n_isotopes*n_gridpoints*n_isotopes*sizeof(int));
#endif

	if(mype == 0) printf("Assigning pointers to Unionized Energy Grid...\n");

	#pragma omp parallel for default(none) \
	shared( energy_grid, grid_ptrs, nuclide_grids, n_isotopes, n_gridpoints, mype ) \
	private( quarry, i, j )
	for(i=0; i<n_isotopes*n_gridpoints; i++){
		quarry = energy_grid[i];
#if OMP == 1
		if(INFO && mype == 0 && omp_get_thread_num() == 0 && i % 200 == 0)
			printf("\rAligning Unionized Grid...(%.0lf%% complete)",
			       100.0 * (double) i / (n_isotopes*n_gridpoints /
				                         omp_get_num_threads())     );
#else
		if(INFO && mype == 0 && i % 200 == 0)
			printf("\rAligning Unionized Grid...(%.0lf%% complete)",
			       100.0 * (double) i / (n_isotopes*n_gridpoints /
				                        1)     );
#endif
		for(j=0; j<n_isotopes; j++){
			// j is the nuclide i.d.
			// log n binary search
#if MEM == NVL
			grid_ptrs_v[n_isotopes*i + j] = binary_search(nvl_bare_hack(nuclide_grids + j*n_gridpoints*6), quarry, n_gridpoints);
#else
			grid_ptrs[n_isotopes*i + j] = binary_search(&nuclide_grids[j*n_gridpoints*6], quarry, n_gridpoints);
#endif
		}
	}
	if(mype == 0) printf("\n");

	return grid_ptrs;
}

#include "XSbench_header.h"

#ifdef MPI
#include<mpi.h>
#endif

#if MEM == NVL
struct root {
	nvl double *nuclide_grids;
	nvl double *energy_grid;
	nvl int *grid_ptrs;
	int i;
	unsigned long long vhash;
}; 
#endif

#if MEM == NVL
nvl_heap_t *heap = 0;
#elif MEM == VHEAP
nvl_vheap_t *vheap = 0;
#endif

int main(int argc, char* argv[])
{
	// =====================================================================
	// Initialization & Command Line Read-In
	// =====================================================================
	int version = 13;
	int mype = 0;
#if OMP == 1
	int max_procs = omp_get_num_procs();
#else 
	int max_procs = 1;
#endif
	int thread, mat;
	unsigned long seed;
	double omp_start, omp_end, p_energy;
	int nprocs;
	double acc_start, acc_end;
#if MEM == NVL
	#define i (root_nv->i)
	#define vhash (root_nv->vhash)
#else
	int i = 0;
	unsigned long long vhash = 0;
#endif

  //Inputs
	int nthreads;
	long n_isotopes;
	long n_gridpoints;
	int lookups;
	char HM[6];

	NVL_PREFIX double *nuclide_grids = 0;
	NVL_PREFIX double *energy_grid = 0;
	NVL_PREFIX int *grid_ptrs = 0;
	int size_mats, *num_nucs, *mats_ptr, *mats;
	double *concs;
	int bench_n; // benchmark loop index
	double macro_xs_vector[5];
	char line[256]; // verification hash
	unsigned long long vhash_local; // verification hash

	#ifdef MPI
	MPI_Status stat;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &mype);
	#endif
	
	// rand() is only used in the serial initialization stages.
	// A custom RNG is used in parallel portions.
	#ifdef VERIFICATION
	srand(26);
	#else
	srand(time(NULL));
	#endif

	// Process CLI Fields -- store in "Inputs" structure
	read_CLI(argc, argv, &nthreads, &n_isotopes, &n_gridpoints, &lookups, HM);

	// Set number of OpenMP Threads
#if OMP == 1
	omp_set_num_threads(nthreads); 
#endif

	// Print-out of Input Summary
  if(mype == 0) print_inputs(nthreads, n_isotopes, n_gridpoints, lookups, HM, nprocs, version);

	// =====================================================================
	// Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
	// =====================================================================

	// Allocate & fill energy grids
	#ifndef BINARY_READ
	if(mype == 0) printf("Generating Nuclide Energy Grids...\n");
	#endif

#if MEM == NVL
	heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
	if(!heap) {
		perror("nvl_create failed");
		return 1;
	}
	nvl struct root *root_nv = 0;
	if(!(root_nv = nvl_alloc_nv(heap, 1, struct root))) {
		perror("nvl_alloc_nv failed");
		return 1;
	}
	nvl_set_root(heap, root_nv);
	nuclide_grids = nvl_alloc_nv(heap, n_isotopes *n_gridpoints * 6,double);
	if(!nuclide_grids) {
		perror("nvl_alloc_nv failed");
		return 1;
	}
	root_nv->nuclide_grids = nuclide_grids;
#elif MEM == VHEAP
	vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
	if(!vheap) {
		perror("nvl_vcrate failed");
		return 1;
	}
	nuclide_grids = nvl_vmalloc(vheap, n_isotopes *n_gridpoints * 6 * sizeof(double));
	if(!nuclide_grids) {
		perror("nvl_vmalloc failed");
		return 1;
	}
#else
	nuclide_grids = (double *) malloc(n_isotopes *n_gridpoints * 6 * sizeof(double));
#endif
	
	#ifdef VERIFICATION
	generate_grids_v(nuclide_grids,n_isotopes,n_gridpoints);	
	#else
	generate_grids(nuclide_grids,n_isotopes,n_gridpoints);	
	#endif

	// Sort grids by energy
	#ifndef BINARY_READ
	if(mype == 0) printf("Sorting Nuclide Energy Grids...\n");
	sort_nuclide_grids(nuclide_grids,n_isotopes,n_gridpoints);
	#endif

	// Prepare Unionized Energy Grid Framework
	// Double Indexing. Filling in energy_grid with pointers to the
	// nuclide_energy_grids.
	#ifndef BINARY_READ
	energy_grid = generate_energy_grid(n_isotopes,n_gridpoints, nuclide_grids);
	grid_ptrs = generate_grid_ptrs(n_isotopes,n_gridpoints, nuclide_grids, energy_grid);	
	#else
#if MEM == NVL
	energy_grid = nvl_alloc_nv(heap, n_isotopes*n_gridpoints, double);
	grid_ptrs = nvl_alloc_nv(heap, n_isotopes*n_gridpoints*n_isotopes, int);
	if(!energy_grid || !grid_ptrs) {
		perror("nvl_alloc_nv failed");
		return 1;
	}
#elif MEM == VHEAP
	energy_grid = nvl_vmalloc(vheap, n_isotopes*n_gridpoints*sizeof(double));
	grid_ptrs = nvl_vmalloc(vheap, n_isotopes*n_gridpoints*n_isotopes*sizeof(int));
	if(!energy_grid || !grid_ptrs) {
		perror("nvl_vmalloc failed");
		return 1;
	}
#else
	energy_grid = malloc(n_isotopes*n_gridpoints*sizeof(double));
	grid_ptrs = (int *) malloc(n_isotopes*n_gridpoints*n_isotopes*sizeof(int));
#endif
	#endif

	#ifdef BINARY_READ
	if(mype == 0) printf("Reading data from \"XS_data.dat\" file...\n");
#if MEM == NVL
	binary_read(n_isotopes,n_gridpoints, nvl_bare_hack(nuclide_grids), nvl_bare_hack(energy_grid), nvl_bare_hack(grid_ptrs));
#else
	binary_read(n_isotopes,n_gridpoints, nuclide_grids, energy_grid, grid_ptrs);
#endif
	#endif
	
	// Get material data
	if(mype == 0) printf("Loading Mats...\n");
	if(n_isotopes == 68) size_mats = 197;
	else size_mats = 484;
	num_nucs  = load_num_nucs(n_isotopes);
	mats_ptr  = load_mats_ptr(num_nucs);
	mats      = load_mats(num_nucs, mats_ptr, size_mats,n_isotopes);

	#ifdef VERIFICATION
	concs = load_concs_v(size_mats);
	#else
	concs = load_concs(size_mats);
	#endif

	#ifdef BINARY_DUMP
	if(mype == 0) printf("Dumping data to binary file...\n");
#if MEM == NVL
	binary_dump(n_isotopes,n_gridpoints, nvl_bare_hack(nuclide_grids), nvl_bare_hack(energy_grid), nvl_bare_hack(grid_ptrs));
#else
	binary_dump(n_isotopes,n_gridpoints, nuclide_grids, energy_grid, grid_ptrs);
#endif
	if(mype == 0) printf("Binary file \"XS_data.dat\" written! Exiting...\n");
	return 0;
	#endif

	// =====================================================================
	// Cross Section (XS) Parallel Lookup Simulation Begins
	// =====================================================================

	// Outer benchmark loop can loop through all possible # of threads
	#if defined(BENCHMARK) && (OMP == 1) 
	for(bench_n = 1; bench_n <=omp_get_num_procs(); bench_n++)
	{
		nthreads = bench_n;
		omp_set_num_threads(nthreads);
 	#endif

	if(mype == 0)
	{
		printf("\n");
		border_print();
		center_print("SIMULATION", 79);
		border_print();
	}

	#ifndef OPENACC
	#if OMP == 1
	omp_start = omp_get_wtime();
	#else
	omp_start = timer();
	#endif
	#else
	acc_start = timer();
	#endif

	//initialize papi with one thread (master) here
	#ifdef PAPI
	if ( PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT){
		fprintf(stderr, "PAPI library init error!\n");
		exit(1);
	}
	#endif	

	#ifndef OPENACC
	#pragma omp parallel default(none) \
	private(i, thread, p_energy, mat, seed, vhash_local, line, macro_xs_vector) \
	shared( max_procs, nthreads, n_isotopes, n_gridpoints, lookups, HM, energy_grid, \
      nuclide_grids, grid_ptrs, mats_ptr, mats, concs, num_nucs, mype, vhash) 
	#else
	#pragma acc data \
	copy(vhash) \
	copyin(lookups, n_isotopes, n_gridpoints, \
         num_nucs[0:n_isotopes], concs[0:size_mats], mats[0:size_mats], mats_ptr[0:12], \
	       energy_grid[0:n_isotopes*n_gridpoints], \
	       grid_ptrs[0:n_isotopes*n_isotopes*n_gridpoints], \
	       nuclide_grids[0:n_isotopes*n_gridpoints*6])
	#endif
	{
		// Initialize parallel PAPI counters
		#ifdef PAPI
		int eventset = PAPI_NULL; 
		int num_papi_events;
		#pragma omp critical
		{
			counter_init(&eventset, &num_papi_events);
		}
		#endif
	
		#ifndef OPENACC
		#if OMP == 1
		thread = omp_get_thread_num();
		#else
		thread = 0;
		#endif
		seed   = (thread+1)*19+17;
		#else
		seed = 13; //what to do for openacc?
		#endif

		// XS Lookup Loop
		#ifndef OPENACC
		#pragma omp for schedule(dynamic)
		#else
		#pragma acc parallel loop independent \
		firstprivate(seed) \
		private(macro_xs_vector, p_energy, mat, vhash_local, line)
		#endif
		#if TXS
		while(i<lookups)
		{
		  #pragma nvl atomic heap(heap)
		  for(int i_sub=0; i_sub<ITERS_PER_TX && i<lookups;
		      ++i_sub, ++i)
		  {
		#else
		for (; i<lookups; ++i) {
		#endif
			#ifndef OPENACC
			// Status text
			if( INFO && mype == 0 && thread == 0 && i % 1000 == 0 )
				printf("\rCalculating XS's... (%.0lf%% completed)",
						(i / ( (double)lookups / (double)nthreads ))
						/ (double)nthreads * 100.0);
			#endif

			// Randomly pick an energy and material for the particle
			#ifdef VERIFICATION
			#ifndef OPENACC
			#pragma omp critical
			#endif
			{
				mat = pick_mat(&seed); 
				p_energy = rn_v();
			}
			#else
			mat = pick_mat(&seed); 
			p_energy = rn(&seed);
			#endif
		
			// This returns the macro_xs_vector, but we're not going
			// to do anything with it in this program, so return value
			// is written over.
			calculate_macro_xs(p_energy, mat, n_isotopes, n_gridpoints,
					   num_nucs, concs, energy_grid, nuclide_grids,
					   grid_ptrs, mats, mats_ptr, macro_xs_vector);

			// Verification hash calculation
			// This method provides a consistent hash accross
			// architectures and compilers.
			#ifdef VERIFICATION
			sprintf(line, "%.5lf %d %.5lf %.5lf %.5lf %.5lf %.5lf",
			       p_energy, mat,
				   macro_xs_vector[0],
				   macro_xs_vector[1],
				   macro_xs_vector[2],
				   macro_xs_vector[3],
				   macro_xs_vector[4]);
			vhash_local = hash((unsigned char *)line, 10000);
			#ifndef OPENACC
			#pragma omp atomic
			#endif
			vhash += vhash_local;
			#endif
		#if TXS
		  }
		#endif
		}

		// Prints out thread local PAPI counters
		#ifdef PAPI
		if( mype == 0 && thread == 0 )
		{
			printf("\n");
			border_print();
			center_print("PAPI COUNTER RESULTS", 79);
			border_print();
			printf("Count          \tSmybol      \tDescription\n");
		}
		{
		#pragma omp barrier
		}
		counter_stop(&eventset, num_papi_events);
		#endif
	}

	#ifndef PAPI
	if( mype == 0) printf("\nSimulation complete.\n" );
	#endif

	#ifndef OPENACC
	#if OMP == 1
	omp_end = omp_get_wtime();
	#else
	omp_end = timer();
	#endif
	print_results(nthreads, n_isotopes, n_gridpoints, lookups, HM, mype, omp_end-omp_start, nprocs, vhash);
	#else
	acc_end = timer();
	print_results(nthreads, n_isotopes, n_gridpoints, lookups, HM, mype, acc_end-acc_start, nprocs, vhash);
	#endif

	#if defined(BENCHMARK) && (OMP == 1)
	}
	#endif

	#ifdef MPI
	MPI_Finalize();
	#endif

#if MEM == NVL
	nvl_close(heap);
#elif MEM == VHEAP
	nvl_vfree(vheap, nuclide_grids);
	nvl_vfree(vheap, energy_grid);
	nvl_vfree(vheap, grid_ptrs);
	nvl_vclose(vheap);
#else
	free(nuclide_grids);
	free(energy_grid);
	free(grid_ptrs);
#endif

	return 0;
}

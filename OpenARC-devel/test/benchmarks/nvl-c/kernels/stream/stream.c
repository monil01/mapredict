/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.9 2009/04/11 16:35:00 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#include <assert.h>
# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <float.h>
# include <limits.h>
# include <sys/time.h>
#define NVL 1
#define VHEAP 2
#if MEM == NVL
#include <nvl.h>
#elif MEM == VHEAP
#include <nvl-vheap.h>
#else
# error unknown MEM setting
#endif

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/stream.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/stream.nvl"
//#define NVLFILE "/tmp/f6l/stream.nvl"
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

#ifndef STATIC_CPU_MALLOC
#define STATIC_CPU_MALLOC 0
#endif

/* INSTRUCTIONS:
 *
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of 
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

#ifndef N
#define N	100000
#endif
#ifndef NTIMES
#define NTIMES	10
#endif
#ifndef OFFSET
#define OFFSET	0
#endif

/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

static double	a[N+OFFSET],
		b[N+OFFSET],
		c[N+OFFSET];

static double	avgtime[4] = {0}, maxtime[4] = {0},
		mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char	*label[4] = {"Copy:      ", "Scale:     ",
    "Add:       ", "Triad:     "};

static double	bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

extern double mysecond();
extern void checkSTREAMresults(double *a, double *b, double *c);
#ifdef TUNED
extern void tuned_STREAM_Copy();
extern void tuned_STREAM_Scale(double scalar);
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Triad(double scalar);
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif

#if MEM == NVL
struct root {
#if TXS
	int j1;
	int j2;
	int j3;
	int j4;
#endif
	nvl double *a;
	nvl double *b;
	nvl double *c;
};
#endif

#if MEM == NVL
nvl double *a_nv = 0, *b_nv = 0, *c_nv = 0;
nvl_heap_t *heap = 0;
#if TXS
nvl int *j1_nv = 0;
nvl int *j2_nv = 0;
nvl int *j3_nv = 0;
nvl int *j4_nv = 0;
#endif
#elif MEM == VHEAP
double *a_nv = 0, *b_nv = 0, *c_nv = 0;
#endif
#if MEM == NVL
double *a_v, *b_v, *c_v;
#endif
#if STATIC_CPU_MALLOC == 1
double a_CPU[N+OFFSET];
double b_CPU[N+OFFSET];
double c_CPU[N+OFFSET];
#else
double *a_CPU, *b_CPU, *c_CPU;
#endif

int
main()
    {
    int			quantum, checktick();
    int			BytesPerWord;
    register int	j, k;
    double		scalar, t, times[4][NTIMES];

#if MEM == NVL
	heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
    if(!heap) {
        fprintf(stderr, "file %s already exists\n", NVLFILE);
        return 1;
    }   
    nvl struct root *root_nv = 0;
    if( !(root_nv = nvl_alloc_nv(heap, 1, struct root)) 
        || !(a_nv = nvl_alloc_nv(heap, (N + OFFSET), nvl double))
        || !(b_nv = nvl_alloc_nv(heap, (N + OFFSET), nvl double))
        || !(c_nv = nvl_alloc_nv(heap, (N + OFFSET), nvl double)) )
    {   
        perror("nvl_alloc_nv failed");
        return 1;   
    }   
    nvl_set_root(heap, root_nv);
#if TXS
	j1_nv = &root_nv->j1;
	j2_nv = &root_nv->j2;
	j3_nv = &root_nv->j3;
	j4_nv = &root_nv->j4;
#endif
    root_nv->a = a_nv;
    root_nv->b = b_nv;
    root_nv->c = c_nv;
#elif MEM == VHEAP
    nvl_vheap_t *vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
    if(!vheap) {
        perror("nvl_vcreate_failed");
        return 1;
    }   
    if( !(a_nv = (double *)nvl_vmalloc(vheap, (N+OFFSET)*sizeof(double)))
            || !(b_nv = (double *)nvl_vmalloc(vheap, (N+OFFSET)*sizeof(double)))
            || !(c_nv = (double *)nvl_vmalloc(vheap, (N+OFFSET)*sizeof(double))) )
    {   
        perror("nvl_vmalloc failed");
        return 1;   
	}   
#else
# error unknown MEM setting
#endif
#if STATIC_CPU_MALLOC != 1
	a_CPU = (double *)malloc((N+OFFSET)*sizeof(double));
	b_CPU = (double *)malloc((N+OFFSET)*sizeof(double));
	c_CPU = (double *)malloc((N+OFFSET)*sizeof(double));
#endif


    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.9 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(double);
    printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
	BytesPerWord);

    printf(HLINE);
#ifdef NO_LONG_LONG
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
#else
    printf("Array size = %llu, Offset = %d\n", (unsigned long long) N, OFFSET);
#endif

    printf("Total memory required = %.1f MB.\n",
	(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel 
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }
#endif

    printf(HLINE);
#pragma omp parallel
    {
    printf ("Printing one line per active thread....\n");
    }

	{
#if MEM == NVL && PERSIST
    // The initialization loops aren't part of our timings, but they're
    // outrageously slow if persists are inserted within.
	double *a_nv_bare = nvl_bare_hack(a_nv); 
	double *b_nv_bare = nvl_bare_hack(b_nv); 
	double *c_nv_bare = nvl_bare_hack(c_nv); 
	double *a_nv = a_nv_bare; 
	double *b_nv = b_nv_bare; 
	double *c_nv = c_nv_bare; 
#endif
    
    /* Get initial value for system clock. */
#pragma omp parallel for
    for (j=0; j<N; j++) {
	a_nv[j] = 1.0;
	b_nv[j] = 2.0;
	c_nv[j] = 0.0;
	a_CPU[j] = 1.0;
	b_CPU[j] = 2.0;
	c_CPU[j] = 0.0;
	}

    printf(HLINE);

    if  ( (quantum = checktick()) >= 1) 
	printf("Your clock granularity/precision appears to be "
	    "%d microseconds.\n", quantum);
    else {
	printf("Your clock granularity appears to be "
	    "less than one microsecond.\n");
	quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for (j = 0; j < N; j++) {
		a_nv[j] = 2.0E0 * a_nv[j];
	}
    t = 1.0E6 * (mysecond() - t);
#pragma omp parallel for
    for (j = 0; j < N; j++) {
		a_CPU[j] = 2.0E0 * a_CPU[j];
	}
	}

    printf("Each test below will take on the order"
	" of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

#if (MEM == NVL) && (!POOR)
    a_v = nvl_bare_hack(a_nv);
    b_v = nvl_bare_hack(b_nv);
    c_v = nvl_bare_hack(c_nv);
#endif
    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Copy();
#else
#if TXS
	assert(N % ROWS_PER_TX == 0);
	for (j=*j1_nv; j<N; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j1_nv[0:1], c_nv[j:ROWS_PER_TX])
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j1_nv[0:1]) clobber(c_nv[j:ROWS_PER_TX])
#endif
	for (int j_sub=0; j_sub<ROWS_PER_TX; ++j_sub, ++j, ++*j1_nv) {
		if( *j1_nv == N-1 ) { *j1_nv = -1; }
#else
#pragma omp parallel for
	for (j=0; j<N; j++) {
#endif
#if (MEM == VHEAP) || POOR
	    c_nv[j] = a_nv[j];
#else
#if TXS
	    c_nv[j] = a_v[j];
#else
	    c_v[j] = a_v[j];
#endif
#endif
#if TXS
	}
#endif
	}
#endif
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Scale(scalar);
#else
#if TXS
	assert(N % ROWS_PER_TX == 0);
	for (j=*j2_nv; j<N; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j2_nv[0:1], b_nv[j:ROWS_PER_TX])
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j2_nv[0:1]) clobber(b_nv[j:ROWS_PER_TX])
#endif
	for (int j_sub=0; j_sub<ROWS_PER_TX; ++j_sub, ++j, ++*j2_nv) {
		if( *j2_nv == N-1 ) { *j2_nv = -1; }
#else
#pragma omp parallel for
	for (j=0; j<N; j++) {
#endif
#if (MEM == VHEAP) || POOR
	    b_nv[j] = scalar*c_nv[j];
#else
#if TXS
	    b_nv[j] = scalar*c_v[j];
#else
	    b_v[j] = scalar*c_v[j];
#endif
#endif
#if TXS
	}
#endif
	}
#endif
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Add();
#else
#if TXS
	assert(N % ROWS_PER_TX == 0);
	for (j=*j3_nv; j<N; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j3_nv[0:1], c_nv[j:ROWS_PER_TX])
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j3_nv[0:1]) clobber(c_nv[j:ROWS_PER_TX])
#endif
	for (int j_sub=0; j_sub<ROWS_PER_TX; ++j_sub, ++j, ++*j3_nv) {
		if( *j3_nv == N-1 ) { *j3_nv = -1; }
#else
#pragma omp parallel for
	for (j=0; j<N; j++) {
#endif
#if (MEM == VHEAP) || POOR
	    c_nv[j] = a_nv[j]+b_nv[j];
#else
#if TXS
	    c_nv[j] = a_v[j]+b_v[j];
#else
	    c_v[j] = a_v[j]+b_v[j];
#endif
#endif
#if TXS
	}
#endif
	}
#endif
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
#ifdef TUNED
        tuned_STREAM_Triad(scalar);
#else
#if TXS
	assert(N % ROWS_PER_TX == 0);
	for (j=*j4_nv; j<N; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j4_nv[0:1], a_nv[j:ROWS_PER_TX])
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j4_nv[0:1]) clobber(a_nv[j:ROWS_PER_TX])
#endif
	for (int j_sub=0; j_sub<ROWS_PER_TX; ++j_sub, ++j, ++*j4_nv) {
		if( *j4_nv == N-1 ) { *j4_nv = -1; }
#else
#pragma omp parallel for
	for (j=0; j<N; j++) {
#endif
#if (MEM == VHEAP) || POOR
	    a_nv[j] = b_nv[j]+scalar*c_nv[j];
#else
#if TXS
	    a_nv[j] = b_v[j]+scalar*c_v[j];
#else
	    a_v[j] = b_v[j]+scalar*c_v[j];
#endif
#endif
#if TXS
	}
#endif
	}
#endif
	times[3][k] = mysecond() - times[3][k];
	}
#if (MEM == NVL) && !POOR && PERSIST
    nvl_persist_hack(a_nv, N+OFFSET);
    nvl_persist_hack(b_nv, N+OFFSET);
    nvl_persist_hack(c_nv, N+OFFSET);
#endif

    /*	--- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j]/(double)(NTIMES-1);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
#if (MEM == NVL) 
    a_v = nvl_bare_hack(a_nv);
    b_v = nvl_bare_hack(b_nv);
    c_v = nvl_bare_hack(c_nv);
    checkSTREAMresults(a_v, b_v, c_v);
#else
    checkSTREAMresults(a_nv, b_nv, c_nv);
#endif
    printf(HLINE);

#if STATIC_CPU_MALLOC == 1
    printf("\nReference CPU Performance (static malloc)\n");
#else
    printf("\nReference CPU Performance (dynamic malloc)\n");
#endif
    scalar = 3.0;
    for (k=0; k<NTIMES; k++)
	{
	times[0][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<N; j++) {
	    c_CPU[j] = a_CPU[j];
	}
	times[0][k] = mysecond() - times[0][k];
	
	times[1][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<N; j++) {
	    b_CPU[j] = scalar*c_CPU[j];
	}
	times[1][k] = mysecond() - times[1][k];
	
	times[2][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<N; j++) {
	    c_CPU[j] = a_CPU[j]+b_CPU[j];
	}
	times[2][k] = mysecond() - times[2][k];
	
	times[3][k] = mysecond();
#pragma omp parallel for
	for (j=0; j<N; j++) {
	    a_CPU[j] = b_CPU[j]+scalar*c_CPU[j];
	}
	times[3][k] = mysecond() - times[3][k];
	}

    /*	--- SUMMARY --- */
	//Seset ref times.
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = 0.0;
	    mintime[j] = FLT_MAX;
	    maxtime[j] = 0.0;
	    }

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<4; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}
    
    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j]/(double)(NTIMES-1);

	printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults(a_CPU, b_CPU, c_CPU);
    printf(HLINE);

#if MEM == NVL
    nvl_close(heap);
#elif MEM == VHEAP
    nvl_vfree(vheap, a_nv);
    nvl_vfree(vheap, b_nv);
    nvl_vfree(vheap, c_nv);
    nvl_vclose(vheap);
#endif
#if STATIC_CPU_MALLOC != 1
	free(a_CPU);
	free(b_CPU);
	free(c_CPU);
#endif

    return 0;
}

# define	M	20

int
checktick()
    {
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++) {
	t1 = mysecond();
	while( ((t2=mysecond()) - t1) < 1.0E-6 )
	    ;
	timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
	Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
	minDelta = MIN(minDelta, MAX(Delta,0));
	}

   return(minDelta);
    }



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults (double *a, double *b, double *c)
{
	double aj,bj,cj,scalar;
	double asum,bsum,csum;
	double epsilon;
	int	j,k;

    /* reproduce initialization */
	aj = 1.0;
	bj = 2.0;
	cj = 0.0;
    /* a[] is modified during timing check */
	aj = 2.0E0 * aj;
    /* now execute timing loop */
	scalar = 3.0;
	for (k=0; k<NTIMES; k++)
        {
            cj = aj;
            bj = scalar*cj;
            cj = aj+bj;
            aj = bj+scalar*cj;
        }
	aj = aj * (double) (N);
	bj = bj * (double) (N);
	cj = cj * (double) (N);

	asum = 0.0;
	bsum = 0.0;
	csum = 0.0;
	for (j=0; j<N; j++) {
		asum += a[j];
		bsum += b[j];
		csum += c[j];
	}
#ifdef VERBOSE
	printf ("Results Comparison: \n");
	printf ("        Expected  : %f %f %f \n",aj,bj,cj);
	printf ("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
	epsilon = 1.e-8;

	if (abs(aj-asum)/asum > epsilon) {
		printf ("Failed Validation on array a[]\n");
		printf ("        Expected  : %f \n",aj);
		printf ("        Observed  : %f \n",asum);
	}
	else if (abs(bj-bsum)/bsum > epsilon) {
		printf ("Failed Validation on array b[]\n");
		printf ("        Expected  : %f \n",bj);
		printf ("        Observed  : %f \n",bsum);
	}
	else if (abs(cj-csum)/csum > epsilon) {
		printf ("Failed Validation on array c[]\n");
		printf ("        Expected  : %f \n",cj);
		printf ("        Observed  : %f \n",csum);
	}
	else {
		printf ("Solution Validates\n");
	}
}

void tuned_STREAM_Copy()
{
	int j;
#if (MEM == NVL) && (!POOR)
    a_v = nvl_bare_hack(a_nv);
    c_v = nvl_bare_hack(c_nv);
#endif
#if TXS
	assert(N % ROWS_PER_TX == 0);
	for (j=*j1_nv; j<N; ) {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j1_nv[0:1], c_nv[j:ROWS_PER_TX])
#else
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(j1_nv[0:1]) clobber(c_nv[j:ROWS_PER_TX])
#endif
	for (int j_sub=0; j_sub<ROWS_PER_TX; ++j_sub, ++j, ++j1_nv) {
#else
	#pragma omp parallel for
	for (j=0; j<N; j++) {
#endif
#if (MEM == VHEAP) || POOR
            c_nv[j] = a_nv[j];
#else
#if TXS
            c_nv[j] = a_v[j];
#else
            c_v[j] = a_v[j];
#endif
#endif
#if TXS
	}
#endif
	}
}

void tuned_STREAM_Scale(double scalar)
{
	int j;
#if (MEM == NVL) && (!POOR)
    b_v = nvl_bare_hack(b_nv);
    c_v = nvl_bare_hack(c_nv);
#endif
#pragma omp parallel for
	for (j=0; j<N; j++) {
#if (MEM == VHEAP) || POOR
	    b_nv[j] = scalar*c_nv[j];
#else
	    b_v[j] = scalar*c_v[j];
#endif
	}
}

void tuned_STREAM_Add()
{
	int j;
#if (MEM == NVL) && (!POOR)
    a_v = nvl_bare_hack(a_nv);
    b_v = nvl_bare_hack(b_nv);
    c_v = nvl_bare_hack(c_nv);
#endif
#pragma omp parallel for
	for (j=0; j<N; j++) {
#if (MEM == VHEAP) || POOR
	    c_nv[j] = a_nv[j]+b_nv[j];
#else
	    c_v[j] = a_v[j]+b_v[j];
#endif
	}
}

void tuned_STREAM_Triad(double scalar)
{
	int j;
#if (MEM == NVL) && (!POOR)
    a_v = nvl_bare_hack(a_nv);
    b_v = nvl_bare_hack(b_nv);
    c_v = nvl_bare_hack(c_nv);
#endif
#pragma omp parallel for
	for (j=0; j<N; j++) {
#if (MEM == VHEAP) || POOR
	    a_nv[j] = b_nv[j]+scalar*c_nv[j];
#else
	    a_v[j] = b_v[j]+scalar*c_v[j];
#endif
	}
}


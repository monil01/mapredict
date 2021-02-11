//////////////////////////////////////////////////////////////////////
// This header file contains macros used for fault-injection tests. //
// These macros may need to be customized for each benchmark.       //
// (Any undefined macro will be evaluated to 0.)                    //
//////////////////////////////////////////////////////////////////////

//Decide target GPU thread to inject faults.
#if _FTTHREAD == 1
#pragma openarc #define FTTHREAD ftthread(0)
#else
#pragma openarc #define FTTHREAD 
#endif

//Decide at which resilience region fault will be injected.
//If multiple resilience regions exist, users may have to create
//multiple macros to control each region separately.
#if RES_REGION0 ==   0
#pragma openarc #define RES_REGION0 0
#else
#pragma openarc #define RES_REGION0 1
#endif

//Decide variables to inject faults
#if _FTVAR == 0
#pragma openarc #define FTVAR0	feature[0:_NPOINTS][0:_NATTRIBUTES]
#pragma openarc #define FTCNT0 33754207193
#elif _FTVAR == 1
#pragma openarc #define FTVAR0	membership[0:_NPOINTS]
#pragma openarc #define FTCNT0 33754207193
#elif _FTVAR == 2
#pragma openarc #define FTVAR0	clusters[0:_NCLUSTERS][0:_NATTRIBUTES]
#pragma openarc #define FTCNT0 33754207193
#elif _FTVAR == 3
#pragma openarc #define FTVAR0	partial_new_centers[0:_NTHREADS][0:_NCLUSTERS][0:_NATTRIBUTES]
#pragma openarc #define FTCNT0 33754207193
#else
#pragma openarc #define FTVAR0	partial_new_centers_len[0:_NTHREADS][0:_NCLUSTERS]
#pragma openarc #define FTCNT0 33754207193
#endif

//Declare fault types to inject
#if _FTKIND == 0
#pragma openarc #define FTKIND integer_arg
#elif _FTKIND == 1
#pragma openarc #define FTKIND integer_res
#elif _FTKIND == 2
#pragma openarc #define FTKIND floating_arg
#elif _FTKIND == 3
#pragma openarc #define FTKIND floating_res
#elif _FTKIND == 4
#pragma openarc #define FTKIND arithmetic_arg
#elif _FTKIND == 5
#pragma openarc #define FTKIND arithmetic_res
#elif _FTKIND == 6
#pragma openarc #define FTKIND pointer_arg
#elif _FTKIND == 7
#pragma openarc #define FTKIND pointer_res
#endif

#if R_MODE == 5

#if _FTKIND == 0
#pragma openarc #define FTCNT0 19817090118
#elif _FTKIND == 1
#pragma openarc #define FTCNT0 11984104480
#elif _FTKIND == 2
#pragma openarc #define FTCNT0 6421192222
#elif _FTKIND == 3
#pragma openarc #define FTCNT0 6397343743
#elif _FTKIND == 4
#pragma openarc #define FTCNT0 26238282340
#elif _FTKIND == 5
#pragma openarc #define FTCNT0 18381448223
#elif _FTKIND == 6
#pragma openarc #define FTCNT0 25780927355
#else
#pragma openarc #define FTCNT0 11938201600
#endif

#endif

//Decide total number of faults to be injected.
#if TOTAL_NUM_FAULTS == 0
#pragma openarc #define TOTAL_NUM_FAULTS    0
#elif TOTAL_NUM_FAULTS == 1
#pragma openarc #define TOTAL_NUM_FAULTS    1
#elif TOTAL_NUM_FAULTS == 2
#pragma openarc #define TOTAL_NUM_FAULTS    2
#elif TOTAL_NUM_FAULTS == 4
#pragma openarc #define TOTAL_NUM_FAULTS    4
#elif TOTAL_NUM_FAULTS == 8
#pragma openarc #define TOTAL_NUM_FAULTS    8
#elif TOTAL_NUM_FAULTS == 16
#pragma openarc #define TOTAL_NUM_FAULTS    16
#elif TOTAL_NUM_FAULTS == 128
#pragma openarc #define TOTAL_NUM_FAULTS    128
#elif TOTAL_NUM_FAULTS == 1024
#pragma openarc #define TOTAL_NUM_FAULTS    1024
#else
#pragma openarc #define TOTAL_NUM_FAULTS    1
#endif

//Decide total number of faulty bits to be changed per fault injection.
#if NUM_FAULTYBITS ==   0
#pragma openarc #define NUM_FAULTYBITS  0
#elif NUM_FAULTYBITS ==   1
#pragma openarc #define NUM_FAULTYBITS  1
#elif NUM_FAULTYBITS == 2
#pragma openarc #define NUM_FAULTYBITS  2
#elif NUM_FAULTYBITS == 4
#pragma openarc #define NUM_FAULTYBITS  4
#elif NUM_FAULTYBITS == 8
#pragma openarc #define NUM_FAULTYBITS  8
#elif NUM_FAULTYBITS == 16
#pragma openarc #define NUM_FAULTYBITS  16
#else
#pragma openarc #define NUM_FAULTYBITS  1
#endif

//Decide number of repeating of a target kernel; with this, the kernel
//execution will be repeated as specified in the clause.
#if NUM_REPEATS ==   0
#pragma openarc #define NUM_REPEATS 0
#elif NUM_REPEATS ==   1
#pragma openarc #define NUM_REPEATS 1
#elif NUM_REPEATS ==   128
#pragma openarc #define NUM_REPEATS 128
#elif NUM_REPEATS ==   1024
#pragma openarc #define NUM_REPEATS 1024
#else
#pragma openarc #define NUM_REPEATS 1
#endif



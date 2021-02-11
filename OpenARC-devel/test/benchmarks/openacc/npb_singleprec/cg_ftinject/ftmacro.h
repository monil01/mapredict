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
#if RES_REGION1 ==   0
#pragma openarc #define RES_REGION1 0
#else
#pragma openarc #define RES_REGION1 1
#endif
#if RES_REGION2 ==   0
#pragma openarc #define RES_REGION2 0
#else
#pragma openarc #define RES_REGION2 1
#endif
#if RES_REGION3 ==   0
#pragma openarc #define RES_REGION3 0
#else
#pragma openarc #define RES_REGION3 1
#endif
#if RES_REGION4 ==   0
#pragma openarc #define RES_REGION4 0
#else
#pragma openarc #define RES_REGION4 1
#endif
#if RES_REGION5 ==   0
#pragma openarc #define RES_REGION5 0
#else
#pragma openarc #define RES_REGION5 1
#endif
#if RES_REGION6 ==   0
#pragma openarc #define RES_REGION6 0
#else
#pragma openarc #define RES_REGION6 1
#endif
#if RES_REGION7 ==   0
#pragma openarc #define RES_REGION7 0
#else
#pragma openarc #define RES_REGION7 1
#endif
#if RES_REGION8 ==   0
#pragma openarc #define RES_REGION8 0
#else
#pragma openarc #define RES_REGION8 1
#endif
#if RES_REGION9 ==   0
#pragma openarc #define RES_REGION9 0
#else
#pragma openarc #define RES_REGION9 1
#endif
#if RES_REGION10 ==   0
#pragma openarc #define RES_REGION10 0
#else
#pragma openarc #define RES_REGION10 1
#endif
#if RES_REGION11 ==   0
#pragma openarc #define RES_REGION11 0
#else
#pragma openarc #define RES_REGION11 1
#endif
#if RES_REGION12 ==   0
#pragma openarc #define RES_REGION12 0
#else
#pragma openarc #define RES_REGION12 1
#endif
#if RES_REGION13 ==   0
#pragma openarc #define RES_REGION13 0
#else
#pragma openarc #define RES_REGION13 1
#endif
#if RES_REGION14 ==   0
#pragma openarc #define RES_REGION14 0
#else
#pragma openarc #define RES_REGION14 1
#endif
#if RES_REGION15 ==   0
#pragma openarc #define RES_REGION15 0
#else
#pragma openarc #define RES_REGION15 1
#endif
#if RES_REGION16 ==   0
#pragma openarc #define RES_REGION16 0
#else
#pragma openarc #define RES_REGION16 1
#endif
#if RES_REGION17 ==   0
#pragma openarc #define RES_REGION17 0
#else
#pragma openarc #define RES_REGION17 1
#endif
#if RES_REGION18 ==   0
#pragma openarc #define RES_REGION18 0
#else
#pragma openarc #define RES_REGION18 1
#endif
#if RES_REGION19 ==   0
#pragma openarc #define RES_REGION19 0
#else
#pragma openarc #define RES_REGION19 1
#endif
#if RES_REGION20 ==   0
#pragma openarc #define RES_REGION20 0
#else
#pragma openarc #define RES_REGION20 1
#endif
#if RES_REGION21 ==   0
#pragma openarc #define RES_REGION21 0
#else
#pragma openarc #define RES_REGION21 1
#endif
#if RES_REGION22 ==   0
#pragma openarc #define RES_REGION22 0
#else
#pragma openarc #define RES_REGION22 1
#endif
#if RES_REGION23 ==   0
#pragma openarc #define RES_REGION23 0
#else
#pragma openarc #define RES_REGION23 1
#endif
#if RES_REGION24 ==   0
#pragma openarc #define RES_REGION24 0
#else
#pragma openarc #define RES_REGION24 1
#endif
#if RES_REGION25 ==   0
#pragma openarc #define RES_REGION25 0
#else
#pragma openarc #define RES_REGION25 1
#endif
#if RES_REGION26 ==   0
#pragma openarc #define RES_REGION26 0
#else
#pragma openarc #define RES_REGION26 1
#endif
#if RES_REGION27 ==   0
#pragma openarc #define RES_REGION27 0
#else
#pragma openarc #define RES_REGION27 1
#endif
#if RES_REGION28 ==   0
#pragma openarc #define RES_REGION28 0
#else
#pragma openarc #define RES_REGION28 1
#endif

//Decide variables to inject faults
//[DEBUG] scalar shared variables are skipped.
#if _FTVAR0 == 0
#pragma openarc #define FTVAR0 colidx[0:NZ+1]
#else
#pragma openarc #define FTVAR0 rowstr[0:NA+2]
#endif

#pragma openarc #define FTVAR1 x[0:NA+3]

#if _FTVAR2 == 0
#pragma openarc #define FTVAR2 p[0:NA+3]
#elif _FTVAR2 == 1
#pragma openarc #define FTVAR2 q[0:NA+3]
#elif _FTVAR2 == 2
#pragma openarc #define FTVAR2 r[0:NA+3]
#elif _FTVAR2 == 3
#pragma openarc #define FTVAR2 w[0:NA+3]
#elif _FTVAR2 == 4
#pragma openarc #define FTVAR2 x[0:NA+3]
#else
#pragma openarc #define FTVAR2 z[0:NA+3]
#endif

#pragma openarc #define FTVAR3 x[0:NA+3]

#if _FTVAR4 == 0
#pragma openarc #define FTVAR4 a[0:NZ+1]
#elif _FTVAR4 == 1
#pragma openarc #define FTVAR4 oolidx[0:NZ+1]
#elif _FTVAR4 == 2
#pragma openarc #define FTVAR4 p[0:NA+3]
#elif _FTVAR4 == 3
#pragma openarc #define FTVAR4 rowstr[0:NA+2]
#else
#pragma openarc #define FTVAR4 w[0:NA+3]
#endif

#if _FTVAR5 == 0
#pragma openarc #define FTVAR5 q[0:NA+3]
#else
#pragma openarc #define FTVAR5 w[0:NA+3]
#endif

#if _FTVAR6 == 0
#pragma openarc #define FTVAR6 p[0:NA+3]
#elif _FTVAR6 == 1
#pragma openarc #define FTVAR6 q[0:NA+3]
#else
#pragma openarc #define FTVAR6 w[0:NA+3]
#endif

#if _FTVAR7 == 0
#pragma openarc #define FTVAR7 p[0:NA+3]
#elif _FTVAR7 == 1
#pragma openarc #define FTVAR7 q[0:NA+3]
#elif _FTVAR7 == 2
#pragma openarc #define FTVAR7 r[0:NA+3]
#else
#pragma openarc #define FTVAR7 z[0:NA+3]
#endif

#pragma openarc #define FTVAR8 r[0:NA+3]

#if _FTVAR9 == 0
#pragma openarc #define FTVAR9 p[0:NA+3]
#else
#pragma openarc #define FTVAR9 r[0:NA+3]
#endif

#if _FTVAR10 == 0
#pragma openarc #define FTVAR10 a[0:NZ+1]
#elif _FTVAR10 == 1
#pragma openarc #define FTVAR10 colidx[0:NZ+1]
#elif _FTVAR10 == 2
#pragma openarc #define FTVAR10 rowstr[0:NA+2]
#elif _FTVAR10 == 3
#pragma openarc #define FTVAR10 w[0:NA+3]
#else
#pragma openarc #define FTVAR10 z[0:NA+3]
#endif

#if _FTVAR11 == 0
#pragma openarc #define FTVAR11 r[0:NA+3]
#else
#pragma openarc #define FTVAR11 w[0:NA+3]
#endif

#if _FTVAR12 == 0
#pragma openarc #define FTVAR12 r[0:NA+3]
#else
#pragma openarc #define FTVAR12 x[0:NA+3]
#endif

#if _FTVAR13 == 0
#pragma openarc #define FTVAR13 x[0:NA+3]
#else
#pragma openarc #define FTVAR13 z[0:NA+3]
#endif

#if _FTVAR14 == 0
#pragma openarc #define FTVAR14 x[0:NA+3]
#else
#pragma openarc #define FTVAR14 z[0:NA+3]
#endif

#pragma openarc #define FTVAR15 x[0:NA+3]

#if _FTVAR16 == 0
#pragma openarc #define FTVAR16 p[0:NA+3]
#elif _FTVAR16 == 1
#pragma openarc #define FTVAR16 q[0:NA+3]
#elif _FTVAR16 == 2
#pragma openarc #define FTVAR16 r[0:NA+3]
#elif _FTVAR16 == 3
#pragma openarc #define FTVAR16 w[0:NA+3]
#elif _FTVAR16 == 4
#pragma openarc #define FTVAR16 x[0:NA+3]
#else
#pragma openarc #define FTVAR16 z[0:NA+3]
#endif

#pragma openarc #define FTVAR17 x[0:NA+3]

#if _FTVAR18 == 0
#pragma openarc #define FTVAR18 a[0:NZ+1]
#elif _FTVAR18 == 1
#pragma openarc #define FTVAR18 colidx[0:NZ+1]
#elif _FTVAR18 == 2
#pragma openarc #define FTVAR18 p[0:NA+3]
#elif _FTVAR18 == 3
#pragma openarc #define FTVAR18 rowstr[0:NA+2]
#else
#pragma openarc #define FTVAR18 w[0:NA+3]
#endif

#if _FTVAR19 == 0
#pragma openarc #define FTVAR19 q[0:NA+3]
#else
#pragma openarc #define FTVAR19 w[0:NA+3]
#endif

#if _FTVAR20 == 0
#pragma openarc #define FTVAR20 p[0:NA+3]
#elif _FTVAR20 == 1
#pragma openarc #define FTVAR20 q[0:NA+3]
#else
#pragma openarc #define FTVAR20 w[0:NA+3]
#endif

#if _FTVAR21 == 0
#pragma openarc #define FTVAR21 p[0:NA+3]
#elif _FTVAR21 == 1
#pragma openarc #define FTVAR21 q[0:NA+3]
#elif _FTVAR21 == 2
#pragma openarc #define FTVAR21 r[0:NA+3]
#else
#pragma openarc #define FTVAR21 z[0:NA+3]
#endif

#pragma openarc #define FTVAR22 r[0:NA+3]

#if _FTVAR23 == 0
#pragma openarc #define FTVAR23 p[0:NA+3]
#else
#pragma openarc #define FTVAR23 r[0:NA+3]
#endif

#if _FTVAR24 == 0
#pragma openarc #define FTVAR24 a[0:NZ+1]
#elif _FTVAR24 == 1
#pragma openarc #define FTVAR24 colidx[0:NZ+1]
#elif _FTVAR24 == 2
#pragma openarc #define FTVAR24 rowstr[0:NA+2]
#elif _FTVAR24 == 3
#pragma openarc #define FTVAR24 w[0:NA+3]
#else
#pragma openarc #define FTVAR24 z[0:NA+3]
#endif

#if _FTVAR25 == 0
#pragma openarc #define FTVAR25 r[0:NA+3]
#elif _FTVAR25 == 1
#else
#endif

#if _FTVAR26 == 0
#pragma openarc #define FTVAR26 r[0:NA+3]
#else
#pragma openarc #define FTVAR26 x[0:NA+3]
#endif

#if _FTVAR27 == 0
#pragma openarc #define FTVAR27 x[0:NA+3]
#else
#pragma openarc #define FTVAR27 z[0:NA+3]
#endif

#if _FTVAR28 == 0
#pragma openarc #define FTVAR28 x[0:NA+3]
#else
#pragma openarc #define FTVAR28 z[0:NA+3]
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



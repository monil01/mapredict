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
//Allocated in AllocateNodalPersistent()
#if _FTVAR0 == 0
#pragma openarc #define FTVAR0	m_x[0:T_NUMNODE]
#elif _FTVAR0 == 1
#pragma openarc #define FTVAR0	m_xd[0:T_NUMNODE]
#elif _FTVAR0 == 2
#pragma openarc #define FTVAR0	m_xdd[0:T_NUMNODE]
#elif _FTVAR0 == 3
#pragma openarc #define FTVAR0	m_y[0:T_NUMNODE]
#elif _FTVAR0 == 4
#pragma openarc #define FTVAR0	m_yd[0:T_NUMNODE]
#elif _FTVAR0 == 5
#pragma openarc #define FTVAR0	m_ydd[0:T_NUMNODE]
#elif _FTVAR0 == 6
#pragma openarc #define FTVAR0	m_z[0:T_NUMNODE]
#elif _FTVAR0 == 7
#pragma openarc #define FTVAR0	m_zd[0:T_NUMNODE]
#elif _FTVAR0 == 8
#pragma openarc #define FTVAR0	m_zdd[0:T_NUMNODE]
#elif _FTVAR0 == 9
#pragma openarc #define FTVAR0	m_fx[0:T_NUMNODE]
#elif _FTVAR0 == 10
#pragma openarc #define FTVAR0	m_fy[0:T_NUMNODE]
#elif _FTVAR0 == 11
#pragma openarc #define FTVAR0	m_fz[0:T_NUMNODE]
#elif _FTVAR0 == 12
#pragma openarc #define FTVAR0	m_nodalMass[0:T_NUMNODE]
//Allocated in AllocateElemPersistent()
//Only those included in the checkpoint version are shown here.
#elif _FTVAR0 == 13
#pragma openarc #define FTVAR0	m_e[0:T_NUMELEM]
#elif _FTVAR0 == 14
#pragma openarc #define FTVAR0	m_p[0:T_NUMELEM]
#elif _FTVAR0 == 15
#pragma openarc #define FTVAR0	m_q[0:T_NUMELEM]
#elif _FTVAR0 == 16
#pragma openarc #define FTVAR0	m_ql[0:T_NUMELEM]
#elif _FTVAR0 == 17
#pragma openarc #define FTVAR0	m_qq[0:T_NUMELEM]
#elif _FTVAR0 == 18
#pragma openarc #define FTVAR0	m_v[0:T_NUMELEM]
#elif _FTVAR0 == 19
#pragma openarc #define FTVAR0	m_delv[0:T_NUMELEM]
#elif _FTVAR0 == 20
#pragma openarc #define FTVAR0	m_vdov[0:T_NUMELEM]
#elif _FTVAR0 == 21
#pragma openarc #define FTVAR0	m_arealg[0:T_NUMELEM]
#elif _FTVAR0 == 22
#pragma openarc #define FTVAR0	m_ss[0:T_NUMELEM]
//Allocated in AllocateElemTemporary()
//These are not included in the checkpoint version.
#elif _FTVAR0 == 23
#pragma openarc #define FTVAR0	m_dxx[0:T_NUMELEM]
#elif _FTVAR0 == 24
#pragma openarc #define FTVAR0	m_dyy[0:T_NUMELEM]
#elif _FTVAR0 == 25
#pragma openarc #define FTVAR0	m_dzz[0:T_NUMELEM]
#elif _FTVAR0 == 26
#pragma openarc #define FTVAR0	m_delv_xi[0:T_NUMELEM]
#elif _FTVAR0 == 27
#pragma openarc #define FTVAR0	m_delv_eta[0:T_NUMELEM]
#elif _FTVAR0 == 28
#pragma openarc #define FTVAR0	m_delv_zeta[0:T_NUMELEM]
#elif _FTVAR0 == 29
#pragma openarc #define FTVAR0	m_delx_xi[0:T_NUMELEM]
#elif _FTVAR0 == 30
#pragma openarc #define FTVAR0	m_delx_eta[0:T_NUMELEM]
#elif _FTVAR0 == 31
#pragma openarc #define FTVAR0	m_delx_zeta[0:T_NUMELEM]
#elif _FTVAR0 == 32
#pragma openarc #define FTVAR0	m_vnew[0:T_NUMELEM]
//Allocated in AllocateElemPersistent()
//Only partially listed here.
#elif _FTVAR0 == 33
#pragma openarc #define FTVAR0	m_lxim[0:T_NUMELEM]
#elif _FTVAR0 == 34
#pragma openarc #define FTVAR0	m_lxip[0:T_NUMELEM]
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



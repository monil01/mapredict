#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define NVL 1
#define VHEAP 2
#if MEM == NVL
#include <nvl.h>
#include <nvlrt-test.h>
#elif MEM == VHEAP
#include <nvl-vheap.h>
#else
# error unknown MEM setting
#endif

#ifndef VERIFICATION
#define VERIFICATION 0
#endif

#define ITER 	1000

#ifndef SIZE
//#define SIZE 	2048 //128 * 16
//#define SIZE    4096 //256 * 16
#define SIZE    8192 //256 * 32
//#define SIZE  12288 //256 * 48
#endif

#ifndef SIZE_1
#define SIZE_1 	(SIZE+1)
#endif
#ifndef SIZE_2
#define SIZE_2 	(SIZE+2)
#endif

#ifndef HEAPSIZE
#define HEAPSIZE (SIZE_2*SIZE_2*16)
#endif

#define CHECK_RESULT

#ifndef NVLFILE
#define NVLFILE "_NVLFILEPATH_"
//#define NVLFILE "/opt/fio/scratch/f6l/jacobi.nvl"
//#define NVLFILE "/opt/rd/scratch/f6l/jacobi.nvl"
//#define NVLFILE "/tmp/f6l/jacobi.nvl"
#endif

#ifndef NVLTMP
#define NVLTMP "/opt/fio/scratch/f6l"
#endif

double my_timer ()
{
    struct timeval time;

    gettimeofday (&time, 0);

    return time.tv_sec + time.tv_usec / 1000000.0;
}


#if MEM == NVL
struct root {
#if TXS
	int k;
#endif
	nvl float (*b)[SIZE_2];
};
#endif

int main (int argc, char *argv[])
{
    int i, j, k;
    //int c;
    float sum = 0.0f;
    float (*a_v)[SIZE_2] = 0;
#if MEM == NVL
        nvl float (*b_nv)[SIZE_2] = 0;
#if TXS
	nvl int *k_nv = 0;
#endif
#elif MEM == VHEAP
	float (*b_nv)[SIZE_2] = 0;
#else
# error unknown MEM setting
#endif
#if !POOR
	float (*b_v)[SIZE_2];
#endif

    double strt_time, done_time;
#if VERIFICATION >= 1
	float ** a_CPU = (float**)malloc(sizeof(float*) * SIZE_2);
	float ** b_CPU = (float**)malloc(sizeof(float*) * SIZE_2);

	float * a_data = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);
	float * b_data = (float*)malloc(sizeof(float) * SIZE_2 * SIZE_2);

	for(i = 0; i < SIZE_2; i++)
	{
		a_CPU[i] = &a_data[i * SIZE_2];
		b_CPU[i] = &b_data[i * SIZE_2];
	}
#endif 
#if MEM == NVL
	nvl_heap_t *heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
	if(!heap) {
		perror("nvl_create failed");
		return 1;
	}
	nvl struct root *root_nv = 0;
	if( !(a_v = malloc(sizeof(float) * SIZE_2 * SIZE_2)))
	{
		perror("malloc failed");
		return 1;	
	}
	if( !(root_nv = nvl_alloc_nv(heap, 1, struct root)) 
		|| !(b_nv = nvl_alloc_nv(heap, SIZE_2, nvl float [SIZE_2])) )
	{
		perror("nvl_alloc_nv failed");
		return 1;	
	}
	nvl_set_root(heap, root_nv);
#if TXS
	root_nv->k = 1;
	k_nv = &root_nv->k;
#endif
	root_nv->b = b_nv;
#elif MEM == VHEAP
	nvl_vheap_t *vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
	if(!vheap) {
		perror("nvl_vcreate failed");
		return 1;
	}
	if( !(a_v = malloc(SIZE_2*SIZE_2*sizeof(float))))
	{
		perror("malloc failed");
		return 1;	
	}
	if( !(a_v = malloc(SIZE_2*SIZE_2*sizeof(float)))
			|| !(b_nv = (float (*)[SIZE_2])nvl_vmalloc(vheap, SIZE_2*SIZE_2*sizeof(float))) )
	{
		perror("nvl_vmalloc failed");
		return 1;	
	}
#else
# error unknown MEM setting
#endif

    //while ((c = getopt (argc, argv, "")) != -1);

	{
#if MEM == NVL && PERSIST
    // The initialization loops aren't part of our timings, but they're
    // outrageously slow if persists are inserted within.
    float (*b_nv_bare)[SIZE_2] = nvl_bare_hack(b_nv);
    float (*b_nv)[SIZE_2] = b_nv_bare;
#endif
    for (i = 0; i < SIZE_2; i++)
    {
        for (j = 0; j < SIZE_2; j++)
        {
            b_nv[i][j] = 0;
#if VERIFICATION >= 1
			b_CPU[i][j] = 0;
#endif 
        }
    }

    for (j = 0; j <= SIZE_1; j++)
    {
        b_nv[j][0] = 1.0;
        b_nv[j][SIZE_1] = 1.0;

#if VERIFICATION >= 1
		b_CPU[j][0] = 1.0;
		b_CPU[j][SIZE_1] = 1.0;
#endif 

    }
    for (i = 1; i <= SIZE; i++)
    {
        b_nv[0][i] = 1.0;
        b_nv[SIZE_1][i] = 1.0;

#if VERIFICATION >= 1
		b_CPU[0][i] = 1.0;
		b_CPU[SIZE_1][i] = 1.0;
#endif 
    }
	}

    printf ("Performing %d iterations on a %d by %d array\n", ITER, SIZE, SIZE);

    /* -- Timing starts before the main loop -- */
    printf("-------------------------------------------------------------\n");

    strt_time = my_timer ();

#if TXS
    assert(ITER % ITERS_PER_TX == 0);
    for (k=*k_nv; k<=ITER;)
    {
#if (TXS == 1)
	#pragma nvl atomic heap(heap)
#elif (TXS == 2)
	#pragma nvl atomic heap(heap) default(readonly) \
	backup(k_nv[0:1], b_nv[1:SIZE][0:SIZE_2])
#else
	#error cannot use clobber for this version of jacobi
#endif
	for (int k_sub=0; k_sub<ITERS_PER_TX; ++k_sub, ++k, ++*k_nv)
	{
# if !POOR
	  // Because of shadow updating, we must update the bare pointer
	  // after the tx.add, which the compiler hoists right before the
	  // k_sub loop in the case of TXS=1.
	  b_v = nvl_bare_hack(b_nv);
# endif
#else
# if MEM == NVL && !POOR
    b_v = nvl_bare_hack(b_nv);
# endif
    for (k = 0; k < ITER; k++)
    {
#endif
        for (i = 1; i <= SIZE; i++)
        {
            	for (j = 1; j <= SIZE; j++)
            	{
#if (MEM == VHEAP) || POOR
                	a_v[i][j] = (b_nv[i - 1][j] + b_nv[i + 1][j] + b_nv[i][j - 1] + b_nv[i][j + 1]) / 4.0f;
#else
                	a_v[i][j] = (b_v[i - 1][j] + b_v[i + 1][j] + b_v[i][j - 1] + b_v[i][j + 1]) / 4.0f;
#endif
            	}
        }

        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
#if (MEM == VHEAP) || POOR || TXS
                b_nv[i][j] = a_v[i][j];
#else
                b_v[i][j] = a_v[i][j];
#endif
            }
        }
#if TXS
	}
#endif
    }
#if (MEM == NVL) && !POOR && PERSIST
	nvl_persist_hack(b_nv, SIZE_2);
#endif

    done_time = my_timer ();
    printf ("NVM Elapsed time = %lf sec\n", done_time - strt_time);
#if MEM == NVL && TXS
    printf ("  tx add time  = %lf sec\n", nvlrt_get_txAddTime());
    printf ("  shadow update alloc time  = %lf sec\n",
            nvlrt_get_shadowUpdateAllocTime());
#endif

#if VERIFICATION >= 1

    strt_time = my_timer ();

    for (k = 0; k < ITER; k++)
    {
        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
                a_CPU[i][j] = (b_CPU[i - 1][j] + b_CPU[i + 1][j] + b_CPU[i][j - 1] + b_CPU[i][j + 1]) / 4.0f;
            }
        }

        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
                b_CPU[i][j] = a_CPU[i][j];
            }
        }
    }

    done_time = my_timer ();
    printf ("Volatile Memory Elapsed time = %lf sec\n", done_time - strt_time);
#if VERIFICATION == 1
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;

		for (i = 1; i <= SIZE; i++)
    	{
        	cpu_sum += b_CPU[i][i]*b_CPU[i][i];
			gpu_sum += b_nv[i][i]*b_nv[i][i];
    	}

		cpu_sum = sqrt(cpu_sum);
		gpu_sum = sqrt(gpu_sum);
		if( cpu_sum > gpu_sum) {
			rel_err = (cpu_sum-gpu_sum)/cpu_sum;
		} else {
			rel_err = (gpu_sum-cpu_sum)/cpu_sum;
		}

		if(rel_err < 1e-9)
		{
	    	printf("Verification Successful err = %e\n", rel_err);
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#else
	{
		double cpu_sum = 0.0;
		double gpu_sum = 0.0;
    	double rel_err = 0.0;
		int error_found = 0;

        for (i = 1; i <= SIZE; i++)
        {
            for (j = 1; j <= SIZE; j++)
            {
        		cpu_sum = b_CPU[i][j];
				gpu_sum = b_nv[i][j];
				if( cpu_sum == gpu_sum ) {
					continue;
				}
				if( cpu_sum > gpu_sum) {
					if( cpu_sum == 0.0 ) {
						rel_err = cpu_sum-gpu_sum;
					} else {
						rel_err = (cpu_sum-gpu_sum)/cpu_sum;
					}
				} else {
					if( cpu_sum == 0.0 ) {
						rel_err = gpu_sum-cpu_sum;
					} else {
						rel_err = (gpu_sum-cpu_sum)/cpu_sum;
					}
				}
				if(rel_err < 0.0) {
					rel_err = -1*rel_err;
				}

				if(rel_err >= 1e-9)
				{
					error_found = 1;
					break;
				}
			}
			if( error_found == 1 ) {
				break;
			}
		}
		if( error_found == 0 )
		{
	    	printf("Verification Successful\n");
		}
		else
		{
	    	printf("Verification Fail err = %e\n", rel_err);
		}
	}
#endif
#endif


#ifdef CHECK_RESULT
    for (i = 1; i <= SIZE; i++)
    {
        sum += b_nv[i][i];
    }
    printf("Diagonal sum = %.10E\n", sum);
#endif

#if MEM == NVL
	nvl_close(heap);
#elif MEM == VHEAP
	nvl_vfree(vheap, b_nv);
	nvl_vclose(vheap);
#endif

    return 0;
}


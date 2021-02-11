#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <nvl.h>
#include <nvlrt-test.h>
#include <libpmem.h>

double  timer_()
{
  struct  timeval time;

  gettimeofday(&time, 0);

  return time.tv_sec + time.tv_usec/1000000.0;
}

//////////////////////////
// Macros for debugging //
//////////////////////////
//#define DEBUG_ON1
#define DEBUG_ON2

#ifndef SPMUL_INPUTDIR
#define SPMUL_INPUTDIR "/opt/proj-local/jum/spmul-input/"
#endif
#define ITER  100
#define ITERS_PER_TX 1

#define INPUTFILE  "nlpkkt240.rbC"
#define SIZE  27993600
#define SIZE2  27993600 //debugging purpose (should be replaced with SIZE)
#define NZR    401232976
#ifdef _OPENARC_
#pragma openarc #define SIZE  27993600
#pragma openarc #define SIZE2  27993600 //debugging purpose (should be replaced with SIZE)
#pragma openarc #define NZR    401232976
#endif

// #define INPUTFILE  "rajat31.rbC"
// #define SIZE  4690002
// #define SIZE2  4690002 //debugging purpose (should be replaced with SIZE)
// #define NZR    20316253
// #ifdef _OPENARC_
// #pragma openarc #define SIZE  4690002
// #pragma openarc #define SIZE2  4690002 //debugging purpose (should be replaced with SIZE)
// #pragma openarc #define NZR    20316253
// #endif

// #define INPUTFILE  "kkt_power.rbC"
// #define SIZE  2063494
// #define SIZE2  2063494 //debugging purpose (should be replaced with SIZE)
// #define NZR    8130343
// #ifdef _OPENARC_
// #pragma openarc #define SIZE  2063494
// #pragma openarc #define SIZE2  2063494 //debugging purpose (should be replaced with SIZE)
// #pragma openarc #define NZR    8130343
// #endif

// #define INPUTFILE  "af_shell10.rbC"
// #define SIZE  1508065
// #define SIZE2  1508065 //debugging purpose (should be replaced with SIZE)
// #define NZR    27090195
// #ifdef _OPENARC_
// #pragma openarc #define SIZE  1508065
// #pragma openarc #define SIZE2  1508065 //debugging purpose (should be replaced with SIZE)
// #pragma openarc #define NZR    27090195
// #endif

// #define INPUTFILE   "hood.rbC"
// #define SIZE    220542  
// #define SIZE2   220542  
// #define NZR 5494489 
// #ifdef _OPENARC_
// #pragma openarc #define SIZE  220542
// #pragma openarc #define SIZE2  220542 //debugging purpose (should be replaced with SIZE)
// #pragma openarc #define NZR    5494489
// #endif

// #define INPUTFILE  "af23560.rbC"
// #define SIZE  23560
// #define SIZE2  23560 //debugging purpose (should be replaced with SIZE)
// #define NZR    484256
// #ifdef _OPENARC_
// #pragma openarc #define SIZE  23560
// #pragma openarc #define SIZE2  23560 //debugging purpose (should be replaced with SIZE)
// #pragma openarc #define NZR    484256
// #endif

/*
#define ITER  500
#define INPUTFILE  "msdoor.rbC"
#define SIZE  415863
#define SIZE2  415863
#define NZR    10328399
*/

/*
#define INPUTFILE  "appu.rbC"
//#define INPUTFILE  "appu.rbCR"
//#define INPUTFILE  "appu.rbCRP"
#define SIZE  14000
#define SIZE2  14000
#define NZR    1853104
//#define NZR    1857600
*/

/*
#define INPUTFILE  "nd24k.rbC"
#define SIZE  72000
#define SIZE2  72000
#define NZR    14393817

//#define INPUTFILE  "F1.rbC"
#define INPUTFILE  "F1.rbCRP"
#define SIZE  343791
#define SIZE2  343791
//#define NZR    13590452
#define NZR    13596431

//#define INPUTFILE  "ASIC_680k.rbC"
#define INPUTFILE  "ASIC_680k.rbCR"
#define SIZE  682862
#define SIZE2  682862
#define NZR    3871773

#define INPUTFILE  "ASIC_680ks.rbC"
#define SIZE  682712
#define SIZE2  682712
#define NZR    2329176

#define INPUTFILE  "crankseg_2.rbC"
#define SIZE  63838
#define SIZE2  63838
#define NZR    7106348

#define INPUTFILE  "darcy003.rbC"
#define SIZE  389874
#define SIZE2  389874
#define NZR    1167685

#define INPUTFILE  "Si41Ge41H72.rbC"
#define SIZE  185639
#define SIZE2  185639
#define NZR    7598452

#define INPUTFILE  "SiO2.rbC"
#define SIZE  155331
#define SIZE2  155331
#define NZR    5719417
*/
/*
#define INPUTFILE   "sparsine.rbCR"
#define SIZE    50000
#define SIZE2   50000
#define NZR 799494

#define INPUTFILE   "sparsine.rbCRPF"
#define SIZE    50000   
#define SIZE2   50000   
#define NZR 3200000 
*/
/*
#define INPUTFILE   "ns3Da.rbCRPF"
#define SIZE    20414   
#define SIZE2   20414   
#define NZR 6533120 
*/

/*
#define INPUTFILE   "af23560.rand51M"
#define SIZE    100000  
#define SIZE2   100000
#define NZR 6400000
*/

/*
#define INPUTFILE   "af23560.rand200M"
#define SIZE    100000  
#define SIZE2   100000
#define NZR 25600000
*/

#define HEAPSIZE (20ull*SIZE)
//#define NVLFILE "spmul.nvl"
#define NVLFILE "/opt/fio/scratch/jum/spmul.nvl"
//#define NVLFILE "/opt/rd/scratch/jum/spmul.nvl"

struct root {
  int k;
  nvl float *x;
};

// Linking fails if these are two large, so switch to malloc.
#define BSS_SIZE_MAX 5000000 // limit guessed based on megatron compiles
#if SIZE <= BSS_SIZE_MAX
int colind[NZR];
int rowptr[SIZE+1];
float values[NZR];
float y[SIZE];
#endif

int main() {
#if TXS
  nvlrt_setShadowUpdateCostMode(SHADOW_UPDATE_COST_MODE);
#endif

#if SIZE > BSS_SIZE_MAX
  int *colind = malloc(NZR * sizeof(int));
  int *rowptr = malloc((SIZE+1) * sizeof(int));
  float *values = malloc(NZR * sizeof(float));
  float *y = malloc(SIZE * sizeof(float));
  if (!colind || !rowptr || !values || !y) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }
#endif

  FILE *fp10;
  //FILE *fp12; //Result writing part is disabled
  char filename1[96] = SPMUL_INPUTDIR;
  char filename2[32] = INPUTFILE;

  float temp, x_sum;
  double s_time1, e_time1, s_time2, e_time2;
  double s_time3, e_time3;
  int exp0, i, j;
  int r_ncol, r_nnzero, r_nrow;
  int cpumemsize = 0;

  printf("**** SerialSpmul starts! ****\n");

  strcat(filename1, filename2);

  nvl_heap_t *heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
  if(!heap) {
    perror("nvl_create failed");
    return 1;
  }
  nvl struct root *root = 0;
  if (!(root = nvl_alloc_nv(heap, 1, struct root))
      || !(root->x = nvl_alloc_nv(heap, SIZE, float)))
  {
    perror("nvl_alloc_nv failed");
    return 1;
  }
  nvl_set_root(heap, root);
  root->k = 0;
  nvl int *kp = &root->k;
  nvl float *x = root->x;

  printf("Input file: %s\n", filename2);

  s_time1 = timer_();
  s_time2 = timer_();
  if( (fp10 = fopen(filename1, "r")) == NULL ) {
    printf("FILE %s DOES NOT EXIST; STOP\n", filename1);
    exit(1);
  }
/*
  if( (fp12 = fopen("spmulSP.out", "w")) == NULL ) {
    exit(1);
  }
*/
  printf("FILE open done\n");

  fscanf(fp10, "%d %d %d", &r_nrow, &r_ncol, &r_nnzero);
  if (r_nrow != SIZE) {
    printf("alarm: incorrect row\n");
    exit(1);
  }
  if (r_ncol != SIZE) {
    printf("alarm: incorrect col\n");
    exit(1);
  }
  if (r_nnzero != NZR) {
    printf("alarm: incorrect nzero\n");
    exit(1);
  }
  for( i=0; i<=SIZE; i++ ) {
    fscanf(fp10, "%d", (rowptr+i));
  }

  for( i=0; i<NZR; i++ ) {
    fscanf(fp10, "%d", (colind+i));
  }

  for( i=0; i<NZR; i++ ) {
    fscanf(fp10, "%E", (values+i)); //for float variables
  }
  fclose(fp10);

  j = 0;
    for( i=0; i<SIZE; i++ ) {
LB99:
    temp = values[j];
    if( ((-0.1f) < temp)&&(temp < 0.1f) ) {
      j += 1;
      //goto LB99;
      //Added by SYLee
      if( temp == 0.0f )
        goto LB99;
      x[i] = temp;
      continue;
    }
    exp0 = (int)(log10f(fabsf(temp)));
    x[i] = temp;
    if( (-exp0) <= 0 ) {
      for( int k=1; k<=(1+exp0); k++ ) {
        x[i] = x[i]/10.0f;
      }
    } else if( (1+exp0) <= 0 ) {
      for( int k=1; k<=(-exp0); k++ ) {
        x[i] = 10.0f*x[i];
      }
    }
    if( (1.0f < x[i])||(x[i] < (-1.0f)) ) {
      printf("alarm initial i = %d\n", i);
      printf("x = %E\n", x[i]);
      printf("value = %E\n", values[1000+i]);
      printf("exp = %d\n", exp0);
      exit(1);
    }
    j += 1;
  }

#ifdef DEBUG_ON1
  x_sum = 0.0f;
  for( i=0; i<SIZE; i++ ) {
    x_sum += x[i];
  }
  printf("0: x_sum = %.12E\n", x_sum);
#endif
  cpumemsize += sizeof(int) * (NZR + SIZE + 1);
  cpumemsize += sizeof(float) * (NZR + 2*SIZE);
  printf("Used CPU memory: %d bytes\n", cpumemsize);

  printf("initialization done\n");
  e_time2 = timer_();
  s_time3 = timer_();

  printf("Performing %d iterations with SIZE=%d\n", ITER, SIZE);
  printf("(is_pmem=%d, tx mode=%d",
         pmem_is_pmem(nvl_bare_hack(x), SIZE*sizeof(float)), TXS);
#if TXS
  printf(", ITERS_PER_TX=%d", ITERS_PER_TX);
#endif
  printf(")\n");

  assert(ITER % ITERS_PER_TX == 0);
  while( *kp < ITER ) {
#if !TXS
#elif (TXS == 1)
    #pragma nvl atomic heap(heap)
#elif (TXS == 2)
    #pragma nvl atomic heap(heap) default(readonly) \
            backup(kp[0:1], x[0:SIZE])
#elif (TXS == 3)
    #error cannot use clobber for this version of jacobi
#elif (TXS == 4)
    for( i=0; i<SIZE2; i++ ) {
      y[i] = 0.0f;
      for( j=0; j<(rowptr[1+i]-rowptr[i]); j++ ) {
        y[i] = y[i] + values[rowptr[i]+j-1]*x[colind[rowptr[i]+j-1]-1];
      }
    }
    #pragma nvl atomic heap(heap) default(readonly) \
            backup(kp[0:1]), backup_writeFirst(x[0:SIZE])
#else
    #error unknown TXS setting
#endif
    for (int k_sub=0; k_sub<ITERS_PER_TX; ++k_sub, ++*kp) {
#if TXS == 4
      if( k_sub > 0 )
#endif
      for( i=0; i<SIZE2; i++ ) {
        y[i] = 0.0f;
        for( j=0; j<(rowptr[1+i]-rowptr[i]); j++ ) {
          y[i] = y[i] + values[rowptr[i]+j-1]*x[colind[rowptr[i]+j-1]-1];
        }
      }
      for( i=0; i<SIZE2; i++ ) {
        float tmp = y[i];
        x[i] = tmp;
        if( tmp != 0.0f ) {
          exp0 = (int)(log10f(fabsf(tmp)));
          if( exp0 >= 0 ) {
            for( j=1; j<=(1+exp0); j++ ) {
              x[i] = x[i]/10.0f;
            }
          } else if( exp0 <= -1 ) {
            j = -1;
            for( j=1; j<=(-exp0); j++ ) {
              x[i] = 10.0f*x[i];
            }
          }
        }
      }
    }
  } //end of k-loop

  e_time3 = timer_();
  e_time1 = timer_();
  printf("Total time = %f seconds\n", (e_time1 - s_time1));
  printf("Initialize time = %f seconds\n", (e_time2 - s_time2));
  printf("Computation time = %f seconds\n", (e_time3 - s_time3));

#ifdef DEBUG_ON2
  x_sum = 0.0f;
  for( i=0; i<SIZE2; i++ ) {
    x_sum += x[i];
  }
  printf("%d: x_sum = %.12E\n",(*kp+1), x_sum);
#endif

/*
  for( i=0; i< SIZE; i++ ) {
    fprintf(fp12, "%.9E\n", x[i]);
  }

  fclose(fp12);
*/

  nvl_close(heap);
  return 0;
}

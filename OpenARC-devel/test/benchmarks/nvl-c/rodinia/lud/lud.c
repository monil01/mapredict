/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

static int do_verify = 0;
int omp_num_threads = 1;

#if MEM == NVL
struct root {
#if TXS
	int i;
#endif
	nvl float * m;
};
nvl_heap_t *heap = 0;
#if TXS
nvl int *i_nv = 0;
#endif
#elif MEM == VHEAP
nvl_vheap_t *vheap = 0;
#endif

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULLZ, 'i'},
      {"size", 1, NULLZ, 's'},
      {"verify", 0, NULLZ, 'v'},
      {0,0,0,0}
};

extern void
lud_omp(NVL_PREFIX float *m, int matrix_dim);

int
main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULLZ;
  NVL_PREFIX float *m = 0;
  float *mm = 0;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:n:", 
                            long_options, &option_index)) != -1 ) {
      switch(opt){
        case 'i':
          input_file = optarg;
          break;
		case 'n':
          omp_num_threads = atoi(optarg);
          break;
        case 'v':
          do_verify = 1;
          break;
        case 's':
          matrix_dim = atoi(optarg);
          fprintf(stderr, "Currently not supported, use -i instead\n");
          fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
          exit(EXIT_FAILURE);
        case '?':
          fprintf(stderr, "invalid option\n");
          break;
        case ':':
          fprintf(stderr, "missing argument\n");
          break;
        default:
          fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n",
                  argv[0]);
          exit(EXIT_FAILURE);
      }
  }
  
  if ( (optind < argc) || (optind == 1)) {
      fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
      exit(EXIT_FAILURE);
  }

#if MEM == NVL
    heap = nvl_create(NVLFILE, HEAPSIZE, 0600);
    if(!heap) {
        fprintf(stderr, "file %s already exists\n", NVLFILE);
        exit(1);
    }   
    nvl struct root *root_nv = 0;
    if( !(root_nv = nvl_alloc_nv(heap, 1, struct root)) )  
    {   
        perror("nvl_alloc_nv failed");
        exit(1);
    }   
    nvl_set_root(heap, root_nv);
#elif MEM == VHEAP
    vheap = nvl_vcreate(NVLTMP, HEAPSIZE);
    if(!vheap) {
        perror("nvl_vcreate failed");
        exit(1);   
    }   
#endif

  if (input_file) {
      printf("Reading matrix from file %s\n", input_file);
      ret = create_matrix_from_file(&m, input_file, &matrix_dim);
#if MEM == NVL
      root_nv->m = m;
#if TXS
      i_nv = &root_nv->i;
#endif
#endif
      if (ret != RET_SUCCESS) {
          m = NULLZ;
          fprintf(stderr, "error create matrix from file %s\n", input_file);
          exit(EXIT_FAILURE);
      }
  } else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  if (do_verify){
    printf("Before LUD\n");
    //print_matrix(m, matrix_dim);
#if MEM == NVL
    matrix_duplicate(nvl_bare_hack(m), &mm, matrix_dim);
#else
    matrix_duplicate(m, &mm, matrix_dim);
#endif
  }

     stopwatch_start(&sw);
      lud_omp(m, matrix_dim);
      stopwatch_stop(&sw);
      printf("Accelerator Elapsed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
#if MEM == NVL
    print_matrix(nvl_bare_hack(m), matrix_dim);
#else
    print_matrix(m, matrix_dim);
#endif
    printf(">>>Verify<<<<\n");
#if MEM == NVL
    lud_verify(mm, nvl_bare_hack(m), matrix_dim); 
#else
    lud_verify(mm, m, matrix_dim); 
#endif
    free(mm);
  }

#if MEM == NVL
  nvl_close(heap);
#elif MEM == VHEAP
  nvl_vfree(vheap, m);
  nvl_vclose(vheap);
#else
  free(m);
#endif

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */

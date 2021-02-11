// Build this with the provided Makefile to get correct macro definitions.

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <nvl.h>
#include <nvlrt-test.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

// These just help with syntax highlighting and are overridden in Makefile.
#if !defined(N) || !defined(ITERS) || !defined(ITERS_PER_TX)
# error not compiled with sufficient macros defined
# define __nvl__
# define __builtin_nvl_alloc_nv(Heap, N, Type) NULL
# define __builtin_nvl_get_root(Heap, Type) NULL
# define N -1
# define ITERS -1
# define ITERS_PER_TX -1
#endif

#define HEAPSIZE_FACTOR 10

struct root {
  int iter;
  nvl double (*m)[N];
};

static inline void printMatrix(FILE *file, int iter, int rank,
                               int rowsLocal, int iBase, nvl double m[][N])
{
  for (int i = 1; i <= rowsLocal; i++) {
    fprintf(file, "iter %3d, rank %3d, row %3d:", iter, rank, iBase+i-1);
    for (int j = 0; j < N; j++)
      fprintf(file, " %f", m[i][j]);
    fprintf(file, "\n");
  }
}

static inline int digits(int n) {
  int res = 1;
  if (n != 0) {
    if (n < 0) {
      ++res;
      n = -n;
    }
    res += log10(n);
  }
  return res;
}

static inline long parseLong(const char *desc, long min, long max,
                             const char *str, char expNext,
                             char **endptr)
{
  char *endp;
  char **endpp = endptr ? endptr : &endp;
  long res = strtol(str, endpp, 10);
  if (**endpp != expNext || res < min || res > max) {
    fprintf(stderr, "%s is not an integer in interval [%ld,%ld]: %s\n",
            desc, min, max, str);
    exit(1);
  }
  return res;
}

static void usage(const char *arg0) {
  fprintf(stderr, "Usage: %s NVMDIR RESDIR\n", arg0);
  fprintf(stderr, "NVMDIR=directory on NVM device\n");
  fprintf(stderr, "RESDIR=directory for jacobi results\n");
  fprintf(stderr, "The following options can appear anywhere:\n");
  fprintf(stderr, "  -- to treat remaining arguments as non-options\n");
  fprintf(stderr, "  -kRANK=NSEC to kill rank RANK after NSEC nanosecs\n");
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size, dummy;
  char procName[MPI_MAX_PROCESSOR_NAME];
  MPI_Group group;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(procName, &dummy);
  MPI_Comm_group(MPI_COMM_WORLD, &group);

  // Parse arguments and set any specified alarm.
  const char *nvmDir = NULL;
  const char *resDir = NULL;
  {
    bool endOpts = false;
    int nonOpti = 0;
    for (int argi = 1; argi < argc; ++argi) {
      const char *arg = argv[argi];
      if (!endOpts && arg[0] == '-') {
        if (arg[1] == '-' && arg[2] == 0)
          endOpts = true;
        else if (arg[1] == 'k') {
          char *eq;
          const long killRank = parseLong("RANK in -k", 0, size-1,
                                          arg+2, '=', &eq);
          const long killNsec = parseLong("NSEC in -k", 0, LONG_MAX,
                                          eq+1, 0, NULL);
          if (rank == killRank) {
            struct itimerval timer;
            long killSec = killNsec/1000000000;
            long killUsec = (killNsec%1000000000)/1000;
            if (killSec == 0 && killUsec == 0)
              killUsec = 1; // otherwise no alarm
            timer.it_value.tv_sec = killSec;
            timer.it_value.tv_usec = killUsec;
            timer.it_interval.tv_sec = 0;
            timer.it_interval.tv_usec = 0;
            //printf("rank %d alarms in %ld sec %ld usec\n",
            //       killRank, killSec, killUsec);
            fflush(stdout);
            if (setitimer(ITIMER_REAL, &timer, NULL)) {
              perror("setitimer failed");
              return 1;
            }
          }
        }
        else {
          fprintf(stderr, "unknown option: %s\n", arg);
          return 1;
        }
      }
      else {
        switch (nonOpti++) {
        case 0: nvmDir = arg; break;
        case 1: resDir = arg; break;
        default: usage(argv[0]); return 1;
        }
      }
    }
    if (!nvmDir || !resDir) {
      usage(argv[0]);
      return 1;
    }
  }

  // The rows of the full matrix are divided as evenly as possible over
  // the processes.
  // rowsLocal is the number of rows containing this process's portion of
  // the full matrix.
  // rowsLocal does not include the first and last locally allocated rows,
  // which store rows copied from neighbors, and which are thus unused in
  // the first and last processes, respectively.
  // Within the full matrix, iBase is the index of the first row in
  // rowsLocal.
  // Within the local matrix, iFirst and iLast are the indices of the first
  // and last rows of rowsLocal that are actually modified each iteration and
  // that thus exclude the first and last row of the full matrix.
  const int rowsLocal = N/size + (rank < N%size ? 1 : 0);
  const int iBase = rank < N%size
                    ? rank*rowsLocal
                    : N%size*(rowsLocal+1) + (rank-N%size)*rowsLocal;
  const int iFirst = rank > 0 ? 1 : 2;
  const int iLast = rank < size-1 ? rowsLocal : rowsLocal-1;
  printf("rank = %*d/%d, rowsLocal = %*d/%d, procName = %s\n",
         digits(size-1), rank, size, digits(N), rowsLocal, N, procName);
  fflush(stdout);

  // Create or recover heap. If creating, init matrix with 1 on boundary and
  // 0 otherwise.
  char *heapFileName = malloc(strlen(nvmDir) + 1 + digits(rank) + 5);
  if (!heapFileName) {
    fprintf(stderr, "malloc failed on rank %d: ", rank);
    perror(NULL);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sprintf(heapFileName, "%s/%d.nvl", nvmDir, rank);
  nvl_heap_t *heap
#if TXS
    = nvl_recover_mpi(
#else
    = nvl_recover(
#endif
        heapFileName, HEAPSIZE_FACTOR*(rowsLocal+2)*N * sizeof(double), 0600
#if TXS
        , group);
#else
        );
#endif
  if (!heap) {
    fprintf(stderr, "%s: ", heapFileName);
    perror("failed to recover or create NVM heap");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  printf("rank %d: heapReady\n", rank);
  fflush(stdout);
  nvl struct root *root = nvl_get_root(heap, struct root);
  if (!root) {
#if TXS
    #pragma nvl atomic heap(heap)
#endif
    {
      nvl double (*m)[N] = NULL;
      if (!(root = nvl_alloc_nv(heap, 1, struct root))
          || !(m = root->m = nvl_alloc_nv(heap, rowsLocal+2, double[N])))
      {
        fprintf(stderr, "nvl_alloc_nv failed on rank %d: ", rank);
        perror(NULL);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      nvl_set_root(heap, root);
      root->iter = 0;
      if (rank == 0)
        for (int j = 0; j < N; j++)
          m[iFirst-1][j] = 1;
      for (int i = iFirst; i <= iLast; i++) {
        m[i][0] = 1;
        for (int j = 1; j < N-1; j++)
          m[i][j] = 0;
        m[i][N-1] = 1;
      }
      if (rank == size-1)
        for (int j = 0; j < N; j++)
          m[iLast+1][j] = 1;
    }
  }
  printf("rank %d: rootReady\n", rank);
  fflush(stdout);
  nvl double (*m)[N] = root->m;

  // Print configuration.
  if (rank == 0) {
    printf("NVMDIR = %s, RESDIR = %s\n", nvmDir, resDir);
    printf("ITERS = %d, N = %d, usesMsync = %d, ",
           ITERS, N, nvlrt_usesMsync(heap));
#if TXS
    printf("tx mode = %d, ITERS_PER_TX = %d\n", TXS, ITERS_PER_TX);
#else
    printf("txs disabled\n");
#endif
    fflush(stdout);
  }

  // Init volatile matrix boundary as it will not be written again.
  double (*mTmp)[N] = malloc((rowsLocal+2) * N * sizeof(double));
  if (!mTmp) {
    fprintf(stderr, "malloc failed on rank %d: ", rank);
    perror(NULL);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (rank == 0)
    for (int j = 0; j < N; j++)
      mTmp[iFirst-1][j] = 1;
  for (int i = iFirst; i <= iLast; i++) {
    mTmp[i][0] = mTmp[i][N-1] = 1;
  }
  if (rank == size-1)
    for (int j = 0; j < N; j++)
      mTmp[iLast+1][j] = 1;

  // Create results file.
  char *resFileName = malloc(strlen(resDir) + 1 + digits(rank) + 5);
  if (!resFileName) {
    fprintf(stderr, "malloc failed on rank %d: ", rank);
    perror(NULL);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sprintf(resFileName, "%s/%d.txt", resDir, rank);
  FILE *resFile = fopen(resFileName, "w");
  if (!resFile) {
    fprintf(stderr, "%s: ", resFileName);
    perror("failure opening as output file");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Main loop.
  nvl int *iter = &root->iter;
#if !TXS
  for (; *iter < ITERS; ++*iter) {
#else
  assert(ITERS % ITERS_PER_TX == 0);
  while (*iter < ITERS) {
# if TXS == 1
    #pragma nvl atomic heap(heap) mpiGroup(group)
# elif TXS == 2
    #pragma nvl atomic heap(heap) mpiGroup(group) default(readonly) \
            backup(iter[0:1], m[iFirst:iLast-iFirst+1][0:N])
# elif TXS == 3
    #error cannot use clobber for this version of jacobi
# elif TXS == 4
    for (int i = iFirst; i <= iLast; i++)
      for (int j = 1; j < N-1; j++)
        mTmp[i][j] = m[i][j];
    #pragma nvl atomic heap(heap) mpiGroup(group) default(readonly) \
            backup(iter[0:1]) \
            backup_writeFirst(m[iFirst:iLast-iFirst+1][0:N])
# else
    #error unknown TXS setting
# endif
    for (int txIter = 0; txIter<ITERS_PER_TX; ++txIter, ++*iter) {
#endif
      fprintf(stderr, "rank %*d/%d starting iter %*d/%d\n",
              digits(size-1), rank, size, digits(ITERS-1), *iter, ITERS);
      fflush(stderr);
      //printMatrix(resFile, *iter, rank, rowsLocal, iBase, m);

      // Only transfer the interior points
#if TXS == 4
      if (txIter > 0)
#endif
      for (int i = iFirst; i <= iLast; i++)
        for (int j = 1; j < N-1; j++)
          mTmp[i][j] = m[i][j];

      // Send up unless I'm at the top, then receive from below
      if (rank < size-1)
        MPI_Send(mTmp[iLast], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
      if (rank > 0)
        MPI_Recv(mTmp[iFirst-1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      // Send down unless I'm at the bottom
      if (rank > 0)
        MPI_Send(mTmp[iFirst], N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
      if (rank < size-1)
        MPI_Recv(mTmp[iLast+1], N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

      // Compute new values (but not on boundary)
      for (int i = iFirst; i <= iLast; i++) {
        for (int j = 1; j < N-1; j++)
          m[i][j] = (mTmp[i][j+1] + mTmp[i][j-1]
                     + mTmp[i+1][j] + mTmp[i-1][j])
                    / 4.0;
      }
#if TXS
    }
  }
#else
  }
#endif

  // Print final matrix.
  fprintf(stderr, "rank %*d/%d completed all %d iterations\n",
          digits(size-1), rank, size, ITERS);
  fflush(stderr);
  printMatrix(resFile, *iter, rank, rowsLocal, iBase, m);
  if (fclose(resFile)) {
    fprintf(stderr, "%s: ", resFileName);
    perror("failure closing as output file");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Compute and print sum of interior as quick verification of results.
  double sum = 0;
  for (int i = iFirst; i <= iLast; i++)
    for (int j = 1; j < N-1; j++)
      sum += m[i][j];
  double allSum;
  MPI_Reduce(&sum, &allSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    printf("sum = %f\n", allSum);
    fflush(stdout);
  }
  nvl_close(heap);

  MPI_Finalize();
  return 0;
}

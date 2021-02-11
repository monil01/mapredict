#ifndef JUGGLER_MAIN_H
#define JUGGLER_MAIN_H

//TODO: These were added later to prevent adding the common benchmark headers
//#define _OPENMP 0
#define TITER 1
#define MSIZE 1
#define BSIZE 1

#define MODE_TASK 1
#define MODE_GLOBAL 2


struct user_parameters {
    int check;
    int succeed;
    char string2display[100];
    int niter;
    int titer;
    int matrix_size;
    //int submatrix_size;
    int blocksize;
    int iblocksize;
    int cutoff_depth;
    int cutoff_size;
    int mode;
    int file;
		char app[10];
		char sched[10];
};

extern double run(struct user_parameters* params);
#endif

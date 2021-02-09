# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
void sweep (int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;
    int block_x, block_y;
    double (*f)[MAT_SIZE] = (double (*)[MAT_SIZE])f_;
    double (*u)[MAT_SIZE] = (double (*)[MAT_SIZE])u_;
    double (*unew)[MAT_SIZE] = (double (*)[MAT_SIZE])unew_;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);
    
#pragma omp parallel                                                    \
    shared(u, unew, f, max_blocks_x, max_blocks_y, nx, ny, dx, dy, itold, itnew, block_size) \
    private(it, block_x, block_y)    
    for (it = itold + 1; it <= itnew; it++)
    {
        // Save the current estimate.
#pragma omp for collapse(2)
        for (block_x = 0; block_x < max_blocks_x; block_x++)
            for (block_y = 0; block_y < max_blocks_y; block_y++)
                copy_block(nx, ny, block_x, block_y, u, unew, block_size);

#pragma omp for collapse(2)
        // Compute a new estimate.
        for (block_x = 0; block_x < max_blocks_x; block_x++)
            for (block_y = 0; block_y < max_blocks_y; block_y++)
                compute_estimate(block_x, block_y, u, unew, f, dx, dy,
                                 nx, ny, block_size);
    }
}

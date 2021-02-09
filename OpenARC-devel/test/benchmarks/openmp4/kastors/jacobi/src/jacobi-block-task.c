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

#pragma omp single
    {
#pragma omp target data map (unew[0:nx][0:ny], u[0:nx][0:ny]) map (to:f[0:nx][0:ny])
        for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
#pragma omp target teams distribute collapse(2) \
firstprivate (max_blocks_x, max_blocks_y, nx, ny, block_size) \
private(block_x, block_y)
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
#pragma omp task shared(u, unew, nx, ny, block_size) firstprivate(block_x, block_y)
                    copy_block(nx, ny, block_x, block_y, u, unew, block_size);
                }
            }


            // Compute a new estimate.
#pragma omp target teams distribute collapse(2) \
firstprivate (max_blocks_x, max_blocks_y, nx, ny, dx, dy, block_size) \
private(block_x, block_y)
            for (block_x = 0; block_x < max_blocks_x; block_x++) {
                for (block_y = 0; block_y < max_blocks_y; block_y++) {
#pragma omp task default(none) shared(u, unew, f, dx, dy, nx, ny, block_size) firstprivate(block_x, block_y)
                    compute_estimate(block_x, block_y, u, unew, f, dx, dy,
                                     nx, ny, block_size);
                }
            }

        } //end of for loop with target data directive.
    }
}

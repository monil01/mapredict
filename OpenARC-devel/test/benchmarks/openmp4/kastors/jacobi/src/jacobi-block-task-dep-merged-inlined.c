# include "poisson.h"

/* #pragma omp task/taskwait version of SWEEP. */
void sweep (int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_, int block_size)
{
    int it;

    double (*f)[MAT_SIZE] = (double (*)[MAT_SIZE])f_;
    double (*u)[MAT_SIZE] = (double (*)[MAT_SIZE])u_;
    double (*unew)[MAT_SIZE] = (double (*)[MAT_SIZE])unew_;

    int block_x, block_y;

    if (block_size == 0)
        block_size = nx;

    int max_blocks_x = (nx / block_size);
    int max_blocks_y = (ny / block_size);

#pragma omp single
   {
#pragma openarc devicetask map(coarse_grained)
#pragma omp target teams map (unew[0:nx][0:ny], u[0:nx][0:ny]) map (to:f[0:nx][0:ny]) \
shared(u, unew, f) private(it, block_x, block_y) \
firstprivate(max_blocks_x, max_blocks_y, nx, ny, dx, dy, block_size)
       for (it = itold + 1; it <= itnew; it++) {
            // Save the current estimate.
#pragma omp distribute collapse(2) private(block_x, block_y)
			for (block_x = 0; block_x < max_blocks_x; block_x++) {
				for (block_y = 0; block_y < max_blocks_y; block_y++) {
#pragma omp task shared(u, unew) firstprivate(block_size, nx, ny, block_x, block_y) \
						depend(in: unew[block_x * block_size: block_size][block_y * block_size: block_size]) \
						depend(out: u[block_x * block_size: block_size][block_y * block_size: block_size])
					{
						int i, j, start_i, start_j;
    					start_i = block_x * block_size;
    					start_j = block_y * block_size;
    					for (i = start_i; i < start_i + block_size; i++) {
							#pragma omp parallel for 
        					for (j = start_j; j < start_j + block_size; j++) {
            					u[i][j] = unew[i][j];
        					}   
    					}   
					}

				}
			}

			// Compute a new estimate.
#pragma omp distribute collapse(2) private(block_x, block_y)
			for (block_x = 0; block_x < max_blocks_x; block_x++) {
				for (block_y = 0; block_y < max_blocks_y; block_y++) {
					int xdm1 = block_x == 0 ? 0 : 1;
					int xdp1 = block_x == max_blocks_x-1 ? 0 : +1;
					int ydp1 = block_y == max_blocks_y-1 ? 0 : +1;
					int ydm1 = block_y == 0 ? 0 : 1;
	#pragma omp task shared(u, unew, f) firstprivate(dx, dy nx, ny, block_size, block_x, block_y, xdm1, xdp1, ydp1, ydm1) \
						depend(out: unew[block_x * block_size: block_size][block_y * block_size: block_size]) \
						depend(in: f[block_x * block_size: block_size][block_y * block_size: block_size], \
								u[block_x * block_size: block_size][block_y * block_size: block_size], \
								u[block_x * block_size - xdm1* block_size: block_size][block_y * block_size: block_size], \
								u[block_x * block_size: block_size][(block_y * block_size + ydp1 * block_size: block_size], \
								u[block_x * block_size: block_size][(block_y * block_size - ydm1 * block_size: block_size], \
								u[(block_x * block_size + xdp1* block_size: block_size][block_y * block_size: block_size])
					{
    					int i, j, start_i, start_j;
    					start_i = block_x * block_size;
    					start_j = block_y * block_size;
    					for (i = start_i; i < start_i + block_size; i++) {
						#pragma omp parallel for 
        					for (j = start_j; j < start_j + block_size; j++) {
            					if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
                					unew[i][j] = f[i][j];
            					} else {
                					unew[i][j] = 0.25 * (u[i-1][j] + u[i][j+1]
                                      + u[i][j-1] + u[i+1][j]
                                      + f[i][j] * dx * dy);
            					}   
        					}   
    					}   
					}

				}
			}
		}
    }
}


void sweep_seq(int nx, int ny, double dx, double dy, double *f_,
        int itold, int itnew, double *u_, double *unew_)
{
    int i;
    int it;
    int j;

    for (it = itold + 1; it <= itnew; it++) {
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                u_[i*ny+j] = unew_[i*ny+j];
            }
        }
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                if (i == 0 || j == 0 || i == nx - 1 || j == ny - 1) {
        			unew_[i*ny+j] = f_[i*ny+j];
        		} else {
        			unew_[i*ny+j] = 0.25 * (u_[(i-1)*ny+j] + u_[i*ny+j+1]
        								  + u_[i*ny+j-1] + u_[(i+1)*ny+j]
        								  + f_[i*ny+j] * dx * dy);
        		}
            }
        }
    }
}




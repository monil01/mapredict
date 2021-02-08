
void validate_vec_add_2d(
    const int x, const int y, const double* d_a);
void validate_vec_add_2d_elemental(
    const int x, const int y, const double* d_a);
void validate_vec_add_reverse_indirection(
    const int x, const int y, const double* d_a);
void validate_vec_add_column_indirection(
    const int x, const int y, const double* d_a);
void validate_two_pt_stencil(
    const int x, const int y, const double* d_a);
void validate_two_pt_stencil_dist_10(
    const int x, const int y, const double* d_a);
void validate_vec_add_sqrt(const int x, const int y, double* d_a);
void validate_vec_add_and_mul(
    const int x, const int y, const double* d_a);
void validate_compute_bound(
    const int x, const int y, const double* d_a);
void validate_five_pt_stencil_2d(
    const int x, const int y, const double delta, double* d_a);
void validate_nine_pt_stencil_2d(
    const int x, const int y, const double alpha, const double beta, 
    const double delta, const double* d_a);
void validate_seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, const double beta, 
    double* d_a);
void validate_twenty_seven_pt_stencil_3d(
    const int x, const int y, const int z, const double alpha, const double beta,
    const double delta, const double gamma, const double* d_a);
void validate_five_pt_wavefront(
    const int x, const int y, const int z, double* d_a);
void validate_even_odd_divergence(
    const int x, const int y, const double* d_a, const double* d_d);
void validate_tealeaf_cheby_iter(
    const int x, const int y, const int halo_depth, const double* d_g);
void validate_cloverleaf_energy_flux(
    const int x, const int y, const int halo_depth, const double* d_f);
void validate_dense_mat_vec(
    const int x, const int y, const double* d_g);
void validate_snap_sweep(
    const int x, const int y, const double* d_g);
void validate_matrix_multiply(
    const int x, const int y, double* d_a, const double* d_b);


$benchnum  = '357';
$benchname = 'csp';
$exename   = 'csp';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol   = 0.2;
$abstol   = 1.0e-6;
$skiptol  = 1;

%workloads = ( 'test'  => [ [ '356.sp' ] ],
               'train' => [ [ '356.sp' ] ],
               'ref'   => [ [ '356.sp' ] ],
             );

$bench_flags = "-DSPEC";

$need_math = 'yes';

@sources=qw(
add.c
adi.c
error.c
exact_rhs.c
exact_solution.c
initialize.c
rhs.c
print_results.c
set_constants.c
sp.c
txinvr.c
verify.c
);

#ninvr.c
#pinvr.c
#x_solve.c
#y_solve.c
#z_solve.c
#tzetar.c

sub invoke {
    my ($me) = @_;
    my $name;
    my @rc;

    my $exe = $me->exe_file;

    push (@rc, { 'command' => $exe,
                 'args'    => [ ],
                 'output'  => "sp_out.log",
                 'error'   => "sp_out.err",
              } );
    return @rc;
}


1;

$benchnum  = '370';
$benchname = 'bt';
$exename   = 'bt';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol   = 0.2;
$abstol   = 1.0e-6;
$skiptol  = 1;

%workloads = ( 'test'  => [ [ '370.bt' ] ],
               'train' => [ [ '370.bt' ] ],
               'ref'   => [ [ '370.bt' ] ],
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
solve_subs.c
x_solve.c
y_solve.c
z_solve.c
print_results.c
set_constants.c
bt.c
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
                 'output'  => "bt_out.log",
                 'error'   => "bt_out.err",
              } );
    return @rc;
}


1;

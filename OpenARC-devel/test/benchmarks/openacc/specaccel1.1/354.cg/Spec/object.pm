$benchnum  = '354';
$benchname = 'cg';
$exename   = 'cg';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol   = 0.2;
$abstol   = 1.0e-6;
$skiptol  = 1;

$bench_flags = "-DSPEC";

$need_math = 'yes';

@sources=qw(
cg.c
print_results.c
randdp.c
c_timers.c
wtime.c
);

sub invoke {
    my ($me) = @_;
    my $name;
    my @rc;

    my $exe = $me->exe_file;

    push (@rc, { 'command' => $exe,
                 'args'    => [ ],
                 'output'  => "cg_out.log",
                 'error'   => "cg_out.err",
              } );
    return @rc;
}


1;

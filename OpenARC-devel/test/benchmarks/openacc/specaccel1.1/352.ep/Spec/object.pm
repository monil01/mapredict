$benchnum  = '352';
$benchname = 'ep';
$exename   = 'ep';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol   = 0.2;
$abstol   = 1.0e-6;

$bench_flags = "-DSPEC";

$need_math = 'yes';

@sources=qw(
ep.c
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
                 'output'  => "ep_out.log",
                 'error'   => "ep_out.err",
              } );
    return @rc;
}


1;

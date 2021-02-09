$benchnum  = '314';
$benchname = 'omriq';
$exename   = 'omriq_exe';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol    = 0.002;
$abstol    = 0.002;

@sources = qw(
              main.c
              file.c
              pbcommon_sources/parboil.c
             );

$need_math = 'yes';
$bench_flags = "-I. -I./pbcommon_sources";

%workloads = ( 'test'  => [ [ '114.mriq' ] ],
               'train' => [ [ '114.mriq' ] ],
               'ref'   => [ [ '114.mriq' ] ],
             );


sub invoke {
    my ($me) = @_;
    my @rc;

    my @runs = grep { !m/^#/ } main::read_file('control');
    my $exe = $me->exe_file;

    foreach my $run (@runs) {
        my ($output, $error, @args) = split(/\s+/, $run);
        push (@rc, { 'command' => $exe,
                     'args'    => [ @args ],
                     'output'  => "$output",
                     'error'   => "$error",
                   } );
    }
    return @rc;
}


1;

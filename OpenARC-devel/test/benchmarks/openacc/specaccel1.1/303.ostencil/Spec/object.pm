$benchnum  = '303';
$benchname = 'ostencil';
$exename   = 'ostencil_exe';
$benchlang = 'C';
@base_exe  = ($exename);

$reltol    = 0.0025;

@sources = qw(
              main.c
              file.c
              kernels.c
	      pbcommon_sources/parboil.c
             );

$need_math = 'yes';
$bench_flags = "-I./ -I./pbcommon_sources";

%workloads = ( 'test'  => [ [ '103.stencil' ] ],
               'train' => [ [ '103.stencil' ] ],
               'ref'   => [ [ '103.stencil' ] ],
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

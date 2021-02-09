$benchnum  = '304'; 
$benchname = 'olbm';
$exename   = 'olbm';
$benchlang = 'C';
@base_exe  = ($exename);

$abstol =  0.0000001;

@sources = qw( lbm.c main.c );

$need_math = 'yes';

$bench_flags = "-DSPEC";

sub invoke {
    my ($me) = @_;
    my $name = $me->name;
    open ARGUMENTS, "<lbm.in" ;
    my $arguments;
    chomp($arguments = <ARGUMENTS>);
    close ARGUMENTS;

    return ({ 'command' => $me->exe_file, 
		 'args'    => [ "$arguments" ], 
		 'error'   => "lbm.err",
		 'output'  => "lbm.out",
		});
}

1;

[Default Nested-Parallel Loop Mapping Strategy in OpenARC]
 - Apply gang clause to the outermost loop.
 - If a worker clause does not exist in any of inner loops, collapse all the inner loops and apply worker clause.

[Advanced Nested-Parallel Loops Mapping Strategy]
- OpenMP on CPU (convertOpenMPtoOpenMP, convertOpenACCtoOpenMP4, convertOpenACCtoOpenMP3, deviceType = 1)
    - Apply simd clause to the inner-most loop if vector-friendly.
        - Apply independent clause otherwise (if parallelizable).
    - Collapse all parallelizble loops except for the innermost simd loop if existing
- OpenMP on GPU (convertOpenMPtoOpenMP, convertOpenACCtoOpenMP4, convertOpenACCtoOpenMP3, deviceType = 0)
    - Collapse all parallelizble loops ignoring simd clause.
    - If there exist multiple nested parallelizable loops, apply the teams distribute and parallel for directives in a nested fashion, starting from the outermost loop
- OpenACC (convertOpenMP3toOpenACC, convertOpenMP4toOpenACC, defaults)
    - Apply simd clause to the inner-most loop if vector-friendly.
        - Apply independent clause otherwise (if parallelizable).
    - Collapse all parallelizble loops except for the innermost simd loop if existing
        - Apply independent directive to each parallelizable loop.

[Commandline Options to Enable Advanced Nested-Parallel Loop Mapping]
//Generate output OpenMP or OpenACC source files.
- To translate from OpenMP3 to OpenACC
    ompaccInter=1 //Not supported yet 
- To translate from OpenMP4 to OpenACC (generate output OpenACC)
    ompaccInter=2
    SkipGPUTranslation=1
    //To apply advanced mapping strategy, add the following options too.
	//(Without these options, OpenARC's default mapping will be applied.)
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=0 
    AccParallelization=2
	//To test this mapping, use advmap1.c as an input file.
	//(Try different input patterns by changing the macro MAPPING from 1 to 7.)
	//	$ O2GBuild.script advamp1.c
- To translate from OpenACC to OpenACC (generate output OpenACC)
    SkipGPUTranslation=1
    //To apply advanced mapping strategy, add the following options too.
	//(Without these options, OpenARC's default mapping will be applied.)
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=0 
    AccParallelization=2
	//To test this mapping, use advmap2.c as an input file.
	//(Try different input patterns by changing the macro MAPPING from 1 to 7.)
	//	$ O2GBuild.script advamp2.c
- To translate from OpenACC to OpenMP3
    ompaccInter=3
    //To apply advanced mapping strategy, add the following options too.
	//(Without these options, OpenARC's default mapping will be applied.)
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=1 //for CPU target
    AccParallelization=2
	//To test this mapping, use advmap2.c as an input file.
	//(Try different input patterns by changing the macro MAPPING from 1 to 7.)
	//	$ O2GBuild.script advamp2.c
- To translate from OpenACC to OpenMP4
    ompaccInter=4
    //To apply advanced mapping strategy, add the following options too.
	//(Without these options, OpenARC's default mapping will be applied.)
    AccParallelization=2
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=0 //for GPU target
    or
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=1 //for CPU target
	//To test this mapping, use advmap2.c as an input file.
	//(Try different input patterns by changing the macro MAPPING from 1 to 7.)
	//	$ O2GBuild.script advamp2.c
- To translate from OpenMP to OpenMP
    ompaccInter=5
    //To apply advanced mapping strategy, add the following options too.
	//(Without these options, OpenARC's default mapping will be applied.)
    AccParallelization=2
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=0 //for GPU target
    or
    parallelismMappingStrat=enableAdvancedMapping=1:vectorize=0:devicetype=1 //for CPU target
	//To test this mapping, use advmap1.c as an input file.
	//(Try different input patterns by changing the macro MAPPING from 1 to 7.)
	//	$ O2GBuild.script advamp1.c

[Brief Description of Commandline Options]
########################################
#Option: ompaccInter
########################################
#ompaccInter=N
#Interchange OpenACC directives with OpenMP 3.0 or OpenMP 4.0 directives: 
#        =0 disable this option (default)
#        =1 generate OpenACC directives from OpenMP 3.0 directives
#        =2 generate OpenACC directives from OpenMP 4.0 directives
#        =3 generate OpenMP 3.0 directives from OpenACC directives
#        =4 generate OpenMP 4.0 directives from OpenACC directives
#        =5 generate optimized OpenMP directives from OpenMP directives
#
########################################
#Option: parallelismMappingStrat
########################################
#parallelismMappingStrat=enableAdvancedMapping=number:vectorize=number:devicetype=number
#Set preferred parallelism mapping strategies; 
#enabledadvancedmapping = 0 (disable all advanced mapping strategies (default))
#                         1 (enable advanced mapping strategies)
#vectorize = 0 (vectorize vector or simd loops only if they are vector-friendly (default))
#            1 (vectorize all the loops with explicit vector or simd clauses)
#devicetype = 0 (choose mapping strategies preferred for GPU architectures (default))
#             1 (choose mapping strategies preferred for CPU architectures)
#             2 (choose mapping strategies preferred for FPGA architectures)
#
########################################
#Option: AccParallelization
########################################
#AccParallelization=N
#Find parallelizable loops
#      =0 disable automatic parallelization analysis (default) 
#      =1 add independent clauses to existing OpenACC loop constructs if they are parallelizable 
#      =2 add independent clauses to any loops in compute contructs if they are parallelizable
#         and do not have seq clauses.
#
########################################
#Option: SkipGPUTranslation
########################################
#SkipGPUTranslation=N
#Skip the final GPU translation
#        =1 exit after all analyses are done (default)
#        =2 exit before the final GPU translation
#        =3 exit after private variable transformaion
#        =4 exit after reduction variable transformation
#
########################################
#Option: macro
########################################
#macro
#Sets macros for the specified names with comma-separated list (no space is allowed)
#e.g., -macro=ARCH=i686,OS=linux
#

[Optional Directives to Change Mapping Behaviors or Bugs]
- OpenARC vectorfriendly clause, which can be added to a loop to tell the compiler 
  that the attached loop is vector-friendly, if the compiler fails to analyze this.
	#pragma openarc transform vectorfriendly

- OpenARC novectorize clause, which can be added to a loop to tell the compiler 
  that the attached loop should not be vectorized, if the compiler incorrectly vectorizes the loop.
	#pragma openarc transform novectorize

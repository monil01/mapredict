[Prerequisite]
- Compile both OpenARC compiler and runtime as instructed in ${openarc}/README.md file.

[Step1: To translate OpenACC code to MCL code]
- Set a shell environment varaible, mclinstallroot to the MCL install root directory.
	//For example, in a bash shell
	export mclinstallroot=${HOME}/local
- Set the "targetArch" OpenARC commandline option in the "openarcConf_NORMAL.txt" file to 4.
	targetArch=4
- Run the "O2GBuild.script" script.
	$ ./O2GBuild.script

[Step2: To compile the translated MCL code]
- Set a shell environment varaible, mclinstallroot to the MCL install root directory.
- Compile and build a "libmclext.a" library in the ${openarc}/openarcrt directory.
	$ cd ${openarc}/openarcrt
	$ make mcl
- Compile the MCL code using the "TCPU" make option.
	//In the current benchmark directory where this file resides
	//On MAC OS X, set the CLIBS2 option in Makefile to an alternative one.
	$ make TCPU

[Step3: To run the compiled binary]
- Launch MCL scheduler if not running (see README file in the MCL repository.)
- Launch the compiled binary
	$cd bin; matmul_TCPU

[To compile and run other OpenACC benchmarks in the OpenARC repository]
- Add MCL-related flags and libraries to CFLAGS2 and CLIBS2 macros in the makefile of the target benchmark (see Makefile in the current directory).
- Repeat the above three steps (Step1, Step2, and Step3)
- To compile the translated MCL code without using the default makefile:
	- Compile the translated MCL code together with "mcl_accext.cpp" and "mcl_accext.h" files in ${openarc}/openarcrt directory.
		(Or, link "libmclext.a" library in the ${openarc}/openarcrt directory, in addition to the standard MCL libraries.)

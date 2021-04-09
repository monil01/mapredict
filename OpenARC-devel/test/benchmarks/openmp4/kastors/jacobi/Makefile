include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = jacobi

########################################
# Set the input C source files (CSRCS) #
########################################
#CSRCS = main.c jacobi.c ./src/jacobi-seq.c ./src/poisson.c
CSRCS = main.c jacobi.c ./src/poisson.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
MAT_SIZE ?= 512
DEFSET_CPU = -DMAT_SIZE=$(MAT_SIZE) 
DEFSET_LLVM = -DMAT_SIZE=$(MAT_SIZE) -Warc,-macro=MAT_SIZE=$(MAT_SIZE) $(ARCMACRO)
DEFSET_ACC = -DMAT_SIZE=$(MAT_SIZE)

##########################################################
# Makefile options that the user can overwrite           #
# OMP: set to 1 to use OpenMP (default: 0)               # 
# MODE: set to profile to use a built-in profiling tool  #
#       (default: normal)                                #
#       If this is set to profile, the runtime system    #
#       will print profiling results according to the    #
#       verbosity level set by OPENARCRT_VERBOSITY       #
#       environment variable.                            # 
# AOCL_FLAGS: set Altera OpenCL Compiler (AOC) flags     #
#    - commonly used options                             #
#      -march=emulator //compile a kernel for emulation  #
#      -v //show progress of the compilation on-screen   # 
#      -c //compile the kernel and generate a Quartus II #
#         //hardware design project without creating a   #
#         //hardware configuration file.                 #
#      -profile //instrument the OpenCL kernel pipeline  #
#                //with performance counters.            #
#      -report  //display estimated resource usage on    #
#                //the screen.                           #
#    (default: -march=emulator)                          #
# AOCL_BOARD: set a target Altera FPGA board             #
#    - "-board=$(AOCL_BOARD)" will be added to the AOC   #
#     in addition to the above flags                     # 
#    - Examples                                          #
#    p385_hpc_d5 //for Stratix V                         #
#    p510t_sch_ax115 //for Arria 10 (Nallatech 510T)     #
##########################################################
OMP ?= 1
MODE ?= normal
#AOCL_BOARD ?= p385_hpc_d5
AOCL_BOARD ?= p510t_sch_ax115 
#AOCL_FLAGS ?= -march=emulator
#AOCL_FLAGS ?= -march=emulator -c
#AOCL_FLAGS ?= -v -g -c -report
AOCL_FLAGS ?= -v -c -report
#AOCL_FLAGS ?= -v -report
POSTCMD ?= rm -f jacobi.c

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
#   - To use relative paths, assume that the directory  #
#   containing this Makefile is the current directory.  #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#   - To use relative paths, assume that cetus_output   #
#   is the current directory.                           #
#########################################################
CFLAGS1 = -I./include -I../common 
#CFLAGS1_LLVM =  
CFLAGS2 = -I../include -I../../common 
# For Translated CPU
CLIBS1 = -lm
# For LLVM
CLIBS1_LLVM = -lm
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../../make.template

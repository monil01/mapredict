include ../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = gpu_fpga_matmul_cg

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = gpu_fpga_matmul_cg_FPGA.c gpu_fpga_matmul_cg_GPU.c

#########################################
# Set macros used for the input program #
#########################################
#_N_ ?= 512
#DEFSET_CPU = -D_N_=$(_N_)
#DEFSET_ACC = -D_N_=$(_N_)

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
#    p520_max_sg280l //for Stratix 10 (Nallatech 520)    #
# PRECMD: optional commands to be executed before the    #   
#    main compilation                                    #   
# POSTCMD: optional commands to be executed after the    #   
#    main compilation                                    #
##########################################################
OMP ?= 1
MODE ?= normal
AOCL_BOARD ?= p520_max_sg280l
#AOCL_BOARD ?= a10pl4_dd4gb_gx115_m512
#AOCL_FLAGS ?= -march=emulator
#AOCL_FLAGS ?= -march=emulator -c
#AOCL_FLAGS ?= -v -g -c -report
AOCL_FLAGS ?= -v -c -report
#AOCL_FLAGS ?= -v -report
POSTCMD = cp ../../../../openarcrt/Makefile.buildIRISKernels bin/Makefile

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
# CFLAGS1 = -I/home/CCUSC/tuna/xmpf/bebop/sparse_matrix_converter/include -I/home/CCUSC/tuna/xmpf/bebop/bebop_util/include
#CFLAGS2 =  
# CLIBS1 = -lm
#CLIBS2 = -L/home/CCUSC/tuna/xmpf/bebop/sparse_matrix_converter -L/home/CCUSC/tuna/xmpf/bebop/bebop_util -lsparse_matrix_converter -lbebop_util

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../make.template

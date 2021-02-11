####################################################################
# - Modify the following makefile macros according to user's       #
# environment.                                                     #
# (Example settings can be found in the "Set the target system"    #
#  section below.)                                                 # 
#   - Set OPENARCLIB to the path to openarcrt directory.           #
#       - Set OPENARC_ARCH to 0 if targeting CUDA GPUs             #
#                         1 if targeting OpenCL devices            #
#                         2 if targeting OpenCL for Xeon Phi       #
#                         3 if targeting OpenCL for Intel FPGAs    #
#                         4 if targeting MCL with OpenCL           #
#                         5 if targeting AMD HIP                   #
#                         6 if targeting Iris (Brisbane)           #
#   - Modify target-specific compilation configuration.            #
#       GOMP_FLAGS : compiler flag to enable OpenMP                #
#       GOMP_LIBS  : library to add for OpenMP                     #
#       GPTHREADS_FLAGS : compiler flag to enable Pthreads         #
#       GPTHREADS_LIBS  : library to add for Pthreads              #
#       CC         : frontend C compiler                           # 
#       CXX        : frontend C++ compiler, used to compile        #
#                    OpenARC-generated output program              #
#       CPP        : C preprocessor called by OpenARC              #
#       BACKENDCXX : backend compiler, used to compile OpenARC     #
#                    runtimes (e.g., nvcc for CUDA runtimes)       #
#       GMACROS    : global compiler flags                         #
#       GPROFILEMACROS : macros used to compile profile-version of #
#                        OpenARC runtimes, which should set macro  # 
#                        _OPENARC_PROFILE_ to 1                    #
#       GBACKEND_DEBUG : debug flags for the backend compiler      #
#                     a debugger mode                              # 
#       CC_FLAGS_PREFIX : prefix used to directly pass compiler    #
#                         options in NVCC (Not needed for GCC)     #
#       GBACKEND_FLAGS : flags for the backend compiler            #
#       GBACKEND_INCLUDES : include path for the backend compiler  #
#       GBACKEND_LIBS : libraries for the backend compiler         #
#       GBACKEND_DEBUG : debug flags for the backend compiler      #
#       GFRONTEND_FLAGS : flags for the frontend compiler          #
#       GFRONTEND_INCLUDES : include path for the frontend compiler#
#       GFRONTEND_LIBS : libraries for the frontend compiler       #
#       GFRONTEND_DEBUG : debug flags for the frontend compiler    #
#   - Modify target-specific LLVM backend configuration (required  #
#     only if LLVM support is enabled):                            #
#       LLVM_TARGET_TRIPLE      : LLVM target triple string        #
#       LLVM_TARGET_DATA_LAYOUT : LLVM target data layout string   #
#       PERL                    : location of perl                 #
#   - Modify target-specific test configuration (required only for #
#     some examples, benchmarks, or other tests):                  #
#       MPI_FLAGS       : compiler flags for MPI                   #
#       MPI_INCLUDES    : include path for MPI                     #
#       MPI_LIBDIR      : library path for MPI                     #
#       MPI_EXEC        : mpiexec command                          #
#       FC              : fortran compiler                         #
#       SPEC_CPU2006    : SPEC CPU 2006 installation directory     #
#       SPEC_CFG        : SPEC CPU 2006 config file (if value does #
#                         not contain "/", then it names a sample  #
#                         file included with OpenARC)              #
#       JUNIT_CLASSPATH : Java classpath when building and running #
#                         JUnit test suite. See test/junit/README  #
#                         for Java packages that must be located   #
#                         here.                                    #
#       NVM_TESTDIR     : directory for test suite to write files  #
#                         intended for an NVM device. Defaults to  #
#                         a temporary HDD directory. Setting this  #
#                         to a directory on a RAM disk or on an    #
#                         actual NVM device will accelerate the    #
#                         NVM tests significantly.                 #
#   - Specify availability of Intel's NVML libraries from pmem     #
#     (required only for NVL runtime support):                     #
#       HAVE_PMEM_NVML : whether OpenARC's nvl/pmem-nvml submodule #
#                        has been cloned and built (1 for true and #
#                        otherwise false)                          #
#                                                                  #
# When specifying CPP, you must specify a C preprocessor and       #
# command-line options that disable C language extensions that     #
# OpenARC does not support.  For example, on OS X when CPP is      #
# clang, we have found that /usr/include/stdlib.h uses clang's     #
# block reference extension.  To suppress that usage, we specify   #
# CPP as gcc 4.2 (from MacPorts package apple-gcc42) or gcc 6.1.0  #
# (from MacPorts package gcc6) with the -E and -std=c99 options.   #
# (CF: GCC (Apple LLVM version 6.1.0) on OSX V10.10 ignores        #
# -std=c99 option.)                                                #
# ==> From OSX V10.14, clang and clang++ should be used for CC and #
#     CXX, respectively, while still using gcc for CPP.            #
#                                                                  #
# LLVM_TARGET_TRIPLE and LLVM_TARGET_DATA_LAYOUT describe the      #
# target platform to LLVM.  If you have clang installed, you can   #
# usually use it to find the correct strings for your native       #
# target.  For example, in our OS X installation, we see this:     #
#                                                                  #
#   $ echo 'int main() {return 0;}' > test.c                       #
#   $ clang -S -emit-llvm -o- test.c | grep ^target                #
#   target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"    #
#   target triple = "x86_64-apple-macosx10.9.0"                    #
#                                                                  #
# For more information about these strings, see:                   #
#                                                                  #
#   http://llvm.org/docs/LangRef.html#target-triple                #
#   http://llvm.org/docs/LangRef.html#data-layout                  #
#                                                                  #
# MPI_INCLUDES and MPI_LIBDIR are required to build MPI support    #
# into the NVL runtime.                                            #
#                                                                  #
####################################################################
###################
# OpenARC Setting #
###################
OPENARCLIB ?= $(openarc)/openarcrt
OPENARCINCLUDE ?= $(openarc)/openarcrt
CETUS_OUTPUT = cetus_output

########################################################################
# Set the target system                                                #
########################################################################
# OPENCL: Linux system with general OpenCL devices                     #
########################################################################
TARGET_SYSTEM = OPENCL

#########################
# Target System: OPENCL #
#########################
ifeq ($(TARGET_SYSTEM),OPENCL)

OPENARC_ARCH = 1
GOMP_FLAGS = -fopenmp
GOMP_LIBS = -lgomp
GPTHREADS_FLAGS = 
GPTHREADS_LIBS = -lpthread
CC = gcc
CXX = g++
CPP = $(CC) -E
GMACROS = -D OPENARC_ARCH=$(OPENARC_ARCH)
GPROFILEMACROS = -D _OPENARC_PROFILE_=1
BACKENDCXX = g++
CC_FLAGS_PREFIX = 
GBACKEND_FLAGS = -O3 -fPIC
GBACKEND_LIBS = -lOpenCL
GBACKEND_DEBUG = -g
GFRONTEND_FLAGS = -O3
GFRONTEND_INCLUDES =
GFRONTEND_LIBS = -lOpenCL
GFRONTEND_DEBUG = -g

endif


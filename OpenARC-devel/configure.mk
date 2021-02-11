include make.header

##############################################
# make utilities                             #
##############################################

# $(call cmd2abs, CMD)
#
# Returns CMD specified absolutely if CMD contains "/" or if CMD is found in
# $PATH.  Otherwise, returns just CMD.
#
# The seemingly redundant "< /dev/null" is needed, or make fails to run the
# "type" command on newark.  The trouble is that "type" is a shell built-in,
# and make on some platforms tries to run it as an executable if the command
# line doesn't contain any special shell characters.
cmd2abs = $(strip $(if $(findstring /, $(filter-out /%, \
                                         $(firstword $(1)))), \
                       $(shell pwd)/$(strip $(firstword $(1))), \
                       $(if $(shell type -P $(firstword $(1)) </dev/null), \
                            $(shell type -P $(firstword $(1)) </dev/null), \
                            $(firstword $(1)))) \
                  $(wordlist 2, $(words $(1)), $(1)))

##############################################
# Intel's NVML libraries from pmem           #
##############################################

ifeq ($(HAVE_PMEM_NVML), 1)
  PMEM_INCLUDES=$(openarc)/nvl/pmem-nvml/src/include
  PMEM_LIBDIR=$(openarc)/nvl/pmem-nvml/src/nondebug
endif

COMMON_DEPS = configure.mk make.header

BUILD_CFG_DIR  = class/openacc/exec
BUILD_CFG      = $(BUILD_CFG_DIR)/build.cfg
BUILD_PCFG      = $(BUILD_CFG_DIR)/build.pcfg

OPENARC_CC_IN  = src/openarc-cc.in
OPENARC_CC_DIR = bin
OPENARC_CC     = $(OPENARC_CC_DIR)/openarc-cc

.PHONY: base pbase
.PHONY: llvm
base: $(BUILD_CFG)
pbase: $(BUILD_PCFG)
llvm: $(OPENARC_CC)

$(BUILD_CFG): $(COMMON_DEPS)
	mkdir -p $(BUILD_CFG_DIR)
	echo '# WARNING: This is a generated file. Do not edit.' > $@
	echo 'cpp = $(call cmd2abs, $(CPP))' >> $@
	echo 'cxx = $(call cmd2abs, $(CXX))' >> $@
	echo 'llvmTargetTriple = $(LLVM_TARGET_TRIPLE)' >> $@
	echo 'llvmTargetDataLayout = $(LLVM_TARGET_DATA_LAYOUT)' >> $@
	echo 'mpi_includes = $(MPI_INCLUDES)' >> $@
	echo 'mpi_libdir = $(MPI_LIBDIR)' >> $@
	echo 'mpi_exec = $(MPI_EXEC)' >> $@
	echo 'fc = $(call cmd2abs, $(FC))' >> $@
	echo 'spec_cpu2006 = $(SPEC_CPU2006)' >> $@
	echo 'spec_cfg = $(SPEC_CFG)' >> $@
	echo 'pmem_includes = $(PMEM_INCLUDES)' >> $@
	echo 'pmem_libdir = $(PMEM_LIBDIR)' >> $@
	echo 'nvm_testdir = $(NVM_TESTDIR)' >> $@

$(BUILD_PCFG): $(COMMON_DEPS)
	mkdir -p $(BUILD_CFG_DIR)
	echo '# WARNING: This is a generated file. Do not edit.' > $@
	echo 'cpp = $(CPP)' >> $@
	echo 'cxx = $(CXX)' >> $@
	echo 'llvmTargetTriple = $(LLVM_TARGET_TRIPLE)' >> $@
	echo 'llvmTargetDataLayout = $(LLVM_TARGET_DATA_LAYOUT)' >> $@
	echo 'mpi_includes = $(MPI_INCLUDES)' >> $@
	echo 'mpi_libdir = $(MPI_LIBDIR)' >> $@
	echo 'mpi_exec = $(MPI_EXEC)' >> $@
	echo 'fc = $(FC)' >> $@
	echo 'spec_cpu2006 = $(SPEC_CPU2006)' >> $@
	echo 'spec_cfg = $(SPEC_CFG)' >> $@
	echo 'pmem_includes = $(PMEM_INCLUDES)' >> $@
	echo 'pmem_libdir = $(PMEM_LIBDIR)' >> $@
	echo 'nvm_testdir = $(NVM_TESTDIR)' >> $@
	mv $(BUILD_PCFG) $(BUILD_CFG)

$(OPENARC_CC): $(OPENARC_CC_IN) $(COMMON_DEPS)
	mkdir -p $(OPENARC_CC_DIR)
	echo '#!$(call cmd2abs, $(PERL))' > $@
	echo '#' >> $@
	echo '# WARNING: This is a generated file. Do not edit.' >> $@
	echo '#' >> $@
	sed \
	  -e 's|@CC@|$(call cmd2abs, $(CC))|g' \
	  -e 's|@CPP@|$(call cmd2abs, $(CPP))|g' \
	  -e 's|@PMEM_INCLUDES@|$(PMEM_INCLUDES)|g' \
	  -e 's|@PMEM_LIBDIR@|$(PMEM_LIBDIR)|g' \
	$< >> $@
	chmod +x $@

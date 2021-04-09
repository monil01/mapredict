# This simple example make file demonstrates how to compile an NVL-C
# program using OpenARC's LLVM backend and NVL runtime.  Before doing
# so, you must build the NVL runtime.  In OpenARC's root directory, see
# nvl/README for details.

include ../../../../make.header

NVLCC = ../../../../bin/openarc-cc

all: example $(if $(PMEM_LIBDIR), example-pmemobj example-pmemobj-norefs)

example: example.c
	$(NVLCC) -o $@ example.c -lnvlrt

example-pmemobj: example-pmemobj.c
	$(NVLCC) -o $@ example-pmemobj.c -lnvlrt-pmemobj -lpmemobj -lpmem

example-pmemobj-norefs: example-pmemobj.c
	$(NVLCC) -o $@ example-pmemobj.c \
	         -fno-nvl-add-ref-counting \
	         -lnvlrt-pmemobj-norefs -lpmemobj -lpmem

.PHONY: neat clean

neat:
	rm -rf example.nvl cetus_output

clean: neat
	rm -rf example example-pmemobj example-pmemobj-norefs

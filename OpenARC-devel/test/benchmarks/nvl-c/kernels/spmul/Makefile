include ../../../../../make.header

SHADOW_UPDATE_COST_MODE ?= NVLRT_COST_DEFAULT
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -O3 -DSHADOW_UPDATE_COST_MODE=$(SHADOW_UPDATE_COST_MODE)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: spmul spmul-txs1 spmul-txs2 spmul-txs4

spmul: spmul.c
	$(NVLCC) -DTXS=0 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

spmul-txs1: spmul.c
	$(NVLCC) -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

spmul-txs2: spmul.c
	$(NVLCC) -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

spmul-txs4: spmul.c
	$(NVLCC) -DTXS=4 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

.PHONY: run-spmul run-smpul-txs1 run-smpul-txs2 run-smpul-txs4

run-spmul: spmul
	rm -f spmul.nvl
	./$<

run-spmul-txs1: spmul-txs1
	rm -f spmul.nvl
	./$<

run-spmul-txs2: spmul-txs2
	rm -f spmul.nvl
	./$<

run-spmul-txs4: spmul-txs4
	rm -f spmul.nvl
	./$<

.PHONY: neat clean

neat:
	rm -rf spmul.nvl cetus_output

clean: neat
	rm -rf spmul spmul-txs1 spmul-txs2 spmul-txs4

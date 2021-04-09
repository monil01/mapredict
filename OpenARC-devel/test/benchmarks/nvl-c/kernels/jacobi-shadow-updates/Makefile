include ../../../../../make.header
SIZE ?= 4096
ITERS_PER_TX ?= 1
SHADOW_UPDATE_COST_MODE ?= NVLRT_COST_DEFAULT
SIZE2 = $(shell expr $(SIZE) \+ 2)
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
HEAPSIZE ?= $(shell expr $(SIZE2) \* $(SIZE2) \* 24)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=SIZE=$(SIZE) -Warc,-macro=SIZE_2=$(SIZE2) -Warc,-macro=ITERS_PER_TX=$(ITERS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -DSIZE=$(SIZE) -O3 -DHEAPSIZE=$(HEAPSIZE) \
         -DITERS_PER_TX=$(ITERS_PER_TX) \
         -DSHADOW_UPDATE_COST_MODE=$(SHADOW_UPDATE_COST_MODE)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: jacobi-nosafe-norefs \
     jacobi-safe-norefs \
     jacobi-safe-refs \
     jacobi-safe-refs-txs1 \
     jacobi-safe-refs-txs2 \
     jacobi-safe-refs-txs4 \
     jacobi-vheap

jacobi-nosafe-norefs: jacobi.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-safe-norefs: jacobi.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-safe-refs: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

jacobi-safe-refs-txs1: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs2: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs4: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=4 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-vheap: jacobi.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

.PHONY: run-jacobi-nosafe-norefs \
        run-jacobi-safe-norefs \
        run-jacobi-safe-refs \
        run-jacobi-safe-refs-txs1 \
        run-jacobi-safe-refs-txs2 \
        run-jacobi-safe-refs-txs4 \
        run-jacobi-vheap

run-jacobi-nosafe-norefs: jacobi-nosafe-norefs
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-norefs: jacobi-safe-norefs
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs: jacobi-safe-refs
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs1: jacobi-safe-refs-txs1
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs2: jacobi-safe-refs-txs2
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs4: jacobi-safe-refs-txs4
	rm -f jacobi.nvl
	./$<

run-jacobi-vheap: jacobi-vheap
	./$<

.PHONY: neat clean

neat:
	rm -rf jacobi.nvl cetus_output

clean: neat
	rm -rf jacobi-nosafe-norefs \
	       jacobi-safe-norefs \
	       jacobi-safe-refs \
	       jacobi-vheap \
	       jacobi-safe-refs-txs1 \
	       jacobi-safe-refs-txs2 \
	       jacobi-safe-refs-txs4

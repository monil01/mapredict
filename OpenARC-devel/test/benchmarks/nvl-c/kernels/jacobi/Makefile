include ../../../../../make.header
SIZE ?= 4096
ROWS_PER_TX ?= 8
SIZE2 = $(shell expr $(SIZE) \+ 2)
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
HEAPSIZE ?= $(shell expr $(SIZE2) \* $(SIZE2) \* 24)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=SIZE=$(SIZE) -Warc,-macro=SIZE_2=$(SIZE2) -Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -DSIZE=$(SIZE) -O3 -DHEAPSIZE=$(HEAPSIZE) \
	-DROWS_PER_TX=$(ROWS_PER_TX)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: jacobi-nosafe-norefs jacobi-nosafe-norefs-poor \
     jacobi-safe-norefs jacobi-safe-norefs-poor \
     jacobi-safe-refs jacobi-safe-refs-poor \
     jacobi-safe-refs-persist jacobi-safe-refs-persist-poor \
     jacobi-safe-refs-txs1 jacobi-safe-refs-txs1-poor \
     jacobi-safe-refs-txs2 jacobi-safe-refs-txs2-poor \
     jacobi-safe-refs-txs3 jacobi-safe-refs-txs3-poor \
     jacobi-vheap

jacobi-nosafe-norefs: jacobi.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-nosafe-norefs-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-safe-norefs: jacobi.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-safe-norefs-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

jacobi-safe-refs: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

jacobi-safe-refs-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

jacobi-safe-refs-persist: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

jacobi-safe-refs-persist-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

jacobi-safe-refs-txs1: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs2: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs3: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs1-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs2-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-safe-refs-txs3-poor: jacobi.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

jacobi-vheap: jacobi.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

.PHONY: run-jacobi-nosafe-norefs run-jacobi-nosafe-norefs-poor \
        run-jacobi-safe-norefs run-jacobi-safe-norefs-poor \
        run-jacobi-safe-refs run-jacobi-safe-refs-poor \
        run-jacobi-safe-refs-persist run-jacobi-safe-refs-persist-poor \
        run-jacobi-safe-refs-txs1 run-jacobi-safe-refs-txs1-poor \
        run-jacobi-safe-refs-txs2 run-jacobi-safe-refs-txs2-poor \
        run-jacobi-safe-refs-txs3 run-jacobi-safe-refs-txs3-poor \
        run-jacobi-vheap

run-jacobi-nosafe-norefs: jacobi-nosafe-norefs
	rm -f jacobi.nvl
	./$<

run-jacobi-nosafe-norefs-poor: jacobi-nosafe-norefs-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-norefs: jacobi-safe-norefs
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-norefs-poor: jacobi-safe-norefs-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs: jacobi-safe-refs
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-poor: jacobi-safe-refs-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-persist: jacobi-safe-refs-persist
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-persist-poor: jacobi-safe-refs-persist-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs1: jacobi-safe-refs-txs1
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs2: jacobi-safe-refs-txs2
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs3: jacobi-safe-refs-txs3
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs1-poor: jacobi-safe-refs-txs1-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs2-poor: jacobi-safe-refs-txs2-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-safe-refs-txs3-poor: jacobi-safe-refs-txs3-poor
	rm -f jacobi.nvl
	./$<

run-jacobi-vheap: jacobi-vheap
	./$<

.PHONY: neat clean

neat:
	rm -rf jacobi.nvl cetus_output

clean: neat
	rm -rf jacobi-nosafe-norefs jacobi-nosafe-norefs-poor \
	       jacobi-safe-norefs jacobi-safe-norefs-poor \
	       jacobi-safe-refs jacobi-safe-refs-poor \
	       jacobi-safe-refs-persist jacobi-vheap \
	       jacobi-safe-refs-txs1 jacobi-safe-refs-txs1-poor \
	       jacobi-safe-refs-txs2 jacobi-safe-refs-txs2-poor \
	       jacobi-safe-refs-txs3 jacobi-safe-refs-txs3-poor \
	       jacobi-safe-refs-persist-poor 

include ../../../../../make.header
_N_ ?= 512
ROWS_PER_TX ?= 8
SHADOW_UPDATE_COST_MODE ?= NVLRT_COST_DEFAULT
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
HEAPSIZE ?= $(shell expr $(_N_) \* $(_N_) \* 40 )
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc -Warc,-macro=_N_=$(_N_) \
        -Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -D_N_=$(_N_) -O3 -DHEAPSIZE=$(HEAPSIZE) \
         -DROWS_PER_TX=$(ROWS_PER_TX) \
         -DSHADOW_UPDATE_COST_MODE=$(SHADOW_UPDATE_COST_MODE)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: matmul-pmem matmul-pmem-poor \
     matmul-pmem-txs matmul-pmem-txs-poor \
     matmul-nosafe-norefs matmul-nosafe-norefs-poor \
     matmul-safe-norefs matmul-safe-norefs-poor \
     matmul-safe-refs matmul-safe-refs-poor \
     matmul-safe-refs-persist matmul-safe-refs-persist-poor \
     matmul-safe-refs-txs1 matmul-safe-refs-txs1-poor \
     matmul-safe-refs-txs2 matmul-safe-refs-txs2-poor \
     matmul-safe-refs-txs3 matmul-safe-refs-txs3-poor \
     matmul-safe-refs-txs4-poor \
     matmul-vheap

matmul-pmem: matmul.c
	$(NVLCC) -DMEM=PMEMOBJ $(CFLAGS) -o $@ $< $(LIBS)

matmul-pmem-poor: matmul.c
	$(NVLCC) -DMEM=PMEMOBJ -DPOOR $(CFLAGS) -o $@ $< $(LIBS)

matmul-pmem-txs: matmul.c
	$(NVLCC) -DMEM=PMEMOBJ -DTXS $(CFLAGS) -o $@ $< $(LIBS)

matmul-pmem-txs-poor: matmul.c
	$(NVLCC) -DMEM=PMEMOBJ -DTXS -DPOOR $(CFLAGS) -o $@ $< $(LIBS)

matmul-nosafe-norefs: matmul.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

matmul-nosafe-norefs-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

matmul-safe-norefs: matmul.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

matmul-safe-norefs-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

matmul-safe-refs: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

matmul-safe-refs-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

matmul-safe-refs-persist: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

matmul-safe-refs-persist-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

matmul-safe-refs-txs1: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs2: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs3: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs1-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs2-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs3-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-safe-refs-txs4-poor: matmul.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=4 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

matmul-vheap: matmul.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

.PHONY: run-matmul-pmem run-matmul-pmem-poor \
        run-matmul-pmem-txs run-matmul-pmem-txs-poor \
        run-matmul-nosafe-norefs run-matmul-nosafe-norefs-poor \
        run-matmul-safe-norefs run-matmul-safe-norefs-poor \
        run-matmul-safe-refs run-matmul-safe-refs-poor \
        run-matmul-safe-refs-persist run-matmul-safe-refs-persist-poor \
        run-matmul-safe-refs-txs1 run-matmul-safe-refs-txs1-poor \
        run-matmul-safe-refs-txs2 run-matmul-safe-refs-txs2-poor \
        run-matmul-safe-refs-txs3 run-matmul-safe-refs-txs3-poor \
        run-matmul-safe-refs-txs4-poor \
        run-matmul-vheap

run-matmul-pmem: matmul-pmem
	rm -f matmul.nvl
	./$<

run-matmul-pmem-poor: matmul-pmem-poor
	rm -f matmul.nvl
	./$<

run-matmul-pmem-txs: matmul-pmem-txs
	rm -f matmul.nvl
	./$<

run-matmul-pmem-txs-poor: matmul-pmem-txs-poor
	rm -f matmul.nvl
	./$<

run-matmul-nosafe-norefs: matmul-nosafe-norefs
	rm -f matmul.nvl
	./$<

run-matmul-nosafe-norefs-poor: matmul-nosafe-norefs-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-norefs: matmul-safe-norefs
	rm -f matmul.nvl
	./$<

run-matmul-safe-norefs-poor: matmul-safe-norefs-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs: matmul-safe-refs
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-poor: matmul-safe-refs-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-persist: matmul-safe-refs-persist
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-persist-poor: matmul-safe-refs-persist-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs1: matmul-safe-refs-txs1
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs2: matmul-safe-refs-txs2
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs3: matmul-safe-refs-txs3
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs1-poor: matmul-safe-refs-txs1-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs2-poor: matmul-safe-refs-txs2-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs3-poor: matmul-safe-refs-txs3-poor
	rm -f matmul.nvl
	./$<

run-matmul-safe-refs-txs4-poor: matmul-safe-refs-txs4-poor
	rm -f matmul.nvl
	./$<

run-matmul-vheap: matmul-vheap
	./$<

.PHONY: neat clean

neat:
	rm -rf matmul.nvl cetus_output

clean: neat
	rm -rf matmul-pmem matmul-pmem-poor \
	       matmul-pmem-txs matmul-pmem-txs-poor \
	       matmul-nosafe-norefs matmul-nosafe-norefs-poor \
	       matmul-safe-norefs matmul-safe-norefs-poor \
	       matmul-safe-refs matmul-safe-refs-poor \
	       matmul-safe-refs-persist matmul-safe-refs-persist-poor \
	       matmul-safe-refs-txs1 matmul-safe-refs-txs1-poor \
	       matmul-safe-refs-txs2 matmul-safe-refs-txs2-poor \
	       matmul-safe-refs-txs3 matmul-safe-refs-txs3-poor \
	       matmul-safe-refs-txs4-poor \
	       matmul-vheap

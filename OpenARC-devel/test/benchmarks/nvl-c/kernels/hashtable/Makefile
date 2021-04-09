include ../../../../../make.header
ROWS_PER_TX ?= 8
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -O3 \
	-DROWS_PER_TX=$(ROWS_PER_TX)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: hashtable-nosafe-norefs hashtable-nosafe-norefs-poor \
     hashtable-safe-norefs hashtable-safe-norefs-poor \
     hashtable-safe-refs hashtable-safe-refs-poor \
     hashtable-safe-refs-persist hashtable-safe-refs-persist-poor \
     hashtable-safe-refs-txs1 hashtable-safe-refs-txs1-poor \
     hashtable-vheap hashtable-malloc

hashtable-nosafe-norefs: hashtable.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

hashtable-nosafe-norefs-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

hashtable-safe-norefs: hashtable.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

hashtable-safe-norefs-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

hashtable-safe-refs: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

hashtable-safe-refs-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

hashtable-safe-refs-persist: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

hashtable-safe-refs-persist-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

hashtable-safe-refs-txs1: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-safe-refs-txs2: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-safe-refs-txs3: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-safe-refs-txs1-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-safe-refs-txs2-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-safe-refs-txs3-poor: hashtable.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

hashtable-vheap: hashtable.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

hashtable-malloc: hashtable.c
	$(NVLCC) -g -DMEM=0 $(CFLAGS) -o $@ $< -lm

.PHONY: run-hashtable-nosafe-norefs run-hashtable-nosafe-norefs-poor \
        run-hashtable-safe-norefs run-hashtable-safe-norefs-poor \
        run-hashtable-safe-refs run-hashtable-safe-refs-poor \
        run-hashtable-safe-refs-persist run-hashtable-safe-refs-persist-poor \
        run-hashtable-vheap run-hashtable-malloc

run-hashtable-nosafe-norefs: hashtable-nosafe-norefs
	rm -f hashtable.nvl
	./$<

run-hashtable-nosafe-norefs-poor: hashtable-nosafe-norefs-poor
	rm -f hashtable.nvl
	./$<

run-hashtable-safe-norefs: hashtable-safe-norefs
	rm -f hashtable.nvl
	./$<

run-hashtable-safe-norefs-poor: hashtable-safe-norefs-poor
	rm -f hashtable.nvl
	./$<

run-hashtable-safe-refs: hashtable-safe-refs
	rm -f hashtable.nvl
	./$<

run-hashtable-safe-refs-poor: hashtable-safe-refs-poor
	rm -f hashtable.nvl
	./$<

run-hashtable-safe-refs-persist: hashtable-safe-refs-persist
	rm -f hashtable.nvl
	./$<

run-hashtable-vheap: hashtable-vheap
	./$<

run-hashtable-malloc: hashtable-malloc
	./$<

.PHONY: neat clean

neat:
	rm -rf hashtable.nvl cetus_output

clean: neat
	rm -rf hashtable-nosafe-norefs hashtable-nosafe-norefs-poor \
	       hashtable-safe-norefs hashtable-safe-norefs-poor \
	       hashtable-safe-refs hashtable-safe-refs-poor \
	       hashtable-safe-refs-persist hashtable-vheap \
	       hashtable-safe-refs-txs1 hashtable-safe-refs-txs1-poor \
	       hashtable-safe-refs-txs2 hashtable-safe-refs-txs2-poor \
	       hashtable-safe-refs-txs3 hashtable-safe-refs-txs3-poor \
	       hashtable-safe-refs-persist-poor hashtable-malloc

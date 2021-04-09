include ../../../../../make.header
ROWS_PER_TX ?= 8
N ?= 100000
OFFSET ?= 0
STATIC_CPU_MALLOC ?= 0
SIZE = $(shell expr $(N) \+ $(OFFSET))
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
HEAPSIZE ?= $(shell expr $(SIZE) \* 24 \* 3)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -DN=$(N) -DOFFSET=$(OFFSET) -DSTATIC_CPU_MALLOC=$(STATIC_CPU_MALLOC) -O3 -DHEAPSIZE=$(HEAPSIZE) \
	-DROWS_PER_TX=$(ROWS_PER_TX)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: stream-nosafe-norefs stream-nosafe-norefs-poor \
     stream-safe-norefs stream-safe-norefs-poor \
     stream-safe-refs stream-safe-refs-poor \
     stream-safe-refs-persist stream-safe-refs-persist-poor \
     stream-safe-refs-txs1 stream-safe-refs-txs1-poor \
     stream-safe-refs-txs2 stream-safe-refs-txs2-poor \
     stream-safe-refs-txs3 stream-safe-refs-txs3-poor \
     stream-vheap

stream-nosafe-norefs: stream.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

stream-nosafe-norefs-poor: stream.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

stream-safe-norefs: stream.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

stream-safe-norefs-poor: stream.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

stream-safe-refs: stream.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

stream-safe-refs-poor: stream.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

stream-safe-refs-persist: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

stream-safe-refs-persist-poor: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

stream-safe-refs-txs1: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-safe-refs-txs2: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-safe-refs-txs3: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-safe-refs-txs1-poor: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-safe-refs-txs2-poor: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-safe-refs-txs3-poor: stream.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

stream-vheap: stream.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

.PHONY: run-stream-nosafe-norefs run-stream-nosafe-norefs-poor \
        run-stream-safe-norefs run-stream-safe-norefs-poor \
        run-stream-safe-refs run-stream-safe-refs-poor \
        run-stream-safe-refs-persist run-stream-safe-refs-persist-poor \
        run-stream-vheap

run-stream-nosafe-norefs: stream-nosafe-norefs
	rm -f stream.nvl
	./$<

run-stream-nosafe-norefs-poor: stream-nosafe-norefs-poor
	rm -f stream.nvl
	./$<

run-stream-safe-norefs: stream-safe-norefs
	rm -f stream.nvl
	./$<

run-stream-safe-norefs-poor: stream-safe-norefs-poor
	rm -f stream.nvl
	./$<

run-stream-safe-refs: stream-safe-refs
	rm -f stream.nvl
	./$<

run-stream-safe-refs-poor: stream-safe-refs-poor
	rm -f stream.nvl
	./$<

run-stream-safe-refs-persist: stream-safe-refs-persist
	rm -f stream.nvl
	./$<

run-stream-vheap: stream-vheap
	./$<

.PHONY: neat clean

neat:
	rm -rf stream.nvl cetus_output

clean: neat
	rm -rf stream-nosafe-norefs stream-nosafe-norefs-poor \
	       stream-safe-norefs stream-safe-norefs-poor \
	       stream-safe-refs stream-safe-refs-poor \
	       stream-safe-refs-persist stream-vheap \
	       stream-safe-refs-txs1 stream-safe-refs-txs1-poor \
	       stream-safe-refs-txs2 stream-safe-refs-txs2-poor \
	       stream-safe-refs-txs3 stream-safe-refs-txs3-poor \
	       stream-safe-refs-persist-poor 

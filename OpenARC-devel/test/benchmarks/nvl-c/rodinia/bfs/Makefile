include ../../../../../make.header
NODES ?= 4096
EDGES ?= 24576
ROWS_PER_TX ?= 8
HEAPSIZE ?= $(shell expr $(NODES) \* 72 + $(EDGES) \* 12)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -DNUM_OF_NODES=$(NODES) -DEDGELIST_SIZE=$(EDGES) -O3 -DHEAPSIZE=$(HEAPSIZE) \
	-DROWS_PER_TX=$(ROWS_PER_TX)
LD_LIBRARY_PATH=../../../../../nvl/rt:$(PMEM_LIBDIR)

all: bfs-nosafe-norefs bfs-nosafe-norefs-poor \
     bfs-safe-norefs bfs-safe-norefs-poor \
     bfs-safe-refs bfs-safe-refs-poor \
     bfs-safe-refs-persist bfs-safe-refs-persist-poor \
     bfs-safe-refs-txs1 bfs-safe-refs-txs1-poor \
     bfs-safe-refs-txs2 bfs-safe-refs-txs2-poor \
     bfs-safe-refs-txs3 bfs-safe-refs-txs3-poor \
     bfs-vheap bfs-malloc

bfs-nosafe-norefs: bfs.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

bfs-nosafe-norefs-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

bfs-safe-norefs: bfs.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

bfs-safe-norefs-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

bfs-safe-refs: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

bfs-safe-refs-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

bfs-safe-refs-persist: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

bfs-safe-refs-persist-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

bfs-safe-refs-txs1: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-safe-refs-txs2: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-safe-refs-txs3: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-safe-refs-txs1-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-safe-refs-txs2-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-safe-refs-txs3-poor: bfs.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

bfs-vheap: bfs.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

bfs-malloc: bfs.c
	$(NVLCC) -g -DMEM=0 $(CFLAGS) -o $@ $< $(LIBS) -lm

.PHONY: run-bfs-nosafe-norefs run-bfs-nosafe-norefs-poor \
        run-bfs-safe-norefs run-bfs-safe-norefs-poor \
        run-bfs-safe-refs run-bfs-safe-refs-poor \
        run-bfs-safe-refs-persist run-bfs-safe-refs-persist-poor \
        run-bfs-vheap

run-bfs-nosafe-norefs: bfs-nosafe-norefs
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-nosafe-norefs-poor: bfs-nosafe-norefs-poor
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-safe-norefs: bfs-safe-norefs
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-safe-norefs-poor: bfs-safe-norefs-poor
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-safe-refs: bfs-safe-refs
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-safe-refs-poor: bfs-safe-refs-poor
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-safe-refs-persist: bfs-safe-refs-persist
	rm -f bfs.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

run-bfs-vheap: bfs-vheap
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< "../../../../data/rodinia1.0/data/bfs/graph4096.txt"

.PHONY: neat clean

neat:
	rm -rf bfs.nvl cetus_output

clean: neat
	rm -rf bfs-nosafe-norefs bfs-nosafe-norefs-poor \
	       bfs-safe-norefs bfs-safe-norefs-poor \
	       bfs-safe-refs bfs-safe-refs-poor \
	       bfs-safe-refs-persist bfs-vheap \
	       bfs-safe-refs-txs1 bfs-safe-refs-txs1-poor \
	       bfs-safe-refs-txs2 bfs-safe-refs-txs2-poor \
	       bfs-safe-refs-txs3 bfs-safe-refs-txs3-poor \
	       bfs-safe-refs-persist-poor bfs-malloc

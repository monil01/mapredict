include ../../../../../make.header
_M_SIZE ?= 4096
ROWS_PER_TX ?= 8
HEAPSIZE ?= $(shell expr $(_M_SIZE) \* 12)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../../nvl/include -L../../../../../nvl/rt
NVLCC = ../../../../../bin/openarc-cc \
	-Warc,-macro=ROWS_PER_TX=$(ROWS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -DVERIFICATION=1 -D_M_SIZE=$(_M_SIZE) -O3 -DHEAPSIZE=$(HEAPSIZE) \
	-DROWS_PER_TX=$(ROWS_PER_TX)
LD_LIBRARY_PATH=../../../../../nvl/rt:$(PMEM_LIBDIR)

all: lud-nosafe-norefs lud-nosafe-norefs-poor \
     lud-safe-norefs lud-safe-norefs-poor \
     lud-safe-refs lud-safe-refs-poor \
     lud-safe-refs-persist lud-safe-refs-persist-poor \
     lud-safe-refs-txs1 lud-safe-refs-txs1-poor \
     lud-safe-refs-txs2 lud-safe-refs-txs2-poor \
     lud-safe-refs-txs3 lud-safe-refs-txs3-poor \
     lud-vheap lud-malloc

lud-nosafe-norefs: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $^ $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lud-nosafe-norefs-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lud-safe-norefs: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lud-safe-norefs-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lud-safe-refs: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj

lud-safe-refs-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DPOOR -DREFS $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj

lud-safe-refs-persist: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

lud-safe-refs-persist-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

lud-safe-refs-txs1: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-safe-refs-txs2: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-safe-refs-txs3: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-safe-refs-txs1-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-safe-refs-txs2-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-safe-refs-txs3-poor: lud.c lud_omp.c common.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lud-vheap: lud.c lud_omp.c common.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $^ $(LIBS) -lnvlrt-vmem

lud-malloc: lud.c lud_omp.c common.c
	$(NVLCC) -g -DMEM=0 $(CFLAGS) -o $@ $^ $(LIBS) -lm

.PHONY: run-lud-nosafe-norefs run-lud-nosafe-norefs-poor \
        run-lud-safe-norefs run-lud-safe-norefs-poor \
        run-lud-safe-refs run-lud-safe-refs-poor \
        run-lud-safe-refs-persist run-lud-safe-refs-persist-poor \
        run-lud-vheap run_lud-malloc

run-lud-nosafe-norefs: lud-nosafe-norefs
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-nosafe-norefs-poor: lud-nosafe-norefs-poor
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-safe-norefs: lud-safe-norefs
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-safe-norefs-poor: lud-safe-norefs-poor
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-safe-refs: lud-safe-refs
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-safe-refs-poor: lud-safe-refs-poor
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-safe-refs-persist: lud-safe-refs-persist
	rm -f lud.nvl
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

run-lud-vheap: lud-vheap
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) ./$< -v -i "../../../../data/rodinia1.0/data/lud/64.dat"

.PHONY: neat clean

neat:
	rm -rf lud.nvl cetus_output dismatch.txt ludOutput.txt

clean: neat
	rm -rf lud-nosafe-norefs lud-nosafe-norefs-poor \
	       lud-safe-norefs lud-safe-norefs-poor \
	       lud-safe-refs lud-safe-refs-poor \
	       lud-safe-refs-persist lud-vheap \
	       lud-safe-refs-txs1 lud-safe-refs-txs1-poor \
	       lud-safe-refs-txs2 lud-safe-refs-txs2-poor \
	       lud-safe-refs-txs3 lud-safe-refs-txs3-poor \
	       lud-safe-refs-persist-poor lud-malloc

include ../../../../make.header
ITERS_PER_TX ?= 4096
PMEMOBJ_MIN_POOL = $(shell expr 1024 \* 1024 \* 8) # from libpmemobj.h
#HEAPSIZE ?= $(shell expr $(PMEMOBJ_MIN_POOL) \* 2)
HEAPSIZE ?= $(shell expr 250 \* 1000000 \* 3)
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../nvl/include -L../../../../nvl/rt
NVLCC = ../../../../bin/openarc-cc \
	-Warc,-macro=ITERS_PER_TX=$(ITERS_PER_TX)
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -O3 -I. \
	-DITERS_PER_TX=$(ITERS_PER_TX)
LD_LIBRARY_PATH:=../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: xsbench-nosafe-norefs xsbench-nosafe-norefs-poor \
     xsbench-safe-norefs xsbench-safe-norefs-poor  \
     xsbench-safe-refs xsbench-safe-refs-poor  \
     xsbench-safe-refs-persist xsbench-safe-refs-persist-poor  \
     xsbench-safe-refs-txs1 xsbench-safe-refs-txs1-poor \
     xsbench-vheap xsbench-malloc

xsbench-nosafe-norefs: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $^ $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

xsbench-nosafe-norefs-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

xsbench-safe-norefs: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

xsbench-safe-norefs-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

xsbench-safe-refs: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj

xsbench-safe-refs-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj

xsbench-safe-refs-persist: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

xsbench-safe-refs-persist-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DPOOR -DPERSIST $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

xsbench-safe-refs-txs1: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-safe-refs-txs2: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-safe-refs-txs3: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-safe-refs-txs1-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-safe-refs-txs2-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-safe-refs-txs3-poor: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=3 -DPOOR $(CFLAGS) -o $@ $^ $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

xsbench-vheap: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $^ $(LIBS) -lnvlrt-vmem

xsbench-malloc: Main.c CalculateXS.c GridInit.c Materials.c XSutils.c io.c 
	$(NVLCC) -g -DMEM=0 $(CFLAGS) -o $@ $^ -lm

.PHONY: run-xsbench-nosafe-norefs run-xsbench-nosafe-norefs-poor \
        run-xsbench-safe-norefs run-xsbench-safe-norefs-poor \
        run-xsbench-safe-refs run-xsbench-safe-refs-poor \
        run-xsbench-safe-refs-persist run-xsbench-safe-refs-persist-poor \
        run-xsbench-safe-refs-txs1 run-xsbench-safe-refs-txs1-poor \
        run-xsbench-safe-refs-txs2 run-xsbench-safe-refs-txs2-poor \
        run-xsbench-safe-refs-txs3 run-xsbench-safe-refs-txs3-poor \
        run-xsbench-vheap

run-xsbench-nosafe-norefs: xsbench-nosafe-norefs
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-nosafe-norefs-poor: xsbench-nosafe-norefs-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-norefs: xsbench-safe-norefs
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-norefs-poor: xsbench-safe-norefs-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs: xsbench-safe-refs
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-poor: xsbench-safe-refs-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-persist: xsbench-safe-refs
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-persist-poor: xsbench-safe-refs-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs1: xsbench-safe-refs-txs1
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs2: xsbench-safe-refs-txs2
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs3: xsbench-safe-refs-txs3
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs1-poor: xsbench-safe-refs-txs1-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs2-poor: xsbench-safe-refs-txs2-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-safe-refs-txs3-poor: xsbench-safe-refs-txs3-poor
	rm -f xsbench.nvl
	./$< -s small

run-xsbench-vheap: xsbench-vheap
	./$< -s small

.PHONY: neat clean

neat:
	rm -rf xsbench.nvl cetus_output x.out results.txt

clean: neat
	rm -rf xsbench-nosafe-norefs xsbench-nosafe-norefs-poor \
	       xsbench-safe-norefs xsbench-safe-norefs-poor \
	       xsbench-safe-refs xsbench-safe-refs-poor \
	       xsbench-safe-refs-persist xsbench-safe-refs-persist-poor \
	       xsbench-safe-refs-txs1 xsbench-safe-refs-txs1-poor \
	       xsbench-safe-refs-txs2 xsbench-safe-refs-txs2-poor \
	       xsbench-safe-refs-txs3 xsbench-safe-refs-txs3-poor \
	       xsbench-vheap xsbench-malloc

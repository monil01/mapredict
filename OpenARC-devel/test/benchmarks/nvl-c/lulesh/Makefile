include ../../../../make.header
_N_ ?= 10
SHADOW_UPDATE_COST_MODE ?= NVLRT_COST_DEFAULT
CC = gcc -I$(PMEM_INCLUDES) -L$(PMEM_LIBDIR) \
     -I../../../../nvl/include -L../../../../nvl/rt
NVLCC = ../../../../bin/openarc-cc
LIBS = -lvmem -lpmemobj -lpmem -lm
CFLAGS = -D_N_=$(_N_) -O3  \
	-DSHADOW_UPDATE_COST_MODE=$(SHADOW_UPDATE_COST_MODE)
LD_LIBRARY_PATH:=../../../../nvl/rt:$(PMEM_LIBDIR):$(LD_LIBRARY_PATH)

all: lulesh-nosafe-norefs lulesh-nosafe-norefs-poor \
     lulesh-safe-norefs lulesh-safe-norefs-poor  \
     lulesh-safe-refs lulesh-safe-refs-poor  \
     lulesh-safe-refs-persist lulesh-safe-refs-persist-poor  \
     lulesh-safe-refs-txs1 lulesh-safe-refs-txs1-poor \
     lulesh-safe-refs-txs2 lulesh-safe-refs-txs2-poor \
     lulesh-safe-refs-txs4 lulesh-safe-refs-txs4-poor \
     lulesh-vheap lulesh-malloc

lulesh-nosafe-norefs: lulesh.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lulesh-nosafe-norefs-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -fno-nvl-add-safety \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lulesh-safe-norefs: lulesh.c
	$(NVLCC) -DMEM=NVL $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lulesh-safe-norefs-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-norefs -fno-nvl-add-ref-counting

lulesh-safe-refs: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

lulesh-safe-refs-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

lulesh-safe-refs-persist: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

lulesh-safe-refs-persist-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DPOOR -DPERSIST $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-persist -fnvl-add-persists

lulesh-safe-refs-txs1: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-safe-refs-txs2: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-safe-refs-txs4: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=4 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-safe-refs-txs1-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=1 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-safe-refs-txs2-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=2 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-safe-refs-txs4-poor: lulesh.c
	$(NVLCC) -DMEM=NVL -DREFS -DTXS=4 -DPOOR $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs -fnvl-add-txs

lulesh-vheap: lulesh.c
	$(NVLCC) -g -DMEM=VHEAP $(CFLAGS) -o $@ $< $(LIBS) -lnvlrt-vmem

lulesh-malloc: lulesh.c
	$(NVLCC) -g -DMEM=0 $(CFLAGS) -o $@ $< -lm

.PHONY: run-lulesh-nosafe-norefs run-lulesh-nosafe-norefs-poor \
        run-lulesh-safe-norefs run-lulesh-safe-norefs-poor \
        run-lulesh-safe-refs run-lulesh-safe-refs-poor \
        run-lulesh-safe-refs-persist run-lulesh-safe-refs-persist-poor \
        run-lulesh-safe-refs-txs1 run-lulesh-safe-refs-txs1-poor \
        run-lulesh-safe-refs-txs2 run-lulesh-safe-refs-txs2-poor \
        run-lulesh-safe-refs-txs4 run-lulesh-safe-refs-txs4-poor \
        run-lulesh-vheap run-lulesh-malloc

run-lulesh-nosafe-norefs: lulesh-nosafe-norefs
	rm -f lulesh.nvl
	./$<

run-lulesh-nosafe-norefs-poor: lulesh-nosafe-norefs-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-norefs: lulesh-safe-norefs
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-norefs-poor: lulesh-safe-norefs-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs: lulesh-safe-refs
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-poor: lulesh-safe-refs-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-persist: lulesh-safe-refs-persist
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-persist-poor: lulesh-safe-refs-persist-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs1: lulesh-safe-refs-txs1
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs1-poor: lulesh-safe-refs-txs1-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs2: lulesh-safe-refs-txs2
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs2-poor: lulesh-safe-refs-txs2-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs4: lulesh-safe-refs-txs4
	rm -f lulesh.nvl
	./$<

run-lulesh-safe-refs-txs4-poor: lulesh-safe-refs-txs4-poor
	rm -f lulesh.nvl
	./$<

run-lulesh-vheap: lulesh-vheap
	./$<

run-lulesh-malloc: lulesh-malloc
	./$<

.PHONY: neat clean

neat:
	rm -rf lulesh.nvl cetus_output x.out

clean: neat
	rm -rf lulesh-nosafe-norefs lulesh-nosafe-norefs-poor \
	       lulesh-safe-norefs lulesh-safe-norefs-poor \
	       lulesh-safe-refs lulesh-safe-refs-poor \
	       lulesh-safe-refs-persist lulesh-safe-refs-persist-poor \
	       lulesh-safe-refs-txs1 lulesh-safe-refs-txs1-poor \
	       lulesh-safe-refs-txs2 lulesh-safe-refs-txs2-poor \
	       lulesh-safe-refs-txs4 lulesh-safe-refs-txs4-poor \
	       lulesh-vheap lulesh-malloc

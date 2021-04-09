include ../../../../../make.header
#NVMDIR ?= /opt/fio/scratch/jum/jacobi-mpi-checkpoints
NVMDIR ?= tmp
RESDIR ?= tmp
N ?= 1024
ITERS ?= 10
ITERS_PER_TX ?= 1
RANKS ?= 8
NVLCC = ../../../../../bin/openarc-cc \
        -Warc,-macro=N=$(N) -Warc,-macro=ITERS=$(ITERS) \
        -Warc,-macro=ITERS_PER_TX=$(ITERS_PER_TX) \
        -I$(MPI_INCLUDES) -L$(MPI_LIBDIR)
LIBS = -lpmemobj -lpmem -lmpi -lm
CFLAGS = -DN=$(N) -O3 -DITERS=$(ITERS) \
         -DITERS_PER_TX=$(ITERS_PER_TX)
LD_LIBRARY_PATH:=../../../../../nvl/rt:$(PMEM_LIBDIR):$(MPI_LIBDIR):$(LD_LIBRARY_PATH)

all: jacobi \
     jacobi-txs1 \
     jacobi-txs2 \
     jacobi-txs4 \

jacobi: jacobi.c
	$(NVLCC) $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj

jacobi-txs1: jacobi.c
	$(NVLCC) -DTXS=1 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs-mpi -fnvl-add-txs

jacobi-txs2: jacobi.c
	$(NVLCC) -DTXS=2 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs-mpi -fnvl-add-txs

jacobi-txs4: jacobi.c
	$(NVLCC) -DTXS=4 $(CFLAGS) -o $@ $< $(LIBS) \
	  -lnvlrt-pmemobj-txs-mpi -fnvl-add-txs

.PHONY: run-jacobi \
        run-jacobi-txs1 \
        run-jacobi-txs2 \
        run-jacobi-txs4

run-jacobi: jacobi
	rm -rf $(RESDIR) $(NVMDIR)
	mkdir -p $(RESDIR) $(NVMDIR)
	$(MPI_EXEC) -n $(RANKS) ./$< '$(NVMDIR)' '$(RESDIR)'

run-jacobi-txs1: jacobi-txs1
	rm -rf $(RESDIR) $(NVMDIR)
	mkdir -p $(RESDIR) $(NVMDIR)
	$(MPI_EXEC) -n $(RANKS) ./$< '$(NVMDIR)' '$(RESDIR)'

run-jacobi-txs2: jacobi-txs2
	rm -rf $(RESDIR) $(NVMDIR)
	mkdir -p $(RESDIR) $(NVMDIR)
	$(MPI_EXEC) -n $(RANKS) ./$< '$(NVMDIR)' '$(RESDIR)'

run-jacobi-txs4: jacobi-txs4
	rm -rf $(RESDIR) $(NVMDIR)
	mkdir -p $(RESDIR) $(NVMDIR)
	$(MPI_EXEC) -n $(RANKS) ./$< '$(NVMDIR)' '$(RESDIR)'

.PHONY: neat clean

neat:
	rm -rf jacobi.nvl cetus_output

clean: neat
	rm -rf jacobi \
	       jacobi-txs1 \
	       jacobi-txs2 \
	       jacobi-txs4

include ../../make.header

HAVE_PMEM = $(if $(PMEM_INCLUDES),$(if $(PMEM_LIBDIR),1))
HAVE_MPI = $(if $(MPI_INCLUDES),$(if $(MPI_LIBDIR),1))

all: libnvlrt.a $(if $(HAVE_PMEM), \
                  $(if $(HAVE_MPI), libnvlrt-pmemobj-txs-mpi.so) \
                  libnvlrt-pmemobj-txs.so libnvlrt-pmemobj.so \
                  libnvlrt-pmemobj-norefs.so libnvlrt-pmemobj-persist.so \
                  libnvlrt-vmem.so prof/prof-tx-add)

NVLRT_CFLAGS = -O3

nvlrt-pmemobj-txs-mpi.o: \
  NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC -I$(MPI_INCLUDES)
nvlrt-pmemobj-txs.o: \
  NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC -DNVLRT_NOMPI
nvlrt-pmemobj.o: \
  NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC -DNVLRT_NOMPI -DNVLRT_NOTXS
nvlrt-pmemobj-norefs.o: \
  NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC -DNVLRT_NOMPI -DNVLRT_NOTXS \
                  -DNVLRT_NOREFS
nvlrt-pmemobj-persist.o: \
  NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC -DNVLRT_NOMPI -DNVLRT_NOTXS \
                  -DNVLRT_PERSIST
prof/prof-tx-add.o: NVLRT_CFLAGS += -fnvl-add-txs
nvlrt-vmem.o: NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -I../include -fPIC
vrefs-pmemobj-txs.o: NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC
vrefs-pmemobj.o: NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC \
                                 -DNVLRT_NOTXS
vrefs-pmemobj-persist.o: NVLRT_CFLAGS += -I$(PMEM_INCLUDES) -fPIC \
                                         -DNVLRT_NOTXS -DNVLRT_PERSIST
nvlrt.o: NVLRT_CFLAGS +=
killpos.o: NVLRT_CFLAGS += -fPIC

# TODO: The usual C calling conventions for Linux, as implemented by other C
# compilers, require our 192-bit structs that represent wide NVM pointers to
# be passed to and returned from functions as parameters that are pointers
# to those structs. However, OpenARC+LLVM backend does not properly
# implement these calling conventions, so it just passes these structs
# naively by value. Thus, if we compile our runtime with a standard C
# compiler and we compile applications with OpenARC+LLVM, they will not be
# compatible. According to what I've read, Clang's code for implementing
# these calling conventions is complex, so we've decided not to fix
# OpenARC+LLVM for now. Instead, we choose a quicker solution: compile our
# runtime with OpenARC+LLVM as well.
nvlrt-pmemobj-txs-mpi.o: CC=../../bin/openarc-cc
nvlrt-pmemobj-txs.o: CC=../../bin/openarc-cc
nvlrt-pmemobj.o: CC=../../bin/openarc-cc
nvlrt-pmemobj-norefs.o: CC=../../bin/openarc-cc
nvlrt-pmemobj-persist.o: CC=../../bin/openarc-cc
prof/prof-tx-add.o: CC=../../bin/openarc-cc

nvlrt-pmemobj-txs-mpi.o: common-pmemobj.h vrefs-pmemobj.h killpos.h \
                         ../include/nvlrt-test.h prof/prof-tx-add.h
nvlrt-pmemobj-txs.o: common-pmemobj.h vrefs-pmemobj.h killpos.h \
                     ../include/nvlrt-test.h prof/prof-tx-add.h
nvlrt-pmemobj.o: common-pmemobj.h vrefs-pmemobj.h ../include/nvlrt-test.h
nvlrt-pmemobj-norefs.o: common-pmemobj.h ../include/nvlrt-test.h
nvlrt-pmemobj-persist.o: common-pmemobj.h vrefs-pmemobj.h \
                         ../include/nvlrt-test.h
prof/prof-tx-add.o: ../include/nvl.h ../include/nvlrt-test.h
vrefs-pmemobj-txs.o: vrefs-pmemobj.h common-pmemobj.h killpos.h
vrefs-pmemobj.o: vrefs-pmemobj.h common-pmemobj.h
vrefs-pmemobj-persist.o: vrefs-pmemobj.h common-pmemobj.h
nvlrt-vmem.o: ../include/nvl-vheap.h
killpos.o: killpos.h

PMEM_CUCKOO_LIBS = \
  $(PMEM_LIBDIR)/libpmemobj/cuckoo.o \
  $(PMEM_LIBDIR)/libpmemobj/out.o \
  $(PMEM_LIBDIR)/libpmemobj/util.o \
  $(PMEM_LIBDIR)/libpmemobj/util_linux.o

libnvlrt-pmemobj-txs-mpi.so: \
  vrefs-pmemobj-txs.o killpos.o $(PMEM_CUCKOO_LIBS)
libnvlrt-pmemobj-txs.so: \
  vrefs-pmemobj-txs.o killpos.o $(PMEM_CUCKOO_LIBS)
libnvlrt-pmemobj.so: \
  vrefs-pmemobj.o $(PMEM_CUCKOO_LIBS)
libnvlrt-pmemobj-persist.so: \
  vrefs-pmemobj-persist.o $(PMEM_CUCKOO_LIBS)

lib%.so: %.o
	$(CC) -shared -o $@ $^

lib%.a: %.o
	ar rs $@ $^

%-txs-mpi.o: %.c
	$(CC) $(NVLRT_CFLAGS) -c -o $@ $<

%-txs.o: %.c
	$(CC) $(NVLRT_CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(NVLRT_CFLAGS) -c -o $@ $<

%-norefs.o: %.c
	$(CC) $(NVLRT_CFLAGS) -c -o $@ $<

%-persist.o: %.c
	$(CC) $(NVLRT_CFLAGS) -c -o $@ $<

prof/prof-tx-add: prof/prof-tx-add.o libnvlrt-pmemobj-txs.so
	$(CC) -o $@ $< \
	      -L$(PMEM_LIBDIR) -L. -lnvlrt-pmemobj-txs -lpmemobj -lpmem -lm

.PHONY: neat clean run-prof-tx-add

# This target is intended to be called via the prof/run script.
# See ../README.
run-prof-tx-add: prof/prof-tx-add
	cd prof \
	&& LD_LIBRARY_PATH=..:$(PMEM_LIBDIR) ./prof-tx-add \
	     $(PROFILE_N) $(PROFILE_MODE)

neat:
	rm -rf *.o cetus_output

clean: neat
	rm -f *.a *.dylib *.so

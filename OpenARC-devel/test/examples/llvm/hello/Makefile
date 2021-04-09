# This simple "hello world" example make file demonstrates how to compile a
# C program using OpenARC's LLVM backend.

include ../../../../make.header

OPENARC = ../../../../bin/openarc -emitLLVM -outdir=.
LLVM_LLC = ../../../../llvm/install/bin/llc

all: hello

hello.bc: hello.c hello.h
	$(OPENARC) hello.c

main.bc: main.c hello.h
	$(OPENARC) main.c

hello.o: hello.bc
	$(LLVM_LLC) -filetype=obj hello.bc

main.o: main.bc
	$(LLVM_LLC) -filetype=obj main.bc

hello: hello.o main.o
	$(CC) -o hello hello.o main.o

clean:
	rm -rf hello *.o *.bc

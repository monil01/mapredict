.PHONY: build clean purge compile runtime copyheader install help

COMMON_DEPS = check copyheader make.header make.template

ifndef TARGET_SYSTEM
	TARGET_SYSTEM = CURRENT
endif

ifeq ($(OPENARC_ARCH), 0)
DEFAULT_TARGET_SYSTEM = CUDA 
else ifeq ($(OPENARC_ARCH), 1)
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
DEFAULT_TARGET_SYSTEM = OSX 
else
DEFAULT_TARGET_SYSTEM = OPENCL
endif
else ifeq ($(OPENARC_ARCH), 2)
DEFAULT_TARGET_SYSTEM = OPENCL
else ifeq ($(OPENARC_ARCH), 3)
DEFAULT_TARGET_SYSTEM = INTELFPGA
else ifeq ($(OPENARC_ARCH), 5)
DEFAULT_TARGET_SYSTEM = HIP
else ifeq ($(OPENARC_ARCH), 6)
DEFAULT_TARGET_SYSTEM = IRIS
else
DEFAULT_TARGET_SYSTEM = NONE
endif

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
PREDICTED_TARGET_SYSTEM = OSX 
endif

ifndef PREDICTED_TARGET_SYSTEM
  UNAME := $(shell which nvcc &> /dev/null && echo -n CUDA)
  ifeq ($(UNAME),CUDA)
    PREDICTED_TARGET_SYSTEM = CUDA 
  endif
endif

ifndef PREDICTED_TARGET_SYSTEM
  UNAME := $(shell which hipcc &> /dev/null && echo -n HIP)
  ifeq ($(UNAME),HIP)
    PREDICTED_TARGET_SYSTEM = HIP 
  endif
endif

ifndef PREDICTED_TARGET_SYSTEM
  UNAME := $(shell locate libOpenCL.so &> /dev/null && echo -n OPENCL)
  ifeq ($(UNAME),OPENCL)
    PREDICTED_TARGET_SYSTEM = OPENCL 
  endif
endif

ifndef PREDICTED_TARGET_SYSTEM
  UNAME := $(shell locate libbrisbane.so > /dev/null && echo -n IRIS)
  ifeq ($(UNAME),IRIS)
    PREDICTED_TARGET_SYSTEM = IRIS 
  endif
endif

ifndef PREDICTED_TARGET_SYSTEM
	PREDICTED_TARGET_SYSTEM = OPENMP3
endif

TARGET_MAKE_HEADER = make.header.$(TARGET_SYSTEM)

INSTALL_PREFIX ?= $(shell echo $(PWD))/install

build: compile runtime

check: 
	@if [ ! -d "$(openarc)" ]; then echo "[ERROR] environment variable, openarc should be set to the root directory of the OpenARC repository (e.g., export openarc=$(PWD)); exit!"; exit 1; fi

compile: $(COMMON_DEPS)
	./build.sh bin 

runtime: $(COMMON_DEPS)
	diff make.header ./makefiles/make.header.OPENMP3 > /dev/null || { cd ./openarcrt; ./batchmake.bash; cd ../; }

install: 
	@echo "==> Install OpenARC files in $(INSTALL_PREFIX) directory."
	mkdir -p $(INSTALL_PREFIX)/include
	mkdir -p $(INSTALL_PREFIX)/bin
	mkdir -p $(INSTALL_PREFIX)/lib
	echo '#! /bin/sh' > $(INSTALL_PREFIX)/bin/openarc
	echo 'java -cp $(INSTALL_PREFIX)/lib/antlr.jar:$(INSTALL_PREFIX)/lib/cetus.jar -Xmx1g openacc.exec.ACC2GPUDriver $$*' >> $(INSTALL_PREFIX)/bin/openarc
	chmod 755 $(INSTALL_PREFIX)/bin/openarc
	if [ -f ./bin/binBuilder_cuda ]; then cp ./bin/binBuilder_cuda $(INSTALL_PREFIX)/bin/; fi
	if [ -f ./bin/binBuilder_hip ]; then cp ./bin/binBuilder_hip $(INSTALL_PREFIX)/bin/; fi
	if [ -f ./bin/binBuilder_brisbane ]; then cp ./bin/binBuilder_brisbane $(INSTALL_PREFIX)/bin/; fi
	if [ -f ./bin/binBuilder_opencl ]; then cp ./bin/binBuilder_opencl $(INSTALL_PREFIX)/bin/; fi
	cp -f ./lib/antlr.jar $(INSTALL_PREFIX)/lib/
	cp -f ./lib/cetus.jar $(INSTALL_PREFIX)/lib/
	if [ -f ./openarcrt/libopenaccrt_cuda.a ]; then cp -f ./openarcrt/libopenaccrt_cuda.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_cuda.a ]; then cp -f ./openarcrt/libopenaccrtomp_cuda.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_cudapf.a ]; then cp -f ./openarcrt/libopenaccrt_cudapf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_cudapf.a ]; then cp -f ./openarcrt/libopenaccrtomp_cudapf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_hip.a ]; then cp -f ./openarcrt/libopenaccrt_hip.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_hip.a ]; then cp -f ./openarcrt/libopenaccrtomp_hip.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_hippf.a ]; then cp -f ./openarcrt/libopenaccrt_hippf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_hippf.a ]; then cp -f ./openarcrt/libopenaccrtomp_hippf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_brisbane.a ]; then cp -f ./openarcrt/libopenaccrt_brisbane.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_brisbane.a ]; then cp -f ./openarcrt/libopenaccrtomp_brisbane.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_brisbanepf.a ]; then cp -f ./openarcrt/libopenaccrt_brisbanepf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_brisbanepf.a ]; then cp -f ./openarcrt/libopenaccrtomp_brisbanepf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_opencl.a ]; then cp -f ./openarcrt/libopenaccrt_opencl.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_opencl.a ]; then cp -f ./openarcrt/libopenaccrtomp_opencl.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrt_openclpf.a ]; then cp -f ./openarcrt/libopenaccrt_openclpf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libopenaccrtomp_openclpf.a ]; then cp -f ./openarcrt/libopenaccrtomp_openclpf.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libomphelper.a ]; then cp -f ./openarcrt/libomphelper.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libmclext.a ]; then cp -f ./openarcrt/libmclext.a $(INSTALL_PREFIX)/lib/; fi
	if [ -f ./openarcrt/libresilience.a ]; then cp -f ./openarcrt/libresilience.a $(INSTALL_PREFIX)/lib/; fi
	cp -f ./openarcrt/openacc.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/openaccrt.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/resilience.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/omp4_device.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/omp_helper.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/mcl_accext.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/profile.h $(INSTALL_PREFIX)/include/
	cp -f ./openarcrt/Makefile.buildIRISKernels $(INSTALL_PREFIX)/
	cp -f ./make.header $(INSTALL_PREFIX)/
	cp -f ./make.template $(INSTALL_PREFIX)/
	@echo "==> Set the following environment variables for proper make configuration."
	@echo "export OPENARCLIB=$(INSTALL_PREFIX)/lib"
	@echo "export OPENARCINCLUDE=$(INSTALL_PREFIX)/include"

uninstall:
	@echo "==> Uninnstall OpenARC files"
	rm -f $(INSTALL_PREFIX)/bin/openarc
	rm -f $(INSTALL_PREFIX)/bin/binBuilder_*
	rm -f $(INSTALL_PREFIX)/lib/libopenaccrt*
	rm -f $(INSTALL_PREFIX)/lib/libomphelper.a
	rm -f $(INSTALL_PREFIX)/lib/libmclext.a
	rm -f $(INSTALL_PREFIX)/lib/libresilience.a
	rm -f $(INSTALL_PREFIX)/lib/antlr.jar
	rm -f $(INSTALL_PREFIX)/lib/cetus.jar
	rm -f $(INSTALL_PREFIX)/include/openacc.h
	rm -f $(INSTALL_PREFIX)/include/openaccrt.h
	rm -f $(INSTALL_PREFIX)/include/resilience.h
	rm -f $(INSTALL_PREFIX)/include/omp4_device.h
	rm -f $(INSTALL_PREFIX)/include/omp_helper.h
	rm -f $(INSTALL_PREFIX)/include/mcl_accext.h
	rm -f $(INSTALL_PREFIX)/include/profile.h
	rm -f $(INSTALL_PREFIX)/make.header
	rm -f $(INSTALL_PREFIX)/make.template
	rm -f $(INSTALL_PREFIX)/Makefile.buildIRISKernels

copyheader: 
	if [ "$(TARGET_SYSTEM)" != "CURRENT" ] && [ -f ./makefiles/$(TARGET_MAKE_HEADER) ]; then cp ./makefiles/$(TARGET_MAKE_HEADER) ./make.header; fi
	if [ ! -f "./make.header" ] && [ -f ./makefiles/make.header.$(DEFAULT_TARGET_SYSTEM) ]; then  cp ./makefiles/make.header.$(DEFAULT_TARGET_SYSTEM) ./make.header; fi
	if [ ! -f "./make.header" ] && [ -f ./makefiles/make.header.$(PREDICTED_TARGET_SYSTEM) ]; then  cp ./makefiles/make.header.$(PREDICTED_TARGET_SYSTEM) ./make.header; fi
	if [ ! -f "./make.header" ]; then echo "[ERROR] no make.header file is found!"; fi

help:
	@echo "==> Available make options: build, compile, runtime, install, copyheader, clean, purge, help, TARGET_SYSTEM=[target-system], INSTALL_PREFIX=[install-directory]"
	@echo "    - Available options for [target-system]: CURRENT, CUDA, OSX, OPENCL, HIP, IRIS, INTELFPGA, ..."
	@echo "       - Any [suffix] in makefiles/make.header.[suffix] can be used as [target-system]" 

clean: copyheader
	./build.sh clean
	cd ./openarcrt; make purge; cd ../

purge: clean uninstall
	rm -f ./make.header
	rm -rf ./install

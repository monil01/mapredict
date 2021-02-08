# Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
TOP=.
#TOP=/home/users/mmonil/ORNL_seyong/mapmc
ASPEN=${TOP}/aspen
SRC=${TOP}/model_parser
#TOPDIR=/home/users/mmonil/ORNL_seyong/mapmc/aspen
#TOPDIR=/home/users/mmonil/ORNL_seyong/aspen
DEPMODE=gcc3
CC=gcc
CXX=g++
YACC=bison -y
LEX=flex
JRE=java
JAVAH=
JAVAC=
SUBDIRS=$(SUBDIRS_COMMON)

MPI=no
MPI_CPPFLAGS=
MPI_LDFLAGS=
MPI_LIBS=

BOOST=yes
BOOST_CPPFLAGS=
BOOST_LDFLAGS=
BOOST_LIBS=

PYTHON_CPPFLAGS=-I/packages/python/3.6.8/include/python3.6m
PYTHON_LDFLAGS=-L/packages/python/3.6.8/lib -lpython3.6m
PYTHON_EXTRA_LIBS=-lpthread -ldl  -lutil -lm
PYTHON_EXTRA_LDFLAGS=-Xlinker -export-dynamic

NLOPT=no
NLOPT_CPPFLAGS=
NLOPT_LDFLAGS=
NLOPT_LIBS=

JAVA=no
JAVA_CPPFLAGS=

CFLAGS=-fPIC -g
CXXFLAGS= -DHAVE_CXXABI_H -fPIC -g -std=c++11
CPPFLAGS=
LDFLAGS= -rdynamic 
AR=ar
RM=rm -rf

CPPFLAGS+=-I$(ASPEN)/aspen
LIBS=-L$(ASPEN)/lib -laspen
LIBDEP=$(ASPEN)/lib/libaspen.a

TESTS=mapmc

#TESTS=stream_access
#TESTS=mapmc \
#  stream_access


OBJ=./model_parser/main.o ./model_parser/traverser.o ./model_parser/analytical_model.o

#OBJ=./model_parser/$(TESTS:=.o)

all: $(TESTS)

# programs

stream_access1: $(LIBDEP) stream_access.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(@:=.o) -o $@ $(LIBS)

stream_access: $(LIBDEP) ${SRC}/stream_access.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(@:=.o) -o $@ $(LIBS)


#mapmc: $(LIBDEP) ./model_parser/main.o ./model_parser/traverser.o ./model_parser/analytical_model.o
mapmc: $(LIBDEP) $(OBJ)
	#$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) ./model_parser/$(@:=.o) -o $@ $(LIBS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(OBJ) -o $@ $(LIBS)

run:
	./run_all.sh


# In case someone messes up and inserts this file at the top of their
# Makefile.in instead of at the bottom, we will make a default target
# and assume they used the "all" convention as the true default target
#
backupdefaulttarget: all

#
# Suffixes
#
.SUFFIXES: .c .C .cpp .cu .y .l .d .java .class

.PHONY: $(SUBDIRS) clean distclean depclean

#
# Compilation targets
#
.c.o:
	source='$<' object='$@' libtool=no depfile='./$*.d'  \
	depmode=$(DEPMODE) $(ASPEN)/config/depcomp   \
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

#source='./model_parser/$<' object='./model_parser/$@' libtool=no depfile='./model_parser/$*.d'  \
#depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \

.C.o:
	source='$<' object='$@' libtool=no depfile='./$*.d'  \
	depmode=$(DEPMODE) $(ASPEN)/config/depcomp   \
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

#depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \
#source='./model_parser/$<' object='./model_parser/$@' libtool=no depfile='./model_parser/$*.d'  \

.cpp.o:
	source='$<' object='$@' libtool=no depfile='./$*.d'  \
	depmode=$(DEPMODE) $(ASPEN)/config/depcomp   \
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

#depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \
#source='./model_parser/$<' object='./model_parser/$@' libtool=no depfile='./model_parser/$*.d'  \
#
# Dependency targets
#
DEP=$(OBJ:.o=.d)
-include $(DEP)

#
# Main targets
#
$(SUBDIRS) or_no_subdirs:
	(cd $@ && $(MAKE) all)

$(ARCHIVE) or_no_archive: $(OBJ) $(LIBDEP) $(ASPEN)/config/make-variables
#$(ARCHIVE) or_no_archive: $(OBJ) $(LIBDEP) $(TOPDIR)/config/make-variables
	$(AR) -rc $@ $(OBJ) 

$(LIBRARY) or_no_library: $(OBJ) $(LIBDEP) $(ASPEN)/config/make-variables
#$(LIBRARY) or_no_library: $(OBJ) $(LIBDEP) $(TOPDIR)/config/make-variables
	$(CXX) -shared $(OBJ) -o $@ $(LDFLAGS)

$(PROGRAM) or_no_program: $(OBJ) $(LIBDEP) $(LIBRARY) $(ARCHIVE) $(ASPEN)/config/make-variables
#$(PROGRAM) or_no_program: $(OBJ) $(LIBDEP) $(LIBRARY) $(ARCHIVE) $(TOPDIR)/config/make-variables
	$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(LIBS)

#
# Clean targets
#
clean:
	@if test -n "$(SUBDIRS)"; then \
	    for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) $@); done \
	fi
	echo $(OBJ) $(PROGRAM) $(LIBRARY) $(ARCHIVE) $(TESTS) $(CLEAN_FILES)
	$(RM) $(OBJ) $(PROGRAM) $(LIBRARY) $(ARCHIVE) $(TESTS) $(CLEAN_FILES)

distclean:
	@if test -n "$(SUBDIRS)"; then \
	    for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) $@); done \
	fi
	$(RM) $(OBJ) $(PROGRAM) $(LIBRARY) $(ARCHIVE) $(TESTS) $(CLEAN_FILES)
	$(RM) $(DEP)
	$(RM) -r Makefile $(DISTCLEAN_FILES)
	$(RM) *~ */*~

depclean:
	@if test -n "$(SUBDIRS)"; then \
	    for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) $@); done \
	fi
	$(RM) $(DEP)

# Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
TOPDIR=/home/users/mmonil/temp/mapmc/aspen
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

PYTHON_CPPFLAGS=-I/usr/include/python2.7
PYTHON_LDFLAGS=-L/usr/lib64 -lpython2.7
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

CPPFLAGS+=-I../aspen $(JAVA_CPPFLAGS)
LDFLAGS+=-L../lib -laspen

LIBRARY=lib/libaspenjni.so

JSRC=\
 aspen/AppModel.java \
 aspen/Expression.java \
 aspen/MachModel.java \
 aspen/MachComponent.java \
 aspen/Aspen.java
OBJ=aspenjni.o

HDR=$(JSRC:.java=.h)
CLASS=$(JSRC:.java=.class)

all: $(CLASS) $(LIBRARY) 

$(OBJ): $(HDR)

$(LIBRARY):

.class.h:
	$(JAVAH) -o $@ aspen.`basename $< | cut -d. -f1`

.java.class:
	$(JAVAC) $<

test:
	$(JAVAC) Test.java
	$(JRE) -Djava.library.path=`pwd`/lib Test

CLEAN_FILES=$(HDR) $(CLASS) Test.class

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
	depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

.C.o:
	source='$<' object='$@' libtool=no depfile='./$*.d'  \
	depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

.cpp.o:
	source='$<' object='$@' libtool=no depfile='./$*.d'  \
	depmode=$(DEPMODE) $(TOPDIR)/config/depcomp   \
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

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

$(ARCHIVE) or_no_archive: $(OBJ) $(LIBDEP) $(TOPDIR)/config/make-variables
	$(AR) -rc $@ $(OBJ) 

$(LIBRARY) or_no_library: $(OBJ) $(LIBDEP) $(TOPDIR)/config/make-variables
	$(CXX) -shared $(OBJ) -o $@ $(LDFLAGS)

$(PROGRAM) or_no_program: $(OBJ) $(LIBDEP) $(LIBRARY) $(ARCHIVE) $(TOPDIR)/config/make-variables
	$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(LIBS)

#
# Clean targets
#
clean:
	@if test -n "$(SUBDIRS)"; then \
	    for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) $@); done \
	fi
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

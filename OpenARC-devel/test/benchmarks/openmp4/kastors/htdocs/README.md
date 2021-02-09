#KASTORS benchmarks

This page is still a WIP

##Overview

KASTORS is composed of six small kernels :

* Poisson2D (aka Jacobi)
* SparseLU
* Strassen
* LU matrix factorization (dgetrf)
* QR matrix factorization (dgeqrf)
* Cholesky factorization (dpotrf)


###Poisson2D

This algorithm solves the Poisson equation on the unit square
[0,1]x[0,1], which is divided into a grid of NxN evenly-spaced
points. This benchmark relies on a 5-point 2D stencil computational
kernel that is repeatedly applied until convergence is detected. We
implemented two main blocked versions of this kernel, using either
independent tasks and tasks with dependencies.

###SparseLU

This benchmark computes the LU decomposition of a sparse matrix. We
modified the original BOTS implementation to express task dependencies,
except only non-NULL blocks are updated to adapt the traditional LU decomposition to sparse
matrices.

###Strassen

The Strassen algorithm uses matrix decompositions to compute the
multiplication of large dense matrices. Similarly to SparseLU, we
modified the BOTS implementation to add parallelism for addition part of the
algorithm and express task dependencies instead
of using taskwait-based synchronizations.


###Plasma
The [Plasma](http://icl.cs.utk.edu/plasma/) library developed at ICL/UTK provides a
large number of key linear algebra algorithms optimized for multi-core
architectures. Several implementations of each algorithm are
available, either using static or dynamic scheduling. Dynamic
scheduled algorithms are built on top of the QUARK
runtime system, which uses a data-flow dependency model to schedule
tasks. The three algorithms we selected are a Cholesky decomposition,
a QR decomposition and LU decomposition, respectively known as DPOTRF, DGEQRF
and DGETRF in PLASMA, which all operate on double precision floating point
matrices.


##Dependencies

You will need an OpenMP 4.0 compiler, like GCC 4.9 or the Intel fork of Clang 3.4 [here](http://clang-omp.github.io/).

###Plasma
You will need additional dependencies for plasma's kernel : they rely on blas and lapacke library, so you will need them too.


##Usage

Just type "make check" to check compiler behavior and "make run" to run the same experiement than in the paper with the configured compiler
You can edit compilation flags in config/config or create your own config/config.user file, and edit the variables you want.

All programs come with their help ("-h" flag), and executing a kernel without arguments will run
it with the default parameters.



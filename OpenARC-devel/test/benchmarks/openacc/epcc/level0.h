/* Copyright (c) 2013 The University of Edinburgh. */

/* Licensed under the Apache License, Version 2.0 (the "License"); */
/* you may not use this file except in compliance with the License. */
/* You may obtain a copy of the License at */

/*     http://www.apache.org/licenses/LICENSE-2.0 */

/* Unless required by applicable law or agreed to in writing, software */
/* distributed under the License is distributed on an "AS IS" BASIS, */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/* See the License for the specific language governing permissions and */
/* limitations under the License. */

#ifdef _OPENARC_
#pragma openarc #define _PPN_ (_DATASIZE_/256)
#endif

//Disable parallel_private() and parallel_firstprivate() tests by default
//since they can run only with very small datasize (_DATASIZE_).
#ifndef ENABLE_PRIVATE_TESTS
#define ENABLE_PRIVATE_TESTS 0
#endif

double contig_htod();
double contig_dtoh();
double sliced_dtoh();
double sliced_htod();
double kernels_if();
double parallel_if();
double parallel_private();
double parallel_firstprivate();
double kernels_combined();
double parallel_combined();
double update();
double kernels_invoc();
double parallel_invoc();
double parallel_reduction();
double kernels_reduction();


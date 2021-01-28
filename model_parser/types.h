#ifndef TYPES_H
#define TYPES_H


extern bool DEBUG_MAPMC;

typedef enum access_patterns{
STREAM,
STRIDE,
STENCIL,
RANDOM,
MIXED}
access_patterns;

typedef enum instructions{
LOAD,
STORE,
FLOPs}
instructions;

typedef enum compilers{
GCC,
INTEL}
compilers;

#define KB 1024
#define MB KB*1024
#define GB MB*1024

#endif

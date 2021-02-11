#ifndef MYUTILS_H
#define MYUTILS_H

#define SAFE_SIZE 1024*1024*1

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <set>

using namespace std;

typedef unsigned long long TIME_MSEC;

void print2DData(FILE* output, double* data, long width, long height);
void print2DData(FILE* output, float* data, long width, long height);

TIME_MSEC get_current_msec();

char* getArgumentValue(int argc, char **argv, char *argName);

int msleep(unsigned long milisec);

long safeSize(long size);

void parseParams(int argc, char *argv[]);

// These names may vary by implementation
#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine
//#define LINEAR_CONGRUENTIAL_ENGINE linear_congruential_engine
#define UNIFORM_INT_DISTRIBUTION uniform_int_distribution
//#define UNIFORM_INT_DISTRIBUTION uniform_int


typedef unsigned int uint;
typedef unsigned long ulong;

typedef set<int> node;


#endif

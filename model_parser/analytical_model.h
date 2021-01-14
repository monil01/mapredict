// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/RuntimeCounter.h"
#include "walkers/RuntimeExpression.h"

#include "model/ASTDataStatement.h"

#include "app/ASTExecutionBlock.h"
#include "model/ASTMachModel.h"
#include "app/ASTRequiresStatement.h"
#include "walkers/AspenTool.h"

#include "types.h"

#define Finegrained_RSC_Print

//using namespace std;
//


class analytical_model{


private:
    int _access_pattern;
    int _compiler;
    int _instruction_type;
    size_t _cache_line_size;
    vector<ASTTrait*> _traits;
    bool _initialized;
    bool _prefetch_enabled; 
    bool _multithreaded; 
    int64_t _data_structure_size;
    int _element_size;
    


public:

analytical_model(int access_pattern,
    int compiler,
    int instruction_type,
    size_t cache_line_size,
    vector<ASTTrait*> traits;
    bool initialized,
    bool prefetch_enabled,
    bool multithreaded,
    int64_t data_structure_size,
    int element_size
);
~analytical_model();

double  predictMemoryAccess(int pattern);
double  streamAccess(int pattern);
double  randomAccess(int pattern);
double  strideAccess(int pattern);
double  stencilAccess(int pattern);

};



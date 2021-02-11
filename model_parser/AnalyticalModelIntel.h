// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.

#ifndef ANAlYTICAL_MODEL_INTEL_H
#define ANAlYTICAL_MODEL_INTEL_H


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

#include "Types.h"
#include "AnalyticalModelUtility.h"

#define Finegrained_RSC_Print

//using namespace std;
//


class AnalyticalModelIntel{


private:
    int _access_pattern;
    int _compiler;
    int _instruction_type;
    int _cache_line_size;
    vector<ASTTrait*> _traits;
    bool _initialized;
    bool _prefetch_enabled; 
    bool _multithreaded; 
    int64_t _data_structure_size;
    int _element_size;
    int _page_size;
    int _microarchitecture;
    std::string _pattern_string;
    AnalyticalModelUtility* _analytical_model_utility_obj;
    


public:

AnalyticalModelIntel(
    int compiler,
    int instruction_type,
    int cache_line_size,
    vector<ASTTrait*> traits,
    bool prefetch_enabled,
    bool multithreaded,
    std::int64_t data_structure_size,
    int element_size,
    int microarchitecture
);
~AnalyticalModelIntel();

std::int64_t  predictMemoryAccess();
std::int64_t  streamAccess();
std::int64_t  randomAccess();
std::int64_t  strideAccess();
std::int64_t  stencilAccess();
};

#endif

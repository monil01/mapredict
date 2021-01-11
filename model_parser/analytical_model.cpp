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
#include "analytical_model.h"

#define Finegrained_RSC_Print

//using namespace std;
//


analytical_model::analytical_model(int _access_pattern,
    int _compiler,
    int _instruction_type,
    std::size_t _cache_line_size,
    std::vector<std::string> _traits,
    bool _initialized,
    bool _prefetch_enabled,
    bool _multithreaded,
    std::int64_t _data_structure_size
):( _access_pattern = access_patern,
    _compiler = compiler,
    _instruction_type = instruction_type,
    _cache_line_size = cache_line_size,
    _traits = traits,
    _initialized = initialized,
    _prefetch_enabled = prefetch_enabled,
    _multithreaded = multithreaded,
    _data_structure_size = data_structure_size
){ 

}

~analytical_model::analytical_model(){
}

double  analytical_model::predictMemoryAccess(int pattern){

}
double  analytical_model::streamAccess(int pattern) {

}
double  analytical_model::randomAccess(int pattern){

}
double  analytical_model::strideAccess(int pattern){

}
double  analytical_model::stencilAccess(int pattern){

}




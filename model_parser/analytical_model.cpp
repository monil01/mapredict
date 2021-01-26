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

/*
analytical_model(
    int compiler,
    int instruction_type,
    size_t cache_line_size,
    vector<ASTTrait*> traits,
    bool prefetch_enabled,
    bool multithreaded,
    int64_t data_structure_size,
    int element_size
);

*/


analytical_model::analytical_model(
    int compiler,
    int instruction_type,
    int cache_line_size,
    vector<ASTTrait*> traits,
    bool prefetch_enabled,
    bool multithreaded,
    std::int64_t data_structure_size,
    int element_size
)   
     
{ 
    _compiler = compiler;
    _instruction_type = instruction_type;
    _cache_line_size = cache_line_size;
    _traits = traits;
    _prefetch_enabled = prefetch_enabled;
    _multithreaded = multithreaded;
    _data_structure_size = data_structure_size;
    _element_size = element_size;
    if(DEBUG_MAPMC == true) std::cout << " Analytical model trait size " <<  traits.size() << std::endl;

    _access_pattern = findPattern(_traits);
    _initialized = findInitialized(_traits);
    if(DEBUG_MAPMC == true) std::cout << " Analytical model access pattern " <<  _access_pattern << " init " << _initialized << std::endl;


    if ( _element_size * _data_structure_size > 4 * KB ) _page_size = 2 * MB;

    //predictMemoryAccess();
        


}

analytical_model::~analytical_model(){
}

bool analytical_model::findInitialized(vector<ASTTrait*> traits){
    bool init = true;
    if (traits.size() < 1) return init;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "initialized"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            if(DEBUG_MAPMC == true) std::cout << " traits value " << 
                traits[k]->GetValue()->Evaluate() << std::endl;
            init = (bool) traits[k]->GetValue()->Evaluate();
            //stride = stride * element_size;
        }
    }
    return init;
}

int analytical_model::findPattern(vector<ASTTrait*> traits){
    int pattern = access_patterns::STREAM;
    if (traits.size() < 1) return pattern;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "pattern"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            if(DEBUG_MAPMC == true) std::cout << " traits value " << 
                traits[k]->GetValue()->GetText() << std::endl;
            std::string temp_pattern = traits[k]->GetValue()->GetText();
            //std::string temp_pattern = traits[k]->GetValue()->Evaluate();
            if ( temp_pattern.compare("stream")) pattern = access_patterns::STREAM;
            if ( temp_pattern.compare("stride")) pattern = access_patterns::STRIDE;
            if ( temp_pattern.compare("stencil")) pattern = access_patterns::STENCIL;
            if ( temp_pattern.compare("random")) pattern = access_patterns::RANDOM;
            //stride = stride * element_size;
        }
    }
    return pattern;
}



std::int64_t analytical_model::predictMemoryAccess(){

    std::int64_t memory_access = 0;
    if ( _access_pattern == access_patterns::STREAM) memory_access = streamAccess();
    if ( _access_pattern == access_patterns::STRIDE) memory_access = strideAccess();
    if ( _access_pattern == access_patterns::STENCIL) memory_access = stencilAccess();
    if ( _access_pattern == access_patterns::RANDOM)  memory_access = randomAccess();

    return memory_access;
}

std::int64_t  analytical_model::streamAccess() {
    std::int64_t memory_access = 0;
    /// renaming the variables as per model described in the paper
    std::int64_t N = _data_structure_size;
    int CL = _cache_line_size;
    int ES = _element_size;
    // converting the stride and data size to bytes
    
    if (_compiler == compilers::GCC) {
        if (_initialized == true) {
            if ( _instruction_type == instructions::LOAD) {
                //memory_access =  N * ES / (double) CL/10000 ;
                //memory_access = ceil( N * ES / (double) CL/10000 );
                memory_access = ceil( N * ES / (double) CL ) * CL;
                if(DEBUG_MAPMC == true) std::cout << " Memory access " << memory_access << std::endl;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = 2 * ceil( N * ES / (double) CL ) * CL;
            }
        } else {
            if ( _instruction_type == instructions::LOAD) {
                memory_access = ceil( N * ES / (double) CL ) * CL;
            } else if ( _instruction_type == instructions::STORE) {
                memory_access = ceil( N * ES / (double) _page_size ) * _page_size;
            }
        }
    } 
    
    memory_access = ceil( memory_access / (double) CL );

    if(DEBUG_MAPMC == true) std::cout << " Analytical Model " << memory_access << " data size "  
        <<  N  << " element size " << ES << " cacheline " 
        << CL << " page size " << _page_size  << "  instruction " << _instruction_type << std::endl;

    return memory_access;

}

std::int64_t analytical_model::randomAccess(){

}
std::int64_t analytical_model::strideAccess(){

}
std::int64_t analytical_model::stencilAccess(){

}




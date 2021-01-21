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
    std::size_t cache_line_size,
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
    std::cout << " ana model trait size " <<  traits.size() << std::endl;

    _access_pattern = findPattern(_traits);
    _initialized = findInitialized(_traits);
    std::cout << " access pattern " <<  _access_pattern << " init " << _initialized << std::endl;

}

analytical_model::~analytical_model(){
}

bool analytical_model::findInitialized(vector<ASTTrait*> traits){
    bool init = true;
    std::cout << " size " <<  traits.size() << std::endl;
    if (traits.size() < 1) return init;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "initialized"){
            //std::cout << "traits Name " << ttrait;
            std::cout << " traits value " << 
                traits[k]->GetValue()->Evaluate() << std::endl;
            init = (bool) traits[k]->GetValue()->Evaluate();
            //stride = stride * element_size;
        }
    }
    return init;
}

int analytical_model::findPattern(vector<ASTTrait*> traits){
    int pattern = access_patterns::STREAM;
    std::cout << " size " <<  traits.size() << std::endl;
    if (traits.size() < 1) return pattern;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "pattern"){
            //std::cout << "traits Name " << ttrait;
            std::cout << " traits value " << 
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

int findPattern(vector<ASTTrait*> traits);


double  analytical_model::predictMemoryAccess(){

}
double  analytical_model::streamAccess(int pattern) {

}
double  analytical_model::randomAccess(int pattern){

}
double  analytical_model::strideAccess(int pattern){

}
double  analytical_model::stencilAccess(int pattern){

}




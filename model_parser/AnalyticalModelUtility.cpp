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

#include "Types.h"
#include "AnalyticalModelUtility.h"

#define Finegrained_RSC_Print



AnalyticalModelUtility::AnalyticalModelUtility()   
     
{ 

}

AnalyticalModelUtility::~AnalyticalModelUtility(){
}

bool AnalyticalModelUtility::findInitialized(vector<ASTTrait*> traits){
    bool init = true;
    if (traits.size() < 1) return init;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "initialized"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            if(DEBUG_MAPMC == true) std::cout << " initialize traits value " << 
                traits[k]->GetValue()->Evaluate() << std::endl;
            init = (bool) traits[k]->GetValue()->Evaluate();
            //stride = stride * element_size;
        }
    }
    return init;
}

std::string AnalyticalModelUtility::findAlgorithm(vector<ASTTrait*> traits){
    std::string alg = "log";
    if (traits.size() < 1) return alg;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "algorithm"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            if(DEBUG_MAPMC == true) std::cout << " algorithm traits value " << 
                traits[k]->GetValue()->GetText() << std::endl;
            alg = traits[k]->GetValue()->GetText();
            //stride = stride * element_size;
        }
    }
    return alg;
}


int AnalyticalModelUtility::findPattern(vector<ASTTrait*> traits){
    int pattern = access_patterns::STREAM;
    if (traits.size() < 1) return pattern;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "pattern"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            std::string temp_pattern = traits[k]->GetValue()->GetText();
            if(DEBUG_MAPMC == true) std::cout << " Pattern trait value " << temp_pattern << std::endl;
            //std::string temp_pattern = traits[k]->GetValue()->Evaluate();
            if ( temp_pattern.find("stream") != std::string::npos) pattern = access_patterns::STREAM;
            if ( temp_pattern.find("stride") != std::string::npos) pattern = access_patterns::STRIDE;
            if ( temp_pattern.find("stencil") != std::string::npos) {
                    pattern = access_patterns::STENCIL;
                    //_pattern_string = temp_pattern;
                    if(DEBUG_MAPMC == true) std::cout << " Pattern found " <<  pattern << access_patterns::STENCIL << std::endl;

            }
            if ( temp_pattern.find("random") != std::string::npos) pattern = access_patterns::RANDOM;
            //stride = stride * element_size;
        }
    }
    return pattern;
}

int AnalyticalModelUtility::findStride(vector<ASTTrait*> traits){
    int stride = 1;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "stride"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            stride = traits[k]->GetValue()->Evaluate();
            if(DEBUG_MAPMC == true) std::cout << " Stride trait value " << stride << std::endl;
        }
    }
    if(stride == 0)  { 
        std::cout << " ERROR Stride not found and set the stride to 1 " << std::endl;
        stride = 1;
    }
    return stride;
}

double AnalyticalModelUtility::findFactor(vector<ASTTrait*> traits){
    double factor = 0;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == "factor"){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            factor = (double) traits[k]->GetValue()->Evaluate();
            if(DEBUG_MAPMC == true) std::cout << " Factor trait value " << factor << std::endl;
        }
    }
    if(factor == 0)  { 
        std::cout << " ERROR Factor not found and set the factor to 1 " << std::endl;
        factor = 1;
    }
    return factor;
}

double AnalyticalModelUtility::findCorrection(vector<ASTTrait*> traits, std::string correction_str){
    double correction = 0;
    for (int k = 0; k < traits.size(); k++){
        std::string ttrait = traits[k]->GetName();
        if (ttrait == correction_str){
            //if(DEBUG_MAPMC == true) std::cout << "traits Name " << ttrait;
            correction = (double) traits[k]->GetValue()->Evaluate();
            if(DEBUG_MAPMC == true) std::cout << " Correction trait name " << correction_str << " correction trait value " << correction << std::endl;
        }
    }
    if(correction == 0)  {
        std::cout << " ERROR Factor not found and set the correction to 1 " << std::endl;
        correction = 1;
    }
    return correction;
}





std::string AnalyticalModelUtility::generateCorrectionString(int microarchitecture_value, 
    int prefetch_enabled)
{
    double correction = 1;
    std::string correction_str = "correction_";
    switch(microarchitecture_value){
        case microarchitecture::BW:
            correction_str = correction_str + "BW";
            break;
        case microarchitecture::SK:
            correction_str = correction_str + "SK";
            break;
        case microarchitecture::CS:
            correction_str = correction_str + "CS";
            break;
        case microarchitecture::CP:
            correction_str = correction_str + "CP";
            break;
        default:
            correction_str="";
    }

    if(DEBUG_MAPMC == true) std::cout << " Correction String " << correction_str << std::endl;
     
    if (prefetch_enabled == true) correction_str = correction_str + "_prefetch";
    else correction_str = correction_str + "_noprefetch";
    if(DEBUG_MAPMC == true) std::cout << " Correction STR " << correction_str << " micro architecture " << microarchitecture_value << std::endl;
    return correction_str;
}

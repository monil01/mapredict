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

#define Finegrained_RSC_Print

//using namespace std;
//


class AnalyticalModelUtility{


private:


public:

AnalyticalModelUtility();
~AnalyticalModelUtility();

bool findInitialized(vector<ASTTrait*> traits);
int findPattern(vector<ASTTrait*> traits);
int findStride(vector<ASTTrait*> traits);
double findFactor(vector<ASTTrait*> traits);
double findCorrection(vector<ASTTrait*> traits, std::string correction_str);

std::string findAlgorithm(vector<ASTTrait*> traits);
std::string generateCorrectionString();
std::string generateCorrectionString(int microarchitecture_value, 
    int prefetch_enabled);
std::string findPatternString(vector<ASTTrait*> traits);

};



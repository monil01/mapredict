// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.

#ifndef ASPEN_UTILITY_H
#define ASPEN_UTILITY_H

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


class AspenUtility: public AspenTool{


private:


    //ASTAppModel *app = NULL;
    //ASTMachModel *mach = NULL;
//TODo
// All the private varialbe required to call the analytical model function


public:


AspenUtility(ASTAppModel *app, ASTMachModel *mach);
~AspenUtility();

std::int64_t  getApplicationParam(ASTAppModel *app, std::string param);
std::string  getNameOfDataType(std::string str_expression);

const ASTMachComponent* getSocketComponent(ASTMachModel *mach, std::string socket);

std::int64_t getAnyMachineProperty(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property);

int getMicroarchitecture(std::string socket);


ASTAppModel*  getAppModel() { return app;}
ASTMachModel*  getMachineModel() {return mach;}

};

#endif



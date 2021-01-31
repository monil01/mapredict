// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.

#ifndef TRAVERSER_H
#define TRAVERSER_H

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


class traverser: public AspenTool{


private:


    //ASTAppModel *app = NULL;
    //ASTMachModel *mach = NULL;
//TODo
// All the private varialbe required to call the analytical model function


public:


traverser(ASTAppModel *app, ASTMachModel *mach);
~traverser();

std::int64_t  predictMemoryAccess(ASTAppModel *app, 
                                        ASTMachModel *mach, std::string socket);

std::int64_t  predictMemoryStatement(const ASTRequiresStatement *req, std::string socket, 
    std::int64_t inner_parallelism);

std::int64_t  getApplicationParam(ASTAppModel *app, std::string param);
std::string  getNameOfDataType(std::string str_expression);

const ASTMachComponent* getSocketComponent(ASTMachModel *mach, std::string socket);

std::int64_t getAnyMachineProperty(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property);

//std::int64_t analyticalStreamingAccess(std::int64_t D, std::int64_t E, std::int64_t S, std::int64_t CL);
std::int64_t callAnalyticalModel(std::int64_t D, std::int64_t E, std::int64_t S, std::int64_t CL);

std::int64_t
recursiveBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    std::int64_t outer_parallelism,
    const ASTControlStatement *s);

int getMicroarchitecture(std::string socket);


std::int64_t executeBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket,
    const ASTExecutionBlock *exec,
    std::int64_t outer_parallelism);

ASTAppModel*  getAppModel() { return app;}
ASTMachModel*  getMachineModel() {return mach;}

};

#endif



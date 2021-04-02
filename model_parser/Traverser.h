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
#include "Types.h"
#include "AspenUtility.h"



#define Finegrained_RSC_Print

//using namespace std;
//


class Traverser: public AspenTool{


private:


    //ASTAppModel *app = NULL;
    //ASTMachModel *mach = NULL;
//TODo
// All the private varialbe required to call the analytical model function
    AspenUtility *_aspen_utility = NULL;
    std::int64_t _total_loads;
    std::int64_t _total_stores;


public:


Traverser(ASTAppModel *app, ASTMachModel *mach);
~Traverser();

std::int64_t  predictMemoryAccess(ASTAppModel *app, 
                                        ASTMachModel *mach, std::string socket);

std::int64_t  predictMemoryStatement(const ASTRequiresStatement *req, std::string socket, 
    std::int64_t inner_parallelism);




//std::int64_t analyticalStreamingAccess(std::int64_t D, std::int64_t E, std::int64_t S, std::int64_t CL);
std::int64_t callAnalyticalModel(std::int64_t D, std::int64_t E, std::int64_t S, std::int64_t CL);

std::int64_t
recursiveBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    std::int64_t outer_parallelism,
    const ASTControlStatement *s);


std::int64_t executeBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket,
    const ASTExecutionBlock *exec,
    std::int64_t outer_parallelism);

ASTAppModel*  getAppModel() { return app;}
ASTMachModel*  getMachineModel() {return mach;}

std::int64_t getTotalLoads() { return _total_loads;}
std::int64_t getTotalStores() { return _total_stores;}
double getBlockReuseFactor(std::string execute_block_name, std::string socket);

};

#endif



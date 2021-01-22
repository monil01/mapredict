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

double  predictMemoryAccess(ASTAppModel *app, 
                                        ASTMachModel *mach, std::string socket);

double  predictMemoryStatement(const ASTRequiresStatement *req, std::string socket, 
    double inner_parallelism);

double  getApplicationParam(ASTAppModel *app, std::string param);

const ASTMachComponent* getSocketComponent(ASTMachModel *mach, std::string socket);

double getAnyMachineProperty(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property);

double analyticalStreamingAccess(double D, double E, double S, double CL);
double callAnalyticalModel(double D, double E, double S, double CL);

double
recursiveBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    double outer_parallelism,
    const ASTControlStatement *s);



double executeBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket,
    const ASTExecutionBlock *exec,
    double outer_parallelism);

ASTAppModel*  getAppModel() { return app;}
ASTMachModel*  getMachineModel() {return mach;}

};

#endif



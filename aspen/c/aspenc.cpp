/* Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information. */
#include "aspenc.h"

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "mach/ASTMachComponent.h"
#include "parser/Parser.h"

#include <cstring>

// Structures
//   most types are just re-interpreted pointers to the original
//   C++ type, but there are some exceptions
struct ParamMap_t
{
    NameMap<const Expression*>::iterator  iterator;
    NameMap<const Expression*>           *paramMap;
};

// ----------------------------------------------------------------------------
// Parser

AppModel_p
Aspen_LoadAppModel(const char *fn)
{
    ASTAppModel *app = LoadAppModel(fn);
    return reinterpret_cast<AppModel_p>(app);
}

MachModel_p
Aspen_LoadMachModel(const char *fn)
{
    ASTMachModel *amm = LoadMachineModel(fn);
    return reinterpret_cast<MachModel_p>(amm);
}

// ----------------------------------------------------------------------------
// AppModel

const char*
AppModel_GetName(AppModel_p a)
{
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);
    return app->GetName().c_str();
}

Expression_p
AppModel_GetGlobalArraySizeExpression(AppModel_p a)
{
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);
    return reinterpret_cast<Expression_p>(app->GetGlobalArraySizeExpression());
}

Kernel_p AppModel_GetMainKernel(AppModel_p a)
{
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);
    return reinterpret_cast<Kernel_p>(app->GetMainKernel());
}

Expression_p
AppModel_GetResourceRequirementExpression(AppModel_p a, const char *res)
{
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);
    return reinterpret_cast<Expression_p>(app->GetResourceRequirementExpression(res));
}

ParamMap_p
AppModel_GetParamMap(AppModel_p a)
{
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);

    ParamMap_p p;
    p = (ParamMap_p)malloc(sizeof(struct ParamMap_t));
    p->paramMap = &(app->paramMap);
    p->iterator = p->paramMap->begin();
    return p;
}

// ----------------------------------------------------------------------------
// Kernel

const char *Kernel_GetName(Kernel_p k)
{
    const ASTKernel *kernel = reinterpret_cast<const ASTKernel*>(k);
    return kernel->GetName().c_str();
}

Expression_p
Kernel_GetTimeExpression(Kernel_p k, AppModel_p a,MachModel_p m,const char *s)
{
    const ASTKernel *kernel = reinterpret_cast<const ASTKernel*>(k);
    ASTAppModel *app = reinterpret_cast<ASTAppModel*>(a);
    ASTMachModel *amm = reinterpret_cast<ASTMachModel*>(m);
    return reinterpret_cast<Expression_p>(kernel->GetTimeExpression(app,amm,s));
}

// ----------------------------------------------------------------------------
// MachModel

MachComponent_p
MachModel_GetMachine(MachModel_p m)
{
    ASTMachModel *amm = reinterpret_cast<ASTMachModel*>(m);
    return reinterpret_cast<MachComponent_p>(amm->GetMachine());
}

ParamMap_p
MachModel_GetParamMap(MachModel_p a)
{
    ASTMachModel *mach = reinterpret_cast<ASTMachModel*>(a);

    ParamMap_p p;
    p = (ParamMap_p)malloc(sizeof(struct ParamMap_t));
    p->paramMap = &(mach->paramMap);
    p->iterator = p->paramMap->begin();
    return p;
}

// ----------------------------------------------------------------------------
// MachComponent

const char*
MachComponent_GetName(MachComponent_p m)
{
    const ASTMachComponent *mach = reinterpret_cast<const ASTMachComponent*>(m);
    return mach->GetName().c_str();
}

const char*
MachComponent_GetType(MachComponent_p m)
{
    const ASTMachComponent *mach = reinterpret_cast<const ASTMachComponent*>(m);
    return mach->GetType().c_str();
}

// ----------------------------------------------------------------------------
// ParamMap

ParamMap_p
ParamMap_Create(const char *e, double v)
{
    ParamMap_p p;
    p = (ParamMap_p)malloc(sizeof(struct ParamMap_t));
    p->paramMap = new NameMap<const Expression*>();
    p->iterator = p->paramMap->begin();

    (*(p->paramMap))[e] = new Real(v);
    return p;
}

void
ParamMap_BeginIteration(ParamMap_p p)
{
    p->iterator = p->paramMap->begin();
}

int
ParamMap_NextValue(ParamMap_p p, const char **n, Expression_p *e)
{
    if (p->iterator == p->paramMap->end())
        return 0;

    *n = p->iterator->first.c_str();
    *e = reinterpret_cast<Expression_p>(p->iterator->second);
    p->iterator++;
    return 1;
}

// ----------------------------------------------------------------------------
// Expression

char*
Expression_GetText(Expression_p e)
{
    const Expression *expr = reinterpret_cast<const Expression*>(e);
    return strdup(expr->GetText().c_str());
}

double
Expression_Evaluate(Expression_p e)
{
    const Expression *expr = reinterpret_cast<const Expression*>(e);
    return expr->Evaluate();
}

Expression_p
Expression_Expanded(Expression_p e, ParamMap_p p)
{
    const Expression *expr = reinterpret_cast<const Expression*>(e);
    return reinterpret_cast<Expression_p>(expr->Expanded(*(p->paramMap)));
}

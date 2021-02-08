// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "app/ASTSampleStatement.h"
#include "app/ASTRequiresStatement.h"
#include "walkers/AspenTool.h"
#include "walkers/ControlFlowWalker.h"
#include "walkers/CallStackTracer.h"
#include "walkers/ResourceCounter.h"

using namespace std;

class CompoundTool : public AspenTool
{
  protected:
    ControlFlowWalker *cfw;
    ResourceCounter *fc;
    CallStackTracer *cst;
    set<string> execute_stack_locations;
  public:
    CompoundTool(ASTAppModel *app) : AspenTool(app)
    {
        cfw = new ControlFlowWalker(app);
        fc = new ResourceCounter(app, "flops");
        cst = new CallStackTracer(app);
        AddTool(cfw);
        AddTool(fc);
        AddTool(cst);
    }
    double GetResult()
    {
        return fc->GetResult();
    }
    void PrintExecuteStacks()
    {
        for (set<string>::iterator it = execute_stack_locations.begin();
             it != execute_stack_locations.end(); it++)
        {
            cerr << "\n" << *it << endl;
        }
    }
    virtual void Execute(const ASTExecutionBlock *e)
    {
        execute_stack_locations.insert(cst->GetStackAsString());
    }
};

int main(int argc, char **argv)
{
    try {
        if (argc != 2)
        {
            cerr << "Usage: "<<argv[0]<<" [model.aspen]" << endl;
            return 1;
        }

        ASTAppModel *app = LoadAppModel(argv[1]);

        //app->Print(cout);

        AspenTool *t1 = new ControlFlowWalker(app);
        t1->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t1);

        cerr << endl;

        ResourceCounter *t2f = new ResourceCounter(app, "flops");
        ResourceCounter *t2l = new ResourceCounter(app, "loads");
        t2f->InitializeTraversal();
        t2l->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t2f);
        app->kernelMap["main"]->Traverse(t2l);
        cout << "Counted FLOPS = " << t2f->GetResult() << endl;
        cout << "Counted loads = " << t2l->GetResult() << endl;

        cerr << endl;

        CompoundTool *t3 = new CompoundTool(app);
        t3->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t3);
        t3->PrintExecuteStacks();
        cout << "Compound Tool counted FLOPS = " << t3->GetResult() << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

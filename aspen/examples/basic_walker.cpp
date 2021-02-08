// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "walkers/AspenTool.h"
#include "parser/Parser.h"

using namespace std;

class ExampleWalker : public AspenTool
{
    TraversalMode mode;
  public:
    ExampleWalker(ASTAppModel *app) : AspenTool(app)
    {
        mode = Implicit;
    }
    void SetTraversalModeHint(TraversalMode m)
    {
        mode = m;
    }
  protected:
    virtual TraversalMode TraversalModeHint(const ASTControlStatement *here)
    {
        return mode;
    }
    virtual void StartKernel(TraversalMode mode, const ASTKernel *k)
    {
        cout << Indent(level) << "Starting kernel '"<<k->GetName()<<"'\n";
    }

    virtual void StartIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        int c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        cout << Indent(level) << "Starting iterate "<<s->GetLabel()<<" (count="<<c<<")\n";
    }
    
    virtual void StartMap(TraversalMode mode, const ASTControlMapStatement *s)
    {
        int c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        cout << Indent(level) << "Starting map "<<s->GetLabel()<<" (count="<<c<<")\n";
    }

    virtual void Execute(const ASTExecutionBlock *e)
    {
        cout << Indent(level) << "Starting execution block "<<e->GetLabel()<<"\n";
    }
};

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " app.aspen" << endl;
        return -1;
    }

    try
    {
        ASTAppModel *app = LoadAppModel(argv[1]);

        const ASTKernel *k = app->mainKernel;
        ExampleWalker *walker = new ExampleWalker(app);

        cout << "\n\n---- Implicit Mode ----\n\n";
        walker->SetTraversalModeHint(AspenTool::Implicit);
        walker->InitializeTraversal();
        k->Traverse(walker);

        cout << "\n\n---- Explicit Mode ----\n\n";
        walker->SetTraversalModeHint(AspenTool::Explicit);
        walker->InitializeTraversal();
        k->Traverse(walker);
    }
    catch (const AspenException &exc)
    {
        cout << exc.PrettyString() << endl;
        return -1;
    }
}

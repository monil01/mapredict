// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"

using namespace std;

int main(int argc, char **argv)
{
  try {
    ASTAppModel *app = NULL;
    ASTMachModel *mach = NULL;

    bool success = false;
    if (argc == 2 || argc >= 4)
    {
        success = LoadAppOrMachineModel(argv[1], app, mach);
    }
    else
    {
        cerr << "Usage: "<<argv[0]<<" [app.aspen] [param value1 [value2 [value3]] ...]" << endl;
        cerr << "   for example: "<<argv[0]<<" md.aspen nAtoms 100 1000 10000" << endl;
        return 1;
    }

    if (!success)
    {
        cerr << "Errors encountered during parsing.  Aborting.\n";
        return -1;
    }

    if (!app)
    {
        cerr << "Expected an app model!\n";
    }

    cerr << "Success parsing app model.\n";

    const ASTKernel *k = app->mainKernel;

    int niter = 1.;
    string param = "";
    if (argc > 2)
    {
        niter = argc - 3;
        param = argv[2];
        cout << param << ",";
    }
    cout << "flops,"
         << "loads,"
         << "stores,"
         << "bytes,"
         << "msgbytes,"
         << "memory,"
         << endl;

    for (int iter = 0 ; iter < niter; ++iter)
    {
        NameMap<const Expression*> expansions(app->paramMap);
        if (param != "")
        {
            expansions[param] = new Real(strtod(argv[iter+3],NULL));
            cout << argv[iter+3] << ",";
        }

        cout << k->GetResourceRequirementExpression(app, "flops")->Expanded(expansions)->Evaluate() << ",";
        cout << k->GetResourceRequirementExpression(app, "loads")->Expanded(expansions)->Evaluate() << ",";
        cout << k->GetResourceRequirementExpression(app, "stores")->Expanded(expansions)->Evaluate() << ",";
        cout << k->GetResourceRequirementExpression(app, "bytes")->Expanded(expansions)->Evaluate() << ",";
        cout << k->GetResourceRequirementExpression(app, "messages")->Expanded(expansions)->Evaluate() << ",";
        cout << app->GetGlobalArraySizeExpression()->Expanded(expansions)->Evaluate() << ",";
        cout << endl;
    }

    delete app;

    return 0;
  }
  catch (const AspenException &exc)
  {
      cerr << exc.PrettyString() << endl;
      return -1;
  }
}

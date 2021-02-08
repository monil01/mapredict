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
    if (argc == 3 || argc >= 5)
    {
        success = LoadAppOrMachineModel(argv[1], app, mach);
    }
    else
    {
        cerr << "Usage: "<<argv[0]<<" <app.aspen> <resource> [param value1 [value2 [value3]] ...]" << endl;
        cerr << "   for example: "<<argv[0]<<" md.aspen flops nAtoms 100 1000 10000" << endl;
        cerr << "   or simply  : "<<argv[0]<<" md.aspen loads" << endl;
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

    string resource = argv[2];

    const ASTKernel *k = app->mainKernel;

    int niter = 1.;
    string param = "";
    if (argc > 3)
    {
        niter = argc - 4;
        param = argv[3];
        cout << param << "\t" << resource << endl;
        cout << "--\t--" << endl;
    }

    for (int iter = 0 ; iter < niter; ++iter)
    {
        NameMap<const Expression*> expansions(app->paramMap);
        if (param != "")
        {
            double val = strtod(argv[iter+4],NULL);
            expansions[param] = new Real(val);
            cout << val << "\t";
        }

        Expression *expr = NULL;
        if (resource == "memory")
            expr = app->GetGlobalArraySizeExpression();
        else
            expr = k->GetResourceRequirementExpression(app, resource);

        // debugging
        if (true && argc==3)
        {
            cout << "raw expression:\n";
            cout << expr->GetText();
            cout << endl << endl;

            cout << "expanded:\n";
            cout << expr->Expanded(expansions)->GetText();
            cout << endl << endl;

            cout << "simplified:\n";
            cout << expr->OneStepSimplified()->GetText();
            cout << endl << endl;

            cout << "expanded and simplified:\n";
            cout << expr->Expanded(expansions)->OneStepSimplified()->GetText();
            cout << endl << endl;
        }

        expr->ExpandInPlace(expansions);

        cout << expr->Evaluate() << endl;
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

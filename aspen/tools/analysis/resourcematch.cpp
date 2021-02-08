// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"

using namespace std;

double Evaluate(ASTAppModel *app,
                const ASTKernel *k,
                const string &resource,
                const string &param,
                double value,
                NameMap<const Expression*> &expansions)
{
    expansions[param] = new Real(value);
    if (resource == "memory")
        return app->GetGlobalArraySizeExpression()->Expanded(expansions)->Evaluate();
    else
        return k->GetResourceRequirementExpression(app, resource)->Expanded(expansions)->Evaluate();
}

pair<double,double> PositiveBisectionSearch(ASTAppModel *app,
                                            const ASTKernel *k,
                                            const string &resource,
                                            double target,
                                            const string &param)
{
    NameMap<const Expression*> expansions = app->paramMap;

    // to start, let's find the order-of-magnitude bracket

    double x0 = 1e-20;
    double f0 = Evaluate(app,k,resource,param,x0,expansions);
    bool   s0 = (f0-target)>0;
    if (f0 == target)
    {
        cerr << "found exact result in magnitude bracket search" << endl;
        return pair<double,double>(x0,f0);
    }

    double x1;
    double f1;
    bool   s1;
    for (int mag = -19; mag <= +20; ++mag)
    {
        x1 = pow(10., mag);
        f1 = Evaluate(app,k,resource,param,x1,expansions);
        s1 = (f1-target)>0;

        if (f1 == target)
        {
            cerr << "found exact result in magnitude bracket search" << endl;
            return pair<double,double>(x1,f1);
        }

        if (s1 != s0)
            break;

        x0 = x1;
        f0 = f1;
        s0 = s1;
    }

    cerr << "found brackets: ("<<x0<<","<<x1<<")"
         << " -> ("<<f0<<","<<f1<<")" <<endl;

    // Now do bisection.

    // Starting with the right order of magnitude, we need about
    // 25 iterations for exact single precision match, 50 for double.
    // But evaluations don't take long, might as well keep it high.
    int maxiter = 200;
    for (int iter = 0; iter < maxiter; ++iter)
    {
        double x2 = (x0 + x1) / 2.;
        double f2 = Evaluate(app,k,resource,param,x2,expansions);
        bool   s2 = (f2-target)>0;
        //if (f2 == target) // double precision exactness
        if (float(f2) == float(target)) // single precision exactness
        {
            cerr << "found exact result at iter " << iter << endl;
            return pair<double,double>(x2,f2);
        }

        if (s2 == s0)
        {
            x0 = x2;
            f0 = f2;
            s0 = s2;
        }
        else // (s2 == s1)
        {
            x1 = x2;
            f1 = f2;
            s1 = s2;
        }
    }

    return pair<double,double>(-1,-2);
}

int main(int argc, char **argv)
{
  try {
    ASTAppModel *app = NULL;
    ASTMachModel *mach = NULL;

    bool success = false;
    if (argc == 5)
    {
        success = LoadAppOrMachineModel(argv[1], app, mach);
    }
    else
    {
        cerr << "Usage: "<<argv[0]<<" <app.aspen> <resource> <target> <param>" << endl;
        cerr << "   for example: "<<argv[0]<<" md.aspen flops 1e9 nAtoms" << endl;
        cerr << "   ...will find the value for nAtoms which results in 1 billion flops." << endl;
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
    double target   = strtod(argv[3], NULL);
    string param    = argv[4];

    const ASTKernel *k = app->mainKernel;

    pair<double,double> result = PositiveBisectionSearch(app, k,
                                                         resource, target,
                                                         param);

    cout << "Best match: "<<param<<" == " << result.first << endl;
    cout << "Results in: "<<resource<<" == " << result.second
         << " (target=="<<target<<")" << endl;

    return 0;
  }
  catch (const AspenException &exc)
  {
      cerr << exc.PrettyString() << endl;
      return -1;
  }
}

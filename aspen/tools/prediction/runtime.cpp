// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"

using namespace std;

class ContainsIdentifier : public ModelVisitor
{
    string name;
    bool contains;
  public:
    ContainsIdentifier(string name) : name(name)
    {
        contains = false;
    }
    virtual ~ContainsIdentifier()
    {
    }
    bool WasContained() { return contains; }
    virtual bool Visit(ASTNode *astnode)
    {
        Identifier *ident = dynamic_cast<Identifier*>(astnode);
        if (ident)
        {
            //cerr << "checking ident " << ident->name << endl;
            contains |= (ident->GetName() == name);
        }
        return false;
    }
};

int main(int argc, char **argv)
{
  try {
    ASTAppModel *app = NULL;
    ASTMachModel *mach = NULL;

    bool success = false;
    
    if (argc >= 3)
        success = LoadAppAndMachineModels(argv[1], argv[2], app, mach);

    if (argc < 4 || argc == 5 || !success)
    {
        cerr << "Usage: "<<argv[0]<<" <app.aspen> <mach.aspen> <socket> [param value1 [value2 [value3]] ...]" << endl;
        cerr << "   for example: "<<argv[0]<<" md.aspen keeneland.aspen nvidia_m2090 nAtoms 100 1000 10000" << endl;
        cerr << "   or simply:   "<<argv[0]<<" md.aspen keeneland.aspen nvidia_m2090" << endl;
        if (success && argc == 3)
        {
            vector<string> socketnames = mach->socketlist;
            cerr << endl << "Please select a socket." << endl;
            cerr << "Valid choices for your selected machine model are:" << endl;
            for (unsigned int i=0; i<socketnames.size(); ++i)
            {
                cerr << "  " << socketnames[i] << endl;
            }
        }
        return 1;
    }

    string socket = argv[3];

    const ASTKernel *mainkernel = app->mainKernel;
    if (!mainkernel)
    {
        cerr << "No main kernel; iterating across kernels" << endl;
    }

    int nkernels = mainkernel ? 1 : app->GetKernels().size();
    for (int kernel = 0; kernel < nkernels; ++kernel)
    {
        const ASTKernel *k = mainkernel;
        if (!k)
        {
            k = app->GetKernels()[kernel];
            cout << endl << "kernel: " << k->GetName() << endl << endl;
        }

        int niter = 1.;
        string param = "";
        if (argc > 5)
        {
            niter = argc - 5;
            param = argv[4];
            if (argc > 6)
            {
                cout << param << "\t" << "runtime" << endl;
                cout << "--\t--" << endl;
            }
        }

        NameMap<const Expression*> app_expansions(app->paramMap);
        NameMap<const Expression*> mach_expansions(mach->paramMap);
        if (param != "")
        {
            app_expansions.Erase(param);
            mach_expansions.Erase(param);

            // debugging printout:
            if (false)
            {
                k->GetTimeExpression(app, mach, socket)
                    ->Expanded(app_expansions)
                    ->Expanded(mach_expansions)
                    ->Simplified()
                    ->Print(cerr);
            }

            if (false)
            {
                cerr << k->GetTimeExpression(app, mach, socket)
                    ->Expanded(app_expansions)
                    ->Expanded(mach_expansions)
                    ->GetText() << endl;
                cerr << k->GetTimeExpression(app, mach, socket)
                    ->Expanded(app_expansions)
                    ->Expanded(mach_expansions)
                    ->Simplified()
                    ->GetText() << endl;
            }
        }
        for (int iter = 0 ; iter < niter; ++iter)
        {
            NameMap<const Expression*> param_expansion;
            if (param != "")
            {
                double val = strtod(argv[iter+5],NULL);
                param_expansion[param] = new Real(val);
                if (argc > 6)
                    cout << val << "\t";
            }

            bool intermediate_debug = false;
        
            ContainsIdentifier *c = new ContainsIdentifier(param);

            Expression *t = k->GetTimeExpression(app, mach, socket);
            t->Visit(c);

            // This is now a little more robust; it's now possible to expand
            // an array before turning its index into a real value.
            // As such, the order of expansion doesn't matter as much,
            // except for efficiency.  Generally, since you can't
            // insert an array as a command-line parameter, but it's
            // conversely VERY likely you will use an array INDEX as
            // a comamnd-lime parameter, let's do these command-line
            // parameter expansions first, so that by the time we
            // want to expand an array dereference, it's more likely
            // we can just look up the right real value.  (I.e. greatest
            // efficiency.)  

            // For now, though, we still need to expand parameters again at
            // the end, because we may not have generated the thing we're
            // going to expand until we're done with other expansions.


            t->ExpandInPlace(param_expansion);
            t->Visit(c);
            if (intermediate_debug)
            {
                cout << t->GetText() << endl;
            }

            t->ExpandInPlace(app_expansions);
            t->Visit(c);
            if (intermediate_debug)
            {
                cout << t->GetText() << endl;
            }

            t->ExpandInPlace(mach_expansions);
            t->Visit(c);
            if (intermediate_debug)
            {
                cout << t->GetText() << endl;
            }

            if (param != "" && !c->WasContained())
            {
                cerr << "Warning: expression never contained parameter '"<<param<<"'\n";
            }


            t->ExpandInPlace(param_expansion);
            if (intermediate_debug)
            {
                cout << t->GetText() << endl;
            }

            // debugging printout to test simplify:
            if (false)
            {
                cout << "check:\t" << t->Simplified()->Evaluate() << endl;
            }

            cout << std::setprecision(12) << t->Evaluate() << endl;
            delete t;
        }
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

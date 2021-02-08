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
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " app.aspen mach.aspen <socket>" << endl;
        return -1;
    }

    try
    {
        ASTAppModel *app = NULL;
        ASTMachModel *mach = NULL;

        LoadAppAndMachineModels(argv[1], argv[2], app, mach);

        string socket = argv[3];

        const ASTKernel *k = app->mainKernel;
        Expression *t = k->GetTimeExpression(app, mach, socket);

        cout << endl;
        cout << "Expression for runtime: " << endl;
        cout << t->GetText() << endl << endl;

        cout << "... simplified: " << endl;
        cout << t->Simplified()->GetText() << endl << endl;

        t->ExpandInPlace(app->paramMap);
        t->ExpandInPlace(mach->paramMap);

        cout << "... with parameter expansion: " << endl;
        cout << t->GetText() << endl << endl;

        cout << "Value for runtime: " << t->Evaluate() << endl << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

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
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " app.aspen <resource>" << endl;
        return -1;
    }

    try
    {
        ASTAppModel *app = LoadAppModel(argv[1]);
        string resource = argv[2];

        const ASTKernel *k = app->mainKernel;

        Expression *expr = k->GetResourceRequirementExpression(app, resource);

        cout << endl;
        cout << "Expression for '"<<resource<<"': " << endl;
        cout << expr->GetText() << endl << endl;

        cout << "... simplified: " << endl;
        cout << expr->Simplified()->GetText() << endl << endl;

        expr->ExpandInPlace(app->paramMap);

        cout << "... with parameter expansion: " << endl;
        cout << expr->GetText() << endl << endl;

        cout << "... and as a value: " << expr->Evaluate() << endl << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

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

        if (argc != 5)
        {
            cerr << "or:    "<<argv[0]<<" [model.aspen] kernel param mapname" << endl;
            return 1;
        }

        ASTAppModel *app = NULL;
        ASTMachModel *mach = NULL;
        bool success = false;
        success = LoadAppOrMachineModel(argv[1], app, mach);
        if (!success)
            THROW(ModelError, "Unknown error parsing model");

        string kernel = argv[2];
        string param = argv[3];
        string mapname = argv[4];

        const ASTKernel *k = app->kernelMap[kernel];

        Expression *dssize = k->GetSingleRegionInclusiveDataSizeExpression(app, mapname);

        NameMap<const Expression*> app_expansions(app->paramMap);
        app_expansions.Erase(param);

        dssize = dssize->Expanded(app_expansions)->Simplified();

        cout << "#include <cmath>" << endl;
        cout << "double max(double a, double b) {return a<b ? b : a;}" << endl;
        cout << "double max(double a, double b, double c) {return max(max(a,b),c);}" << endl;
        cout << "double dssize(double "<<param<<")" << endl;
        cout << "{" << endl;
        cout << "    return " << dssize->GetText(Expression::C) << ";" << endl;
        cout << "}" << endl;
        cout << endl;
        cout << "#if 0" << endl;
        cout << "#include <iostream>" << endl;
        cout << "using namespace std;" << endl;
        cout << "int main()" << endl;
        cout << "{" << endl;
        cout << "    for (int magnitude=-20; magnitude<=20; ++magnitude)" << endl;
        cout << "    {" << endl;
        cout << "        double value = pow(10.0, magnitude);" << endl;
        cout << "        cerr << \"at "<<param<<"=\"<<value<<\", dssize() returns \"<<dssize(value)<<endl;" << endl;
        cout << "    }" << endl;
        cout << "    return 0;" << endl;
        cout << "}" << endl;
        cout << "#endif" << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

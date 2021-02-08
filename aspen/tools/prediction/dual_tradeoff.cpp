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

        if (argc != 9)
        {
            cerr << "Usage: "<<argv[0]<<" [model.aspen] [machine0.aspen] [machine1.aspen] kernel socket0 socket1 param searchparam" << endl;
            return 1;
        }

        ASTAppModel *app = NULL;
        ASTAppModel *app0_dummy = NULL;
        ASTAppModel *app1_dummy = NULL;
        ASTMachModel *mach_dummy = NULL;
        ASTMachModel *mach0 = NULL;
        ASTMachModel *mach1 = NULL;
        bool success = false;
        success = LoadAppOrMachineModel(argv[1], app, mach_dummy);
        if (!success)
            THROW(ModelError, "Unknown error parsing model");
        success = LoadAppOrMachineModel(argv[2], app0_dummy, mach0);
        if (!success)
            THROW(ModelError, "Unknown error parsing model");
        success = LoadAppOrMachineModel(argv[3], app1_dummy, mach1);
        if (!success)
            THROW(ModelError, "Unknown error parsing model");

        if (app0_dummy || app1_dummy || mach_dummy)
            THROW(ModelError, "Expected an app model followed by two machine models");

        if (!app || !mach0 || !mach1)
            THROW(ModelError, "Expected an app model followed by two machine models");

        string kernel = argv[4];
        string socket0 = argv[5];
        string socket1 = argv[6];
        string param = argv[7];
        string searchparam = argv[8];

        MappingRestriction restriction0("socket", socket0);
        MappingRestriction restriction1("socket", socket1);

        const ASTKernel *k = app->kernelMap[kernel];
        Expression *expr0 = k->GetTimeExpression(app, mach0, socket0);
        Expression *expr1 = k->GetTimeExpression(app, mach1, socket1);

        Expression *dssize = k->GetInclusiveDataSizeExpression(app);
        Expression *xfer0 = mach0->GetMachine()->GetIdealizedTimeExpression("intracomm",
                                                                               vector<ASTTrait*>(),
                                                                               dssize->Cloned(),
                                                                               restriction0);
        Expression *xfer1 = mach1->GetMachine()->GetIdealizedTimeExpression("intracomm",
                                                                               vector<ASTTrait*>(),
                                                                               dssize->Cloned(),
                                                                               restriction1);

        NameMap<const Expression*> app_expansions(app->paramMap);
        NameMap<const Expression*> mach0_expansions(mach0->paramMap);
        NameMap<const Expression*> mach1_expansions(mach1->paramMap);
        app_expansions.Erase(param);
        mach0_expansions.Erase(param);
        mach1_expansions.Erase(param);
        app_expansions.Erase(searchparam);
        mach0_expansions.Erase(searchparam);
        mach1_expansions.Erase(searchparam);

        expr0 = expr0->Expanded(app_expansions)->Expanded(mach0_expansions)->Simplified();
        expr1 = expr1->Expanded(app_expansions)->Expanded(mach1_expansions)->Simplified();
        xfer0 = xfer0->Expanded(app_expansions)->Expanded(mach0_expansions)->Simplified();
        xfer1 = xfer1->Expanded(app_expansions)->Expanded(mach1_expansions)->Simplified();

        string rt0 = "rt_" + socket0;
        string rt1 = "rt_" + socket1;
        string xf0 = "xf_" + socket0;
        string xf1 = "xf_" + socket1;

        cout << "#include <iostream>" << endl;
        cout << "using namespace std;" << endl;
        cout << "#include <math.h>" << endl;
        cout << "double min(double a, double b) {return a>b ? b : a;}" << endl;
        cout << "double min3(double a, double b, double c) {return min(min(a,b),c);}" << endl;
        cout << "double max(double a, double b) {return a<b ? b : a;}" << endl;
        cout << "double max3(double a, double b, double c) {return max(max(a,b),c);}" << endl;
        cout << endl;
        cout << "#define " << rt0 << " 0 + " << expr0->GetText(Expression::C) << "" << endl;
        cout << "#define " << rt1 << " 0 + " << expr1->GetText(Expression::C) << "" << endl;
        cout << "#define " << xf0 << "  " << xfer0->GetText(Expression::C) << "" << endl;
        cout << "#define " << xf1 << "  " << xfer1->GetText(Expression::C) << "" << endl;
        cout << endl;
        cout << "int check(double "<<param<<", double "<<searchparam<<")" << endl;
        cout << "{" << endl;
        cout << "    return " << rt0 << " + " << xf0 << " < " << rt1 << " + " << xf1 << ";" << endl;
        cout << "}" << endl;
        cout << endl;
        cout << "double crossover(double "<<param<<")" << endl;
        cout << "{" << endl;
        cout << "    int N = 1e-6;" << endl;
        cout << "    int startcheck = check("<<param<<", N);" << endl;
        cout << "    //cout << \"start:\" << "<<param<<" << \", \" << "<<rt0<<" << \", \" << "<<xf0<<" << \", \" << "<<rt1<<" << \", \" << "<<xf1<<" << \", \" << ("<<rt0<<"+"<<xf0<<") << \", \" << ("<<rt1<<"+"<<xf1<<") << \", \" << check("<<param<<", "<<searchparam<<") << endl;" << endl;
        cout << "    for (double mag = -6; mag < 10; mag += 0.001)" << endl;
        cout << "    {" << endl;
        cout << "        double "<<searchparam<<" = pow(10.0, mag);" << endl;
        cout << "        int curcheck = check("<<param<<", "<<searchparam<<");" << endl;
        cout << "        if (startcheck != curcheck)" << endl;
        cout << "        {" << endl;
        cout << "            //cout << "<<param<<" << \", \" << "<<rt0<<" << \", \" << "<<xf0<<" << \", \" << "<<rt1<<" << \", \" << "<<xf1<<" << \", \" << ("<<rt0<<"+"<<xf0<<") << \", \" << ("<<rt1<<"+"<<xf1<<") << \", \" << check("<<param<<", "<<searchparam<<") << endl;" << endl;
        cout << "            return "<<searchparam<<";" << endl;
        cout << "        }" << endl;
        cout << "    }" << endl;
        cout << "}" << endl;
        cout << endl;
        cout << "#ifndef VALUES" << endl;
        cout << "#define VALUES {16,32,64,128,256,768,512,1024,1536,2048,3072,4096,8192}" << endl;
        cout << "#endif //VALUES" << endl;
        cout << "int main()" << endl;
        cout << "{" << endl;
        cout << "    cout << \""<<param<<", runtime "<<socket0<<", xfer "<<socket0<<", runtime "<<socket1<<", xfer "<<socket1<<", total "<<socket0<<", total "<<socket1<<", faster\" << endl;" << endl;
        cout << "    const double values[] = VALUES;" << endl;
        cout << "    int nvalues = sizeof(values) / sizeof(values[0]);" << endl;
        cout << "    for (int i=0; i<nvalues; ++i)" << endl;
        cout << "    {" << endl;
        cout << "        double "<<param<<" = values[i];" << endl;
        cout << "        cout << "<<param<<" << \", \" << crossover("<<param<<") << endl;" << endl;
        cout << "    }" << endl;
        cout << "    return 0;" << endl;
        cout << "}" << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

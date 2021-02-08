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
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " mach.aspen" << endl;
        return -1;
    }

    try
    {
        ASTMachModel *mach = LoadMachineModel(argv[1]);

        string socket = mach->socketlist[0];
        MappingRestriction restriction("socket", socket);

        cout << endl;
        cout << "Analysis using first socket ("<<socket<<")" << endl;
        cout << endl;

        // We're going to calculate in GB/sec and GFLOPS/sec, so
        // we'll calculate runtime for 1 billion flops and 1 billion bytes.
        Real *OneBillion = new Real(1.e9);

        // single precision
        const vector<ASTTrait*> DefaultTraits;
        Expression *sp = mach->GetMachine()->GetIdealizedTimeExpression("flops", DefaultTraits,
                                                                           OneBillion, restriction);
        cout << "Expression for runtime of 1e9 SP flops:" << endl;
        cout << sp->GetText() << endl << endl;

        cout << "... simplified:" << endl;
        cout << sp->Simplified()->GetText() << endl << endl;

        cout << "... which gives us a peak of: "
             << 1. / sp->Expanded(mach->paramMap)->Evaluate()
             << " SP GFLOPS/sec" << endl << endl;

        // double precision
        const vector<ASTTrait*> DoublePrecTraits(1,new ASTTrait("dp"));
        Expression *dp = mach->GetMachine()->GetIdealizedTimeExpression("flops", DoublePrecTraits,
                                                                           OneBillion, restriction);
        cout << "And for double precision: "
             << 1. / dp->Expanded(mach->paramMap)->Evaluate()
             << " DP GFLOPS/sec" << endl << endl;


        // memory bandwidth
        Expression *mem = mach->GetMachine()->GetIdealizedTimeExpression("loads", DefaultTraits,
                                                                            OneBillion, restriction);
                                                         
        cout << "Expression for runtime of 1e9 loaded bytes:" << endl;
        cout << mem->GetText() << endl << endl;

        cout << "... which gives us a peak of: "
             << 1. / mem->Expanded(mach->paramMap)->Evaluate()
             << " GB/sec" << endl << endl;

    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

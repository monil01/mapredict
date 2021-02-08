// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/AspenTool.h"
#include "walkers/RuntimeExpression.h"

#include <sys/time.h>

int main(int argc, char **argv)
{
    try {
        ASTAppModel *app = NULL;
        ASTMachModel *mach = NULL;
        if (argc >= 3)
        {
            app = LoadAppModel(argv[1]);
            mach = LoadMachineModel(argv[2]);
        }

        if (argc != 4 || !app || !mach)
        {
            cerr << "Usage: "<<argv[0]<<" [app.aspen] [mach.aspen] [socket]" << endl;
            return 1;
        }

        string socket = argv[3];

        cerr << "Parsed; calculating runtime\n";

        struct timeval tv;
        gettimeofday(&tv,NULL);
        srand(tv.tv_usec);

        RuntimeExpression *t = new RuntimeExpression(app, mach, socket);
        t->SetCacheExecutionBlockExpressions(false);
        t->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t);
        cout << std::setprecision(12) << t->GetResult()->GetText() << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

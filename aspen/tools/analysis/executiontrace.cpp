// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/AspenTool.h"
#include "walkers/ControlFlowWalker.h"

#include <sys/time.h>


int main(int argc, char **argv)
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    srand(tv.tv_usec);

    try {
        if (argc != 2)
        {
            cerr << "Usage: "<<argv[0]<<" [model.aspen]" << endl;
            return 1;
        }

        ASTAppModel *app = LoadAppModel(argv[1]);

        AspenTool *t1 = new ControlFlowWalker(app);
        t1->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t1);

        cerr << endl;

    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

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
    if (argc == 2)
    {
        success = LoadAppOrMachineModel(argv[1], app, mach);
    }
    else
    {
        cerr << "Usage: "<<argv[0]<<" [model.aspen]" << endl;
        return 1;
    }

    if (!success)
    {
        cerr << "Errors encountered during parsing.  Aborting.\n";
        return -1;
    }

    if (mach)
        mach->Export(cout);
    if (app)
        app->Export(cout);

    if (app)
        delete app;
    if (mach)
        delete mach;

    return 0;
  }
  catch (const AspenException &exc)
  {
    cerr << exc.PrettyString() << endl;
    return -1;
  }
}

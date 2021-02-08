// Copyright 2013-2016 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "walkers/ResourceExpression.h"
#include "walkers/ResourceCounter.h"
#include "parser/Parser.h"

using namespace std;

class ResourceWalker : public AspenTool
{
  public:
    ResourceExpression *re;
    ResourceCounter    *rc;

    ResourceWalker(ASTAppModel *app, string res) : AspenTool(app)
    {
        AddTool(re = new ResourceExpression(app, res));
        AddTool(rc = new ResourceCounter(app, res));
    }
};

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <app.aspen> <resource>" << endl;
        return -1;
    }

    try
    {
        ASTAppModel *app = LoadAppModel(argv[1]);

        ///\todo:  my thought was that by using a compound tool, we can
        /// have both the expression and counter evaluate
        /// the same app (i.e. using the same values for random samples),
        /// but that's not true; each derives from "tool" and so 
        /// gets its own paramStack.  Fix that, I think.....
        /// (It may be a big fix, unfortunately.)
        ResourceWalker *rw = new ResourceWalker(app, argv[2]);
        rw->InitializeTraversal();
        app->mainKernel->Traverse(rw);

        cout << rw->re->GetResult()->GetText() << endl;
        cout << rw->rc->GetResult() << endl;

    }
    catch (const AspenException &exc)
    {
        cout << exc.PrettyString() << endl;
        return -1;
    }
}

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
#include "walkers/MultiplicityTracker.h"
#include "walkers/CallStackTracer.h"

#include <algorithm>
using namespace std;

class GProfTool : public AspenTool
{
  protected:
    ASTMachModel *mach;
    string socket;

    ControlFlowWalker *cfw;
    CallStackTracer *cst;
    MultiplicityTracker *mt;
    map<string, double > incl_runtimes;
    map<string, double > excl_runtimes;
    map<string, int > kernel_calls;
    map<string, set<string> > parents;
    map<string, set<string> > children;

    static bool rtcomp(pair<string,double> a, pair<string,double> b)
    {
        return a.second > b.second;
    }

  public:
    GProfTool(ASTAppModel *app, ASTMachModel *mach, string socket)
        : AspenTool(app), mach(mach), socket(socket)
    {
        cfw = new ControlFlowWalker(app);
        cst = new CallStackTracer(app);
        mt = new MultiplicityTracker(app);
        //AddTool(cfw = new ControlFlowWalker(app)); // for debug
        AddTool(cst);
        AddTool(mt);
    }
    virtual TraversalMode TraversalModeHint()
    {
        //return Explicit;
        return Implicit;
    }
    void Print()
    {
        double total_runtime = 0;

        // inclusive (i.e. including children) runtimes
        vector<pair<string,double> > kernels_by_inclruntime;
        for (map<string, double>::iterator it = incl_runtimes.begin();
             it != incl_runtimes.end(); it++)
        {
            kernels_by_inclruntime.push_back(*it);
        }
        std::sort(kernels_by_inclruntime.begin(), kernels_by_inclruntime.end(), rtcomp);

        // exclusive (i.e. self only) runtimes
        vector<pair<string,double> > kernels_by_exclruntime;
        for (map<string, double>::iterator it = excl_runtimes.begin();
             it != excl_runtimes.end(); it++)
        {
            kernels_by_exclruntime.push_back(*it);
            total_runtime += it->second;
        }
        std::sort(kernels_by_exclruntime.begin(), kernels_by_exclruntime.end(), rtcomp);

        // create kernel index
        map<string,int> kernel_index;
        for (unsigned int i=0; i<kernels_by_inclruntime.size(); ++i)
        {
            kernel_index[kernels_by_inclruntime[i].first] = i;
        }

        // flat profile
        printf("\n\n\n");
        printf("Flat profile:\n\n");
        double cum = 0;
        printf("%%time   cum    self    calls  selfms/call  totalms/call    name\n");
        for (unsigned int i=0; i<kernels_by_exclruntime.size(); ++i)
        {
            string name = kernels_by_exclruntime[i].first;
            double self = kernels_by_exclruntime[i].second;
            double pct = self / total_runtime;
            int calls = kernel_calls[name];
            double self_call = self / double(calls);
            double total_call = incl_runtimes[name] / double(calls);
            cum += self;
            printf("%5.2f  %5.2f  %5.2f   % 6d  %9.3f    %9.3f       %s\n",
                   pct*100, cum, self, calls, self_call*1000, total_call*1000, name.c_str());
            //cout << kernels_by_runtime[i].first << ": \t" << kernels_by_runtime[i].second << endl;
        }

        // flat profile
        printf("\n\n\n");
        printf("Call graph:\n\n");
        printf("index  %%time     self   children      name\n");
        for (unsigned int i=0; i<kernels_by_inclruntime.size(); ++i)
        {
            printf("------------------------------------------------------------\n");
            string name = kernels_by_inclruntime[i].first;
            double time = kernels_by_inclruntime[i].second;
            double pct  = time / total_runtime;
            const set<string> &par = parents[name];
            const set<string> &chi = children[name];
            for (set<string>::iterator j = par.begin(); j != par.end(); j++)
            {
                string n = *j;
                printf("                %5.2f    %5.2f            %s [%d]\n",
                       excl_runtimes[n],
                       incl_runtimes[n] - excl_runtimes[n],
                       n.c_str(),
                       kernel_index[n]+1);
            }
            printf("[%3d]  %5.1f    %5.2f    %5.2f        %s [%d]\n",
                   i+1,
                   pct*100.,
                   excl_runtimes[name],
                   incl_runtimes[name] - excl_runtimes[name],
                   name.c_str(),
                   i+1);
            for (set<string>::iterator j = chi.begin(); j != chi.end(); j++)
            {
                string n = *j;
                printf("                %5.2f    %5.2f            %s [%d]\n",
                       excl_runtimes[n],
                       incl_runtimes[n] - excl_runtimes[n],
                       n.c_str(),
                       kernel_index[n]+1);
            }
        }
    }
    virtual void EndKernel(TraversalMode mode, const ASTKernel *k)
    {
        string n = k->GetName();
        if (!cst->EmptyStack())
        {
            string p = cst->GetTopCaller();//GetParentInCallStack();
            parents[n].insert(p);
            children[p].insert(n);
        }

        Expression *ire = k->GetTimeExpression(app, mach, socket);
        ire->ExpandInPlace(app->paramMap);
        ire->ExpandInPlace(mach->paramMap);
        double ir = ire->Evaluate();
        delete ire;

        Expression *ere = k->GetSelfTimeExpression(app, mach, socket);
        ere->ExpandInPlace(app->paramMap);
        ere->ExpandInPlace(mach->paramMap);
        double er = ere->Evaluate();
        delete ere;

        double mult = mt->GetMultiplicity();

        //cerr << Indent(level+1) << "call "<<n<<" multiplicity = "<<mult<< " (stacksize="<<mt->stack.size()<<")\n";
        //cerr << Indent(level+2) << " er="<<er<<" ir="<<ir<<" ir-er="<<ir-er<<endl;

        incl_runtimes[k->GetName()] += ir*mult;
        excl_runtimes[k->GetName()] += er*mult;
        kernel_calls[k->GetName()] += 1*mult;
    }
};


int main(int argc, char **argv)
{
    try {
        if (argc != 4)
        {
            cerr << "Usage: "<<argv[0]<<" [app.aspen] [mach.aspen] [socket]" << endl;
            return 1;
        }

        ASTAppModel *app = LoadAppModel(argv[1]);
        ASTMachModel *mach = LoadMachineModel(argv[2]);
        string socket = argv[3];

        GProfTool *t4 = new GProfTool(app, mach, socket);
        t4->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t4);
        t4->Print();
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

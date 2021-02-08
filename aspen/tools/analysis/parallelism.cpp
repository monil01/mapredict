// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "app/ASTSampleStatement.h"
#include "parser/Parser.h"
#include "walkers/AspenTool.h"

class CallStackTracer : public AspenTool
{
    vector<string> stack;
  public:
    CallStackTracer(ASTAppModel *app) : AspenTool(app)
    {
    }
  protected:
    virtual TraversalMode TraversalModeHint(const ASTControlStatement *here)
    {
        //return Explicit;
        return Implicit;
    }
    virtual void StartKernel(TraversalMode mode, const ASTKernel *k)
    {
        stack.push_back(k->GetName());
    }

    virtual void EndKernel(TraversalMode mode, const ASTKernel *s)
    {
        stack.pop_back();
    }

  public:
    virtual vector<string> GetStackAsStringVector()
    {
        return stack;
    }
    virtual bool EmptyStack()
    {
        return stack.empty();
    }
    virtual string GetTopCaller()
    {
        if (stack.size() < 1)
            return "__START__";
        return stack.back();
    }
    virtual string GetStackAsString()
    {
        string result;
        for (unsigned int i=0; i<stack.size(); i++)
        {
            if (i>0)
                result += "->";
            result += stack[i];
        }
        return result;
    }
};

class ParallelismCounter : public AspenTool
{
    CallStackTracer *cst;

    bool debug;

    bool cache;
    map<const ASTExecutionBlock *, double> expr_cache;

    vector<double> parstack;
    void PushPar(double v)
    {
        parstack.push_back(v);
    }
    const double PopPar()
    {
        double v = parstack.back();
        parstack.pop_back();
        return v;
    }
    const double CurrentPar() const
    {
        return parstack.back();
    }

  public:
    ParallelismCounter(ASTAppModel *app)
        : AspenTool(app)
    {
        AddTool(cst = new CallStackTracer(app));
        cache = true;
        debug = true;
        parstack.reserve(1000);
        PushPar(1);
    }
    double GetResult()
    {
        if (parstack.size() != 1)
            THROW(LogicError, "Par Stack size was not 1");
        return 0;
    }
    void SetCacheExecutionBlockExpressions(bool c)
    {
        cache = c;
        if (!cache)
            expr_cache.clear();
    }


  protected:
    virtual TraversalMode TraversalModeHint(const ASTControlStatement *here)
    {
        return Implicit;
    }

    virtual void StartKernel(TraversalMode mode, const ASTKernel *k)
    {
        if (debug) cerr << Indent(level) << "Starting kernel '"<<k->GetName()<<"'\n";
    }
    virtual void StartIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        if (debug) cerr << Indent(level) << "start iterate, stack size="<<parstack.size()<<", par="<<CurrentPar()<<endl;
    }

    virtual void EndIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        double c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        if (mode == Explicit)
        {
            if (debug) cerr << Indent(level) << "iterate, stack size="<<parstack.size()<<", popping "<<c<<" items and pushing 1\n";
        }
        else
        {
            if (debug) cerr << Indent(level) << "iterate, stack size="<<parstack.size()<<", popping a single item and pushing 1\n";
        }
    }
    
    virtual void StartMap(TraversalMode mode, const ASTControlMapStatement *s)
    {
        double c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        double oldpar = CurrentPar();
        double newpar = oldpar * c;
        PushPar(newpar);

        if (debug) cerr << Indent(level) << "map (start), par now="<<CurrentPar()<<",stack size="<<parstack.size()<<", popping "<<c<<" items and pushing 1\n";
    }
    virtual void EndMap(TraversalMode mode, const ASTControlMapStatement *s)
    {
        double c = s->GetQuantity()->Expanded(app->paramMap)->Evaluate();
        PopPar();

        if (mode == Explicit)
        {
            if (debug) cerr << Indent(level) << "map (end), par now="<<CurrentPar()<<", stack size="<<parstack.size()<<", popping "<<c<<" items and pushing 1\n";
        }
        else
        {
            if (debug) cerr << Indent(level) << "map (end), par now="<<CurrentPar()<<", stack size="<<parstack.size()<<", popping a single item and pushing 1\n";
        }
    }
    virtual void EndPar(TraversalMode mode, const ASTControlParallelStatement *s)
    {
        if (debug) cerr << Indent(level) << "par, stack size="<<parstack.size()<<", popping "<<s->GetItems().size()<<" items and pushing 1\n";
    }
    virtual void EndSeq(TraversalMode mode, const ASTControlSequentialStatement *s)
    {
        if (debug) cerr << Indent(level) << "seq, stack size="<<parstack.size()<<", popping "<<s->GetItems().size()<<" items and pushing 1\n";
    }
    virtual void EndKernel(TraversalMode mode, const ASTKernel *k)
    {
        if (debug) cerr << Indent(level) << "kernel, stack size="<<parstack.size()<<"\n";
        // nothing to do for a kernel anymore; it now contains a single seq statement
    }

    virtual void Execute(const ASTExecutionBlock *e)
    {
        if (debug) cerr << Indent(level) << "execute, stack size="<<parstack.size()<<", pushing 1 item\n";
        double par = e->GetParallelism()->Expanded(app->paramMap)->Evaluate();
        cerr << Indent(level) << "++ "<<cst->GetStackAsString()<<","<<par*CurrentPar()<<endl;
    }
};

int main(int argc, char **argv)
{
    try {
        if (argc != 2)
        {
            cerr << "Usage: "<<argv[0]<<" [app.aspen]" << endl;
            return 1;
        }

        ASTAppModel *app = LoadAppModel(argv[1]);

        cerr << "Parsed\n";

        ParallelismCounter *t = new ParallelismCounter(app);
        t->SetCacheExecutionBlockExpressions(true);
        t->InitializeTraversal();
        app->kernelMap["main"]->Traverse(t);
        cout << std::setprecision(12) << t->GetResult() << endl;
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }
}

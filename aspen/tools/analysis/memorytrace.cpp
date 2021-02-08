// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>
#include <cfloat>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "app/ASTRequiresStatement.h"
#include "parser/Parser.h"
#include "walkers/AspenTool.h"
#include "walkers/ControlFlowWalker.h"

class CacheFilter
{
    vector<long long> lines;
  public:
    CacheFilter(int LineSizeInBytes, 
                int CacheSizeInBytes,
                int Associativity)
        : linesize(LineSizeInBytes),
          cachesize(CacheSizeInBytes),
          assoc(Associativity)
    {
        nlines = cachesize / linesize;
        setsize = linesize * assoc;
        cachesets = cachesize / setsize;
#if 0
        cout << "linesize = " << setsize << endl;
        cout << "cachesize = " << cachesize << endl;
        cout << "assoc = " << assoc << endl;
        cout << "nlines = " << nlines << endl;
        cout << "setsize = " << setsize << endl;
        cout << "cachesets = " << cachesets << endl;
#endif
        lines.resize(nlines, -1);
    }
    long long Touch(long long addr)
    {
        long long lineaddr = ((long long)(addr / linesize)) * linesize;

        int cacheaddr = addr % cachesize;
        int cacheset = cacheaddr / setsize;

#if 0
        cout << endl;
        cout << "addr: " << addr << endl;
        cout << "lineaddr: " << lineaddr << endl;
        cout << "cacheaddr:" << cacheaddr << endl;
        cout << "cacheset:" << cacheset << endl;
#endif
        int setbase = cacheset * assoc;
        bool found = false;
        int index;
        for (index = 0 ; index<assoc; index++)
        {
            if (lines[setbase + index] == -1)
            {
                break;
            }
            if (lines[setbase + index] == lineaddr)
            {
                found = true;
                break;
            }
        }

        int last = (index == assoc) ? assoc-1 : index;
        for (int j=last; j>0; --j)
        {
            lines[setbase + j] = lines[setbase + j-1];
        }
        lines[setbase + 0] = lineaddr;

        int thruaddr = (found) ? -1 : lineaddr;

#if 0
        cout << "--- LINES ---\n";
        for (unsigned int i=0; i<lines.size(); i++)
        {
            cout << "  " << lines[i] << endl;
        }
        cout << "THRU: " << thruaddr << endl;
        cout << endl;
#endif

        ///\todo: must write back cache lines when flushing them...
        return thruaddr;
    }
  protected:
    int linesize;
    int cachesize;
    int assoc;
    int nlines;
    int setsize;
    int cachesets;
};

class MemoryTracer : public AspenTool
{
    bool debug;
    CacheFilter cache;
  public:
    MemoryTracer(ASTAppModel *app) : AspenTool(app)
                                   , cache(64, 512, 4)
    {
        debug = true;
        cache.Touch(0);
        cache.Touch(128);
        cache.Touch(200);
        cache.Touch(1240);
        cache.Touch(256);
        cache.Touch(128+5);
        cache.Touch(512);
        cache.Touch(128+1);
        cache.Touch(512+256);
    }

  protected:
    virtual TraversalMode TraversalModeHint(const ASTControlStatement *here)
    {
        return Explicit;
    }

    virtual void StartKernel(TraversalMode mode, const ASTKernel *k)
    {
        if (debug) cerr << Indent(level) << "Starting kernel '"<<k->GetName()<<"'\n";
    }

    virtual void StartIterate(TraversalMode mode, const ASTControlIterateStatement *s)
    {
        if (debug) cerr << Indent(level) << "iterate\n";
    }
    
    virtual void StartMap(TraversalMode mode, const ASTControlMapStatement *s)
    {
        if (debug) cerr << Indent(level) << "map\n";
    }

    virtual void Execute(const ASTExecutionBlock *e)
    {
        if (debug) cerr << Indent(level) << "execute\n";
        //double par = e->parallelism ? e->parallelism->Expanded(app->paramMap)->Evaluate() : 1;
        vector<bool> isload;
        vector<string> tofrom;
        vector<double> count;
        vector<double> size;
        vector<string> trait;
        vector<double> traitvalue;
        int maxcount = 0;
        vector<double> localctr;
        vector<double> groupctr;
        vector<double> spanctr;
        vector<double> totalctr;
        vector<double> div;
        vector<double> arraysize;
        for (unsigned int i=0; i<e->GetStatements().size(); ++i)
        {
            const ASTRequiresStatement *rs =
                dynamic_cast<const ASTRequiresStatement*>(e->GetStatements()[i]);
            if (!rs)
                continue;
            if (rs->GetResource() != "loads" && rs->GetResource() != "stores")
                continue;
            if (rs->GetSize() == NULL)
            {
                THROW(ModelError, "count and size are separated in loads/stores "
                      "resource usage for this tool to work.");
            }
            if (rs->GetToFrom() == "")
            {
                THROW(ModelError, "for now, must have a to/from in loads/stores.");
            }
            isload.push_back(rs->GetResource() == "loads");
            tofrom.push_back(rs->GetToFrom());
            arraysize.push_back(app->GetSingleArraySize(tofrom.back())->Expanded(app->paramMap)->Evaluate());
            cout << tofrom.back() << ": size in bytes is " << arraysize.back() << endl;
            int thiscount = rs->GetCount()->Expanded(app->paramMap)->Evaluate();
            count.push_back(thiscount);
            size.push_back(rs->GetSize()->Expanded(app->paramMap)->Evaluate());
            //cerr << isload.back() << tofrom.back() << " " << count.back() << " size " << size.back() << endl;
            if (rs->GetNumTraits() > 0)
            {
                trait.push_back(rs->GetTrait(0)->GetName());
                const Expression *expr = rs->GetTrait(0)->GetValue();
                if (expr)
                    traitvalue.push_back(expr->Expanded(app->paramMap)->Evaluate());
                else
                    traitvalue.push_back(FLT_MAX);
            }
            else
            {
                trait.push_back("");
                traitvalue.push_back(0);
            }

            if (thiscount > maxcount)
                maxcount = thiscount;
            localctr.push_back(0);
            groupctr.push_back(0);
            spanctr.push_back(0);
        }

        double array_chunk = 0x1000000; // use "1" to have arrays be adjacent in memory

        double addressStart = 0;
        map<string, double> ArrayAddress;
        for (unsigned int i=0; i<tofrom.size(); i++)
        {
            if (ArrayAddress.count(tofrom[i]) == 0)
            {
                cout << "array "<<tofrom[i]<<" starts at address " << addressStart << endl;
                ArrayAddress[tofrom[i]] = addressStart;

                addressStart += ((long long)((long long)(arraysize[i] + array_chunk-1) / (long long)(array_chunk)))
                     * array_chunk;
            }
        }

        int n = isload.size();
        for (int j=0; j<n; j++)
            div.push_back(maxcount / count[j]);
        cerr << Indent(level+1) << "Got "<<isload.size()<<" memory ops to transcribe\n";
        cerr << Indent(level+1) << "Max count = " << maxcount << endl;
        for (int i=0; i<maxcount; i++)
        {
            //cout << "\n";
            for (int j=0; j<n; j++)
            {
                spanctr[j]++;
                if (spanctr[j] >= div[j])
                {
                    // do it
                    int index = localctr[j];


#if 1
                    double addr = ArrayAddress[tofrom[j]] + index;

#if 0 // CACHING?
                    long long mainmemoryaddr = cache.Touch((long long)addr);
                    if (mainmemoryaddr >= 0)
                    {
#else
                    long long mainmemoryaddr = (long long)addr;
#endif

                    cout << i*n + j << " ";
                    cout << (isload[j] ? "R " : "W ");
                    char addr_str[256];
                    //sprintf(addr_str, "0x%08x", (unsigned int)(addr));
                    sprintf(addr_str, "0x%08x", (unsigned int)(mainmemoryaddr));
                    cout << addr_str;
                    cout << " ";
                    //cout << size[j];
                    cout << " 0";
                    cout << endl;
#if 0 // CACHING?
                    }
#endif

#else
                    cout << (isload[j] ? "LD(" : "ST(");
                    cout << tofrom[j] <<" + "<<index;
                    cout << ", ";
                    cout << size[j];
                    cout << ")" << endl;
#endif
                    spanctr[j] -= div[j];

                    if (trait[j] == "stride" && traitvalue[j] != FLT_MAX)
                    {
                        localctr[j] += traitvalue[j];
                        if (localctr[j] >= arraysize[j])
                        {
                            groupctr[j]++;
                            localctr[j] = groupctr[j] * size[j];
                            if (localctr[j] >= traitvalue[j] || localctr[j] >= arraysize[j])
                            {
                                groupctr[j] = 0;
                                localctr[j] = 0;
                            }
                        }
                    }
                    else
                    {
                        localctr[j] += size[j];
                        if (localctr[j] >= arraysize[j])
                        {
                            //groupctr[j]++;
                            //localctr[j] = groupctr[j] * size[j];
                            //if (localctr[j] >= arraysize[j])
                                localctr[j] = 0;
                        }
                    }
                }
            }
        }
    }
};


int main(int argc, char **argv)
{
    try {
        if (argc != 2)
        {
            cerr << "Usage: "<<argv[0]<<" [model.aspen]" << endl;
            return 1;
        }

        ASTAppModel *app = LoadAppModel(argv[1]);

        AspenTool *t1 = new MemoryTracer(app);
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

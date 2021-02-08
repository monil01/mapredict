// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/RuntimeCounter.h"
#include "walkers/RuntimeExpression.h"

#define Finegrained_RSC_Print

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
    else if (argc == 3)
    {
        success = LoadAppAndMachineModels(argv[1], argv[2], app, mach);
    }
    else
    {
        cerr << "Usage: "<<argv[0]<<" [model.aspen] [machine.amm]" << endl;
        return 1;
    }

    if (!success)
    {
        cerr << "Errors encountered during parsing.  Aborting.\n";
        return -1;
    }

    cout << "\n------------------- Syntax Trees ---------------------\n\n";
    if (mach)
    {
        cout << "----- Main Machine Model -----\n";
        mach->Print(cout);
    }
    cerr << "\n";
    if (app)
    {
        cout << "----- Main Application Model -----\n";
        app->Print(cout);
    }
    cerr << "\n";
    // For the moment, we're not going to print out the detailed imported models.
    //for (multimap<string,ASTAppModel*>::iterator it = imports.begin(); it != imports.end(); ++it)
    //{
    //    cout << "----- Imported Application Model (within "<<it->first<<") -----\n";
    //    it->second->Print(cout);
    //}
    cout << "\n-----------------------------------------------------\n\n";


    vector<const Identifier*> identifiers;
    vector<double> defvals, minvals, maxvals;
    int nparams = app ? app->FindParametersWithRanges(identifiers,defvals,minvals,maxvals) : 0;
    if (app)
    {
        try
        {
            cout <<"\n ------  Application Analysis ------\n";
            cout << ">> Basic control flow expression\n";
            ExprPtr ctrl(app->GetControlFlowExpression());
            ExprPtr simpctrl(ctrl->Simplified());
            cout << "flops    = " << simpctrl->GetText()  << endl;;
            cout << endl;

            cout << ">> Raw expression without variable expansion:\n";
            {
                string flops = app->GetResourceRequirementExpressionText("flops",    false, false);
                string msgs  = app->GetResourceRequirementExpressionText("messages", false, false);
                string loads = app->GetResourceRequirementExpressionText("loads",    false, false);
                string stores = app->GetResourceRequirementExpressionText("stores",   false, false);
                cout << "flops    = " << flops  << "\n";
                cout << "messages = " << msgs   << "\n";
                cout << "loads    = " << loads  << "\n";
                cout << "stores   = " << stores << "\n";
                cout << endl;
            }

            cout << ">> Raw expression with variable expansion:\n";
            {
                string flops = app->GetResourceRequirementExpressionText("flops",    true, false);
                string msgs  = app->GetResourceRequirementExpressionText("messages", true, false);
                string loads = app->GetResourceRequirementExpressionText("loads",    true, false);
                string stores = app->GetResourceRequirementExpressionText("stores",   true, false);
                cout << "flops    = " << flops  << "\n";
                cout << "messages = " << msgs   << "\n";
                cout << "loads    = " << loads  << "\n";
                cout << "stores   = " << stores << "\n";
            }
            cout << endl;

            cout << ">> as values, With parameters using default values\n";
            {
                double flops  = app->Count("flops");
                double msgs   = app->Count("messages");
                double loads  = app->Count("loads");
                double stores = app->Count("stores");
                cout << "flops    = " << flops  << endl;;
                cout << "messages = " << msgs   << endl;;
                cout << "loads    = " << loads  << endl;;
                cout << "stores   = " << stores << endl;;
            }
            cout << endl;

            cout << ">> Simplification test:\n";
            cout << "flops (noexp)       = " << app->GetResourceRequirementExpressionText("flops", false, false) << "\n";
            cout << "flops (noexp,simpl) = " << app->GetResourceRequirementExpressionText("flops", false, true) << "\n";
            cout << "flops (exp)         = " << app->GetResourceRequirementExpressionText("flops", true, false) << "\n";
            cout << "flops (exp,simpl)   = " << app->GetResourceRequirementExpressionText("flops", true, true) << "\n";

            // parameters
            cout << "\nThere are "<<nparams<<" parameters with ranges.\n";

            for (int i=0; i<nparams; ++i)
            {
                string name = identifiers[i]->GetName();
                cout << ">> with parameter ''"<<name<<"'' set to its minimum of "<<minvals[i]<<":"<<endl;
                cout << "flops    = " << app->Count("flops",    name, minvals[i]) << endl;;
                cout << "messages = " << app->Count("messages", name, minvals[i]) << endl;;
                cout << "loads    = " << app->Count("loads",    name, minvals[i]) << endl;;
                cout << "stores   = " << app->Count("stores",   name, minvals[i]) << endl;;
                cout << endl;

                cout << "-> and now with ''"<<name<<"'' set to its maximum of "<<maxvals[i]<<":"<<endl;
                cout << "flops    = " << app->Count("flops",    name, maxvals[i]) << endl;;
                cout << "messages = " << app->Count("messages", name, maxvals[i]) << endl;;
                cout << "loads    = " << app->Count("loads",    name, maxvals[i]) << endl;;
                cout << "stores   = " << app->Count("stores",   name, maxvals[i]) << endl;;
                cout << endl;

                cout << ">> Expression with parameter ''"<<name<<"'' left as a variable (named x) (and simplified):\n";
                NameMap<const Expression*> subst;
                subst[name] = new Identifier("x");
                cout << "flops = " << app->GetResourceRequirementExpressionText("flops", true, false, subst) << "\n";
                cout << "flops = " << app->GetResourceRequirementExpressionText("flops", true, true, subst) << "\n";
                delete subst[name];
                cout << endl;
            }
            cout << endl;

            for (unsigned int i=0; i<app->GetKernels().size(); ++i)
            {
                try
                {
                    ASTKernel *k = app->GetKernels()[i];
                    cout << "\n\n>> Kernel "<<k->GetName()<<endl<<endl;;

#ifdef Finegrained_RSC_Print
                    ExprPtr flops_integer(k->GetResourceRequirementExpression(app,"flops_integer"));
#endif
                    ExprPtr flops(k->GetResourceRequirementExpression(app,"flops"));
#ifdef Finegrained_RSC_Print
                    ExprPtr flops_simd(k->GetResourceRequirementExpression(app,"flops_simd"));
#endif
                    ExprPtr loads(k->GetResourceRequirementExpression(app,"loads"));
#ifdef Finegrained_RSC_Print
                    ExprPtr loads_random(k->GetResourceRequirementExpression(app,"loads_random"));
#endif
                    ExprPtr stores(k->GetResourceRequirementExpression(app,"stores"));
#ifdef Finegrained_RSC_Print
                    ExprPtr stores_random(k->GetResourceRequirementExpression(app,"stores_random"));
#endif
                    ExprPtr messages(k->GetResourceRequirementExpression(app,"messages"));
                    ExprPtr bytes(k->GetResourceRequirementExpression(app,"bytes"));
#ifdef Finegrained_RSC_Print
                    cout << "Raw integer ops for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(flops_integer->Expanded(app->paramMap))->Evaluate() << endl;
#endif
                    cout << "Raw flops for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(flops->Expanded(app->paramMap))->Evaluate() << endl;
#ifdef Finegrained_RSC_Print
                    cout << "Raw simd flops for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(flops_simd->Expanded(app->paramMap))->Evaluate() << endl;
#endif
                    cout << "Raw loads for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(loads->Expanded(app->paramMap))->Evaluate() << endl;
#ifdef Finegrained_RSC_Print
                    cout << "Raw random loads for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(loads_random->Expanded(app->paramMap))->Evaluate() << endl;
#endif
                    cout << "Raw stores for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(stores->Expanded(app->paramMap))->Evaluate() << endl;
#ifdef Finegrained_RSC_Print
                    cout << "Raw random stores for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(stores_random->Expanded(app->paramMap))->Evaluate() << endl;
#endif
                    cout << "Raw messages for kernel '"<<k->GetName()<<"' = "
                         << ExprPtr(messages->Expanded(app->paramMap))->Evaluate() << endl;
                    cout << endl;

                    
                    cout << "Exclusive set size is " << ExprPtr(k->GetExclusiveDataSizeExpression(app))->GetText() << endl;
                    cout << "Inclusive set size is " << ExprPtr(k->GetInclusiveDataSizeExpression(app))->GetText() << endl;
                    cout << endl;

                    cout << "Calculating flops/byte intensity for kernel '"<<k->GetName()<<"':"<<endl;
                    ExprPtr intensity(new BinaryExpr("/",
                                                              flops->Cloned(),
                                                              bytes->Cloned()));


                    cout << "  = " << intensity->GetText() << endl;

                    // note: we can't simply expand once for the substitution,
                    // then expand again for the kernel parameters, because
                    // the variable to substitute for x may not appear
                    // until we start expanding the other variables.
                    // so take the app's paramMap and override the var in that.
                    ///\todo: to be safe, these kinds of substitutions should
                    /// at least ensure that 'x' doesn't appear as a parameter
                    /// in the app, or else we might get odd results.
                    NameMap<const Expression*> subst = app->paramMap;
                    // nAtom for CoMD, tf for echelon, n for fft
                    subst["nAtom"] = new Identifier("x");

                    cout << "  expanding, but in terms of x:\n";
                    cout << "  = " << ExprPtr(intensity->Expanded(subst))->GetText() << endl;
                    delete subst["nAtom"];
                }
                catch (const AspenException &exc)
                {
                    cerr << exc.PrettyString() <<endl;
                }
            }
        }
        catch (const AspenException &exc)
        {
            cerr << exc.PrettyString() << endl;
        }
    }

    if (mach)
    {
        cout << "\n ------  Machine Analysis ------\n";
        for (unsigned int i=0; i<mach->socketlist.size(); ++i)
        {
          try
          {
            const ASTMachComponent *machine = mach->GetMachine();
            string socket = mach->socketlist[i];
            MappingRestriction restriction("socket", socket);

            cout << "\n\n>> for socket type '"<<socket<<"' <<\n";
            // note: to check for cores, we're counting the number of
            // components which can process the "flops" resource
            ExprPtr tcexpr(machine->GetTotalQuantityExpression("flops",
                                                               restriction));
            cout << "  totalcores = "<<tcexpr->GetText()<<endl;
            cout << "  totalcores = "<<ExprPtr(tcexpr->Expanded(mach->paramMap))->Evaluate()<<endl;
            cout << endl;

            Real *value = new Real(1.e9);

            vector<ASTTrait*> traits;
            cout << "  peak sp gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate() << endl;
            traits.push_back(new ASTTrait("simd"));
            cout << "  peak sp/simd gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate() << endl;
            traits.push_back(new ASTTrait("fmad"));
            cout << "  peak sp/simd/fmad gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate() << endl;
            traits.clear();
            traits.push_back(new ASTTrait("dp"));
            cout << "  peak dp gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate()  << endl;
            traits.push_back(new ASTTrait("simd"));
            cout << "  peak dp/simd gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate()  << endl;
            traits.push_back(new ASTTrait("fmad"));
            cout << "  peak dp/simd/fmad gflops: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->Evaluate()  << endl;

            vector<ASTTrait*> memtraits;
            cout << "  ...\n";
            cout << "  peak bw in GB/sec: " << 1. / ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("loads", memtraits, value, restriction))->Expanded(mach->paramMap))->Evaluate() << endl;

            cout << "\n\n>> testing expressions\n";
            traits.clear();
            cout << "  time to process 1e9 sp flops in sec: " << ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->GetText() << endl;
            cout << "  time to process 1e9 sp flops in sec (expanded): " << ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->GetText() << endl;
            traits.push_back(new ASTTrait("simd"));
            cout << "  time to process 1e9 sp/simd flops in sec: " << ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->GetText() << endl;
            cout << "  time to process 1e9 sp/simd flops in sec (expanded): " << ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("flops", traits, value, restriction))->Expanded(mach->paramMap))->GetText() << endl;
            cout << "  time to read 1e9 bytes in sec: " << ExprPtr(machine->GetIdealizedTimeExpression("loads", memtraits, value, restriction))->GetText() << endl;
            cout << "  time to read 1e9 bytes in sec (expanded): " << ExprPtr(ExprPtr(machine->GetIdealizedTimeExpression("loads", memtraits, value, restriction))->Expanded(mach->paramMap))->GetText() << endl;

            delete value;
          }
          catch (const AspenException& exc)
          {
              cerr << exc.PrettyString() <<endl;
          }
        }
    }

    if (app && mach)
    {
        cout << "\n ------  Combined Analysis ------\n";
        for (unsigned int j=0; j<mach->socketlist.size(); ++j)
        {
            string socket = mach->socketlist[j];
            MappingRestriction restriction("socket", socket);
            for (unsigned int i=0; i<app->GetKernels().size(); ++i)
            {
                try
                {
                    const ASTMachComponent *machine = mach->GetMachine();
                    ASTKernel *k = app->GetKernels()[i];
                    ExprPtr datasize(k->GetInclusiveDataSizeExpression(app));
                    cout << endl << endl << "++Predicting runtime on kernel '"<<k->GetName()<<"' for socket type "<<socket<<endl;

                    RuntimeExpression *re = new RuntimeExpression(app, mach, socket);
                    re->SetCacheExecutionBlockExpressions(false);
                    re->InitializeTraversal();
                    k->Traverse(re);
                    ExprPtr expr(re->GetResult());

                    RuntimeCounter *rc = new RuntimeCounter(app, mach, socket);
                    rc->SetCacheExecutionBlockExpressions(false);
                    rc->InitializeTraversal();
                    k->Traverse(rc);
                    double runtime = rc->GetResult();

                    //ExprPtr expr(k->GetTimeExpression(app, mach, socket));
                    cout << "run time (expression) = " << expr->GetText() << endl;
                    cout << "run time (value)      = " << ExprPtr(ExprPtr(expr->Expanded(app->paramMap))->Expanded(mach->paramMap))->Evaluate() << endl;
                    cout << "run time (value2)     = " << runtime << endl;
                    ExprPtr intracomm(machine->GetIdealizedTimeExpression("intracomm",
                                                                          vector<ASTTrait*>(),
                                                                          datasize.get(),
                                                                          restriction));

                    /*
                    ///\todo: big manual hack to check per-block socket override
                    cerr << "run time base = " << k->GetTimeExpression(mach,socket)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    map<string,string> f1,f2,w1,w2,fb,wb;
                    f1["firstblock"] = "nvidia_m2090";
                    w1["firstblock"] = "westmere";
                    f2["secondblock"] = "nvidia_m2090";
                    w2["secondblock"] = "westmere";
                    fb["firstblock"] = "nvidia_m2090";
                    fb["secondblock"] = "nvidia_m2090";
                    wb["firstblock"] = "westmere";
                    wb["secondblock"] = "westmere";
                    cerr << "run time with firstblock=nvidia_m2090     = " << k->GetTimeExpression(mach,socket,f1)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    cerr << "run time with secondblock=nvidia_m2090    = " << k->GetTimeExpression(mach,socket,f2)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    cerr << "run time with both blocks=nvidia_m2090    = " << k->GetTimeExpression(mach,socket,fb)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    cerr << "run time with firstblock=westmere  = " << k->GetTimeExpression(mach,socket,w1)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    cerr << "run time with secondblock=westmere = " << k->GetTimeExpression(mach,socket,w2)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    cerr << "run time with both blocks=westmere = " << k->GetTimeExpression(mach,socket,wb)->Expanded(app->paramMap)->Expanded(mach->paramMap)->Evaluate()<< endl;
                    */

                    if (intracomm)
                        cout << "app model data transfer time = "<<intracomm->GetText()<<endl;
                    int nsteps = 10;
                    for (int i=0; i<nparams; ++i)
                    {
                        string name = identifiers[i]->GetName();
                        cout << "Scaling over param '"<<name<<"':\n";
                        cout << "value\truntime \tdatatime\tsum\n";
                        double scale = maxvals[i] / minvals[i];
                        for (int step=0; step<nsteps; step++)
                        {
                            double val = minvals[i] * exp(log(scale) * double(step)/double(nsteps-1));
                            NameMap<const Expression*> valmap;
                            valmap[name] = new Real(val);
                            double runtime = ExprPtr(ExprPtr(ExprPtr(expr->Expanded(valmap))->Expanded(app->paramMap))->Expanded(mach->paramMap))->Evaluate();
                            double xfertime = intracomm ? ExprPtr(ExprPtr(ExprPtr(intracomm->Expanded(valmap))->Expanded(app->paramMap))->Expanded(mach->paramMap))->Evaluate() : 0;
                            cout << val<<" \t"<<runtime<<" \t"<<xfertime<<" \t"<<runtime+xfertime<<endl;
                            delete valmap[name];
                        }
                    }

#if 0
                    Expression *flopsDE =
                        k->GetDynamicEnergyExpressionForResource("flops",
                                                                 socket,
                                                                 mach);
                    cout << "flops dynamic power (J) = " << (flopsDE ? flopsDE->GetText() : "(nothing)") << endl;
                    cout << "flops dynamic power (J) = " << (flopsDE ? ExprPtr(flopsDE->Expanded(mach->paramMap))->GetText() : "(nothing)") << endl;
                    Expression *loadsDE =
                        k->GetDynamicEnergyExpressionForResource("loads",
                                                                 socket,
                                                                 mach);
                    cout << "loads dynamic power (J) = " << (loadsDE ? loadsDE->GetText() : "(nothing)")  << endl;
                    cout << "loads dynamic power (J) = " << (loadsDE ? ExprPtr(loadsDE->Expanded(mach->paramMap))->GetText() : "(nothing)")  << endl;
                    Expression *storesDE =
                        k->GetDynamicEnergyExpressionForResource("stores",
                                                                 socket,
                                                                 mach);
                    cout << "stores dynamic power (J) = " << (storesDE ? storesDE->GetText() : "(nothing)")  << endl;
                    cout << "stores dynamic power (J) = " << (storesDE ? ExprPtr(storesDE->Expanded(mach->paramMap))->GetText() : "(nothing)")  << endl;

                    if (flopsDE)
                        delete flopsDE;
                    if (loadsDE)
                        delete loadsDE;
                    if (storesDE)
                        delete storesDE;
#endif
                    delete rc;
                    delete re;

                }
                catch (const AspenException &exc)
                {
                    cerr << exc.PrettyString() <<endl;
                }
            }
        }
    }

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

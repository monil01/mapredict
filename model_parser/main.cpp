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

#include "model/ASTDataStatement.h"

#include "app/ASTExecutionBlock.h"
#include "model/ASTMachModel.h"
#include "app/ASTRequiresStatement.h"
#include "walkers/AspenTool.h"
#include "traverser.h"



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

    //cout << "\n------------------- Syntax Trees ---------------------\n\n";
    
    if (app)
    {
        //cout << "----- Main Application Model -----\n";
        app->Print(cout);
    }
    cerr << "\n";

    if (mach)
    {
        //cout << "----- Main Machine Model -----\n";
        //mach->Print(cout);
    }
    cerr << "\n";

    traverser* traverser_obj = new traverser(app, mach);

    cout << "\n-----------------------------------------------------\n\n";
    

    if (mach)
    {
        cout << "\n ------  Memory Analysis ------\n";
        for (unsigned int i=0; i<mach->socketlist.size(); ++i)
        {
          try
          {
            const ASTMachComponent *machine = mach->GetMachine();
            string socket = mach->socketlist[i];

            cout << "\n\n>> for socket type '"<<socket<<"' <<\n\n";

            double memory = traverser_obj->predictMemoryAccess(app, mach, socket); 
            //std::cout << " for Double data type : " << memory * 2 << "\n";
            //std::cout << " Total bytes accessed : " << memory << "\n";
            //std::cout << " for float data type : " << memory << "\n";
          }
          catch (const AspenException &exc)
          {
            cerr << exc.PrettyString() <<endl;
          } 
        }
    }
    



    if (app)
        delete app;
    if (mach)
        delete mach;
    if (traverser_obj)
        delete traverser_obj;



    exit (0);


 
    /* std::string size_name;

    double element_size = 0, stride = 0, total_length = 0;
    std::string param = "aspen_param_sizeof_float";

    element_size = get_application_param(app, param);
    std::cout << " " << param << " : " << element_size << std::endl;

    for (unsigned int i=0; i<1; ++i)
    //for (unsigned int i=0; i<app->GetKernels().size(); ++i)
    {
       try
       {
           //string flops = app->GetResourceRequirementExpressionText("flops",    false, false);
           std::cout << app->GetResourceRequirementExpression("loads")->GetText() << std::endl;

           ASTKernel *k = app->GetKernels()[i];
           cout << "\n\n>> Kernel "<<k->GetName()<<endl<<endl;
           const ASTControlSequentialStatement *statements = k->GetStatements();
           std::cout << "size " << statements->GetItems().size() << std::endl;
           //const ASTExecutionBlock *exec = dynamic_cast<const ASTExecutionBlock*>(s);
           //const ASTControlStatement *ctrl = dynamic_cast<const ASTControlStatement*>(s);
           for (unsigned int i=0; i < statements->GetItems().size(); ++i)
           {
              const ASTControlStatement *s = statements->GetItems()[i];
              const ASTExecutionBlock *exec = dynamic_cast<const ASTExecutionBlock*>(s);
              const ASTControlStatement *ctrl = dynamic_cast<const ASTControlStatement*>(s);
              ExpressionBuilder eb;
              Expression* current_expression;

              if (exec) // it's a kernel-like requires statement
              {
                  ///\todo: now it's the same code as ctrl
                  //eb += exec->GetResourceRequirementExpression(app,resource);
                  std::string resource;
		  resource = "loads"; stride = 0; total_length = 0;
                  //std::cout << exec->GetResourceRequirementExpression(app,resource)->GetText();
                  const vector<ASTExecutionStatement*> statements_exec = exec->GetStatements();
                  for (unsigned int i=0; i<statements_exec.size(); ++i)
                  {
                     const ASTExecutionStatement *s = statements_exec[i];
                     const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
                     if (req) // requires statement
                     {
                       if (req->GetResource() == "loads") 
                       {                             
                          //Expression *expr = NULL;
                          //expr = req->GetQuantity()->Cloned();
                          eb = req->GetQuantity()->Cloned();
			  const vector<ASTTrait*> traits = req->GetTraits();
			  for (int i = 0; i < traits.size(); i++){
	                        string ttrait = traits[i]->GetName();
                                if (ttrait == "stride") {

                                   std::cout << "traits Name " << ttrait;
                                   std::cout << " traits value " << traits[i]->GetValue()->Evaluate() << std::endl;
				   stride = traits[i]->GetValue()->Evaluate();

                                }
			  }
                          
                          
			  //eb + = ;
                          
                          std::cout << "loads " << eb.GetExpression()->GetText() << std::endl;
			  double temp_total = eb.GetExpression()->Expanded(app->paramMap)->Evaluate();
                          //std::cout << "loads " << eb.GetExpression()->Expanded(app->paramMap)->Evaluate() << std::endl;
                          total_length = temp_total / element_size;
                          std::cout << "total length " << total_length << " element size " << element_size << " stride " << stride << std::endl;
                          

                       }
                       else if (req->GetResource() == "stores") 
		       {
                          //Expression *expr = NULL;
                          //expr = req->GetQuantity()->Cloned();
                          eb = req->GetQuantity()->Cloned();
			  const vector<ASTTrait*> traits = req->GetTraits();
			  for (int i = 0; i < traits.size(); i++){
	                        string ttrait = traits[i]->GetName();
                                if (ttrait == "stride") {

                                   std::cout << "traits Name " << ttrait;
                                   std::cout << " traits value " << traits[i]->GetValue()->Evaluate() << std::endl;
				   stride = traits[i]->GetValue()->Evaluate();
                                }
			  }
                          

                          std::cout << "stores " << eb.GetExpression()->GetText() << std::endl;
        		  double temp_total = eb.GetExpression()->Expanded(app->paramMap)->Evaluate();
                          //std::cout << "stores " << eb.GetExpression()->Expanded(app->paramMap)->Evaluate() << std::endl;
                          total_length = temp_total / element_size;
                          std::cout << "total length " << total_length << " element size " << element_size << " stride " << stride << std::endl;

                       }
                     }
		  }


              }
           } 


       }
       catch (const AspenException &exc)
       {
           cerr << exc.PrettyString() <<endl;
       }
    }

    double cacheline = 0;


    //std::string property = "cacheline";
    std::string property = "bandwidth";
    std::string component = "cache";
    std::string socket = "SimpleCPU";
    //std::string socket = "nvidia_k80";
    cacheline = get_any_machine_property(mach, socket, component, property);
    std::cout << socket << " : " << component << " : " << property << " " << cacheline << std::endl; 

*/



/*

    if (app)
    {
        try
        {
            cout <<"\n ------  Application Analysis ------\n";
            cout << ">> Basic control flow expression\n";
            ExprPtr ctrl(app->GetControlFlowExpression());
            ExprPtr simpctrl(ctrl->Simplified());


            //ExprPtr ctrl(app->GetControlFlowExpression());
            //ExprPtr simpctrl(ctrl->Simplified());
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

*/

    if (mach)
    {
        cout << "\n ------  Machine Analysis ------\n";
        for (unsigned int i=1; i<mach->socketlist.size(); ++i)
        //for (unsigned int i=0; i<mach->socketlist.size(); ++i)
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

    exit (0);
#if 0
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
#endif

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

/*

Expression*
GetExpression(ASTAppModel *app,
                                     ASTMachModel *mach,
                                     string sockettype) const
{
    if (statements.empty())
        return new Real(0);
    //std::cout << " this from the time expression" << std::endl;

    MappingRestriction restr("socket", sockettype);

    /// ---------
    /// Experimental (new): Group resource usage (and limit it appropriately) (plus some "smart" things like assuming stride-0 expressions are constant, etc.)
    /// ---------
#if 0


    map<resgrouper, Expression*> resource_usage;
    for (unsigned int i=0; i<statements.size(); ++i)
    {
        const ASTExecutionStatement *s = statements[i];
        const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
        if (!req)
            continue;
        resgrouper rg;
        rg.resource = req->GetResource();
        rg.tofrom   = req->GetToFrom();

        //// hack, skip intracomm!
        //if (rg.resource == "intracomm")
        //{
        //    resource_usage[rg] = new Real(0);
        //    continue;
        //}

        bool skip = false;
        for (unsigned t = 0; t < req->GetNumTraits(); ++t)
        {
            if (req->GetTrait(t)->GetName() == "stride")
            {
                try
                {
                    double stride = req->GetTrait(t)->GetValue()->Evaluate();
                    if (stride == 0)
                        skip = true; // skip this resource requirement entirely; a stride of 0 means assume register usage
                    else if (stride <= 1)
                        continue; // skip this trait; we ignore the value for traits, but removing the stride trait means it's assumed to be continuous
                }
                catch (...)
                {
                    // ignore errors from evaluating the trait quantity
                }
            }
            rg.traits.push_back(req->GetTrait(t)->GetName());
        }
        if (skip)
            continue;
        if (resource_usage.count(rg))
            resource_usage[rg] = new BinaryExpr("+", resource_usage[rg], req->GetQuantity()->Cloned());
        else
            resource_usage[rg] = req->GetQuantity()->Cloned();
    }

#if 1
    // clamp every let of loads/stores to max total size of a given array
    ///\todo: even this modification should probably be limited to stride-0 accesses
    for (map<resgrouper, Expression*>::iterator it = resource_usage.begin();
         it != resource_usage.end(); ++it)
    {
        const resgrouper &rg = it->first;
        if (rg.tofrom != "")
        {
            Expression *size = app->GetSingleArraySize(rg.tofrom);
            if (size)
            {
                // simple option ignored parallelism
                //it->second = new FunctionCall("min", it->second, size);

                // complex option; only read a total of 'size' bytes across ALL parallel instances
                const Expression *par = GetParallelism();
                it->second = new FunctionCall("min", it->second, new BinaryExpr("/",size,par->Cloned()));
            }
        }
    }
#endif

    // DEBUG
#if 1
    for (map<resgrouper, Expression*>::iterator it = resource_usage.begin();
         it != resource_usage.end(); ++it)
    {
        const resgrouper &rg = it->first;
        cerr << "rg=" << rg.resource << " tofrom=" << rg.tofrom << " traits=";
        for (unsigned int i=0; i<rg.traits.size(); ++i)
            cerr << rg.traits[i] << ",";
        cerr << " ::: " << it->second->GetText() << endl;
    }
    cerr << endl;
#endif

    // calculate times for each requirement and add them up
    map<string,Expression*> requirements;
    for (map<resgrouper, Expression*>::iterator it = resource_usage.begin();
         it != resource_usage.end(); ++it)
    {
        const resgrouper &rg = it->first;
        const Expression *quantity = it->second;

        // get the "serial" time expression for this resource
        Expression *sertime = mach->GetMachine()->
            GetSerialTimeExpression(rg.resource, rg.traits, quantity, restr);
        Expression *quant = mach->GetMachine()->
            GetTotalQuantityExpression(rg.resource, restr);
        Expression *depth = new FunctionCall("ceil",
                                 new BinaryExpr("/", parallelism, quant));
        Expression *time = new BinaryExpr("*", depth, sertime);
        //cerr << "   ---- time for resource '"<<resource<<"': "<<time->GetText() << endl;
        if (requirements[rg.resource])
            requirements[rg.resource] = new BinaryExpr("+",requirements[rg.resource],time);
        else
            requirements[rg.resource] = time;
    }


#else

    /// ---------
    /// Standard (old): Calculate runtime for each resource statement independently
    /// ---------


    // Get the raw requirements for each type of resource
    map<string,Expression*> requirements;
    for (unsigned int i=0; i<statements.size(); ++i)
    {
        const ASTExecutionStatement *s = statements[i];
        const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
        if (!req)
            continue;

        // get the resource and traits
        string resource = req->GetResource();
        const vector<ASTTrait*> &traits = req->GetTraits();

        // get the "serial" time expression for this resource
        Expression *sertime = mach->GetMachine()->
            GetSerialTimeExpression(resource, traits, req->GetQuantity(), restr);
        Expression *quant = mach->GetMachine()->
            GetTotalQuantityExpression(resource, restr);
        Expression *depth = new FunctionCall("ceil",
                            new BinaryExpr("/", parallelism->Cloned(), quant));
        Expression *time = new BinaryExpr("*", depth, sertime);
            
        if (!time)
            cerr << "ERROR getting time for '"<<resource<<"' with quantity "<<req->GetQuantity()->GetText()<<endl;
        else
        {
            //cerr << "   ---- time for resource '"<<resource<<"': "<<time->GetText() << endl;
            if (requirements[resource])
                requirements[resource] = new BinaryExpr("+",requirements[resource],time);
            else
                requirements[resource] = time;
        }
    }
#endif

    // Fold any that conflict using "+"
    map<set<string>, Expression*> conflicting;
    bool foundnewconflict = true;
    while (foundnewconflict)
    {
        foundnewconflict = false;

        // first, look for conflicts with an already-folded resource
        //cerr << "looking for conflicts with an already-folded resource\n";
        for (map<string,Expression*>::iterator req = requirements.begin(); req != requirements.end(); req++)
        {
            string reqres = req->first;
            for (map<set<string>,Expression*>::iterator conf = conflicting.begin(); conf != conflicting.end(); conf++)
            {
                for (set<string>::iterator confitem = conf->first.begin(); confitem != conf->first.end(); confitem++)
                {
                    if (mach->GetMachine()->CheckConflict(reqres, *confitem, restr))
                    {
                        //cerr << "   -- Found conflict between so-far-not-conflicting-item "<<reqres<<" and already-conflicting-item "<<*confitem<<endl;
                        foundnewconflict = true;
                        break;
                    }
                }
                if (foundnewconflict)
                {
                    set<string> newset(conf->first);
                    newset.insert(reqres);
                    conflicting[newset] = new BinaryExpr("+", conf->second, req->second);
                    conflicting.erase(conf);
                    requirements.erase(req);
                }
                if (foundnewconflict)
                    break;
            }
            if (foundnewconflict)
                break;
        }

        // next, look for conflicts between iterms that have not yet been folded
        //cerr << "no conflicts with existing conflicting items; looking at unfolded pairs\n";
        for (map<string,Expression*>::iterator req1 = requirements.begin(); req1 != requirements.end(); req1++)
        {
            string reqres1 = req1->first;
            //cerr << " (1) reqres1 = " << reqres1 << endl;
            for (map<string,Expression*>::iterator req2 = requirements.begin(); req2 != requirements.end(); req2++)
            {
                string reqres2 = req2->first;
                //cerr << "    (2) reqres2 = " << reqres2 << endl;
                if (reqres1 != reqres2)
                {
                    if (mach->GetMachine()->CheckConflict(reqres1, reqres2, restr))
                    {
                        //cerr << "   -- Found conflict between so-far-not-conflicting-items "<<reqres1<<" and "<<reqres2<<endl;
                        set<string> newset;
                        newset.insert(reqres1);
                        newset.insert(reqres2);
                        conflicting[newset] = new BinaryExpr("+", req1->second, req2->second);
                        foundnewconflict = true;
                        requirements.erase(req1);
                        requirements.erase(req2);
                    }
                }
                if (foundnewconflict)
                    break;
            }
            if (foundnewconflict)
                break;
        }

        //cerr << "foundnewconflict="<<foundnewconflict<<endl;
    }

    // We should have found all conflicting requirements
    // Whatever's left (from the individual or previously-folded
    // requirements) can be folded using Max()
    Expression *runtime = NULL;
    for (map<string,Expression*>::iterator req = requirements.begin(); req != requirements.end(); req++)
    {
        //cerr << "for "<<req->first<<", serial runtime expr = "<<req->second->GetText()<<endl;
        if (!runtime)
            runtime = req->second;
        else
            runtime = new FunctionCall("max", runtime, req->second);
    }
    for (map<set<string>,Expression*>::iterator conf = conflicting.begin(); conf != conflicting.end(); conf++)
    {
        //cerr << "for (conflicting set) serial runtime expr = "<<conf->second->GetText()<<endl;
        if (!runtime)
            runtime = conf->second;
        else
            runtime = new FunctionCall("max", runtime, conf->second);
    }
    if (!runtime)
        THROW(ModelError, "Couldn't get serial time");
    return runtime;
}

*/




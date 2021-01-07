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



#define Finegrained_RSC_Print

using namespace std;


/*
 * All functions are declared below
 */

double  predict_memory_streaming_access(ASTAppModel *app, 
                                        ASTMachModel *mach, std::string socket);

double  get_application_param(ASTAppModel *app, std::string param);

const ASTMachComponent* 
get_socket_component(ASTMachModel *mach, std::string socket);

double 
get_any_machine_property(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property);

double  
analytical_streaming_access(double D, double E, double S, double CL);

double 
find_mod(double a, double b);

double
execute_block(ASTAppModel *app, ASTMachModel *mach, std::string socket,
    const ASTExecutionBlock *exec,
    double outer_paralellism);

/*double
execute_block(ASTAppModel *app, ASTMachModel *mach, std::string socket,
    const ASTExecutionBlock *exec,
    double outer_paralellism); */

/*
 * functions declarations end here 
 */

/*
 * Function name: find_mod 
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function finds the mod two double values 
 * 
 * Input: This function takes two arguments 
 * 
 * Output: mod in floating/double 
 * 
 * Description: it just loops through to get the mod 
 */

double 
find_mod(double a, double b) 
{ 
    double mod; 
    // Handling negative values 
    if (a < 0 || b < 0)
    { 
	std::cout << " ERROR: a or b is negative a:" << a << " b:" << b << "\n";
	return 0;
    }
  
    // Finding mod by repeated subtraction 
    while (mod >= b) 
        mod = mod - b; 
  
    return mod; 
} 

/*
 * Function name: analytical_streaming_access
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function calculates the analytical streaming access 
 * 
 * Input: This function takes four arguments, 
 * 1.  D = total length of the data structure
 * 2.  E = Element size (i.e., size of float/double/int)
 * 3.  S = Stride length
 * 4.  CL = cacheline length.
 * 
 * Output: memory prediction for the given inputs 
 * 
 * Description: This function calculated the memory traffic as per the given input
 * which are derived from from load/store execution statements of kernel of an application
 * model. . 
 * The analytical model for streaming access pattern is inspired from the paper SC14 
 * paper: "Quantitatively Modeling Application Resilience with the Data Vulnerability 
 * Factor" Link: https://ieeexplore.ieee.org/document/7013044 
 */


double  
analytical_random_access(double D, double E, double S, double CL)
{
    double memory_access = 0;
    // converting the stride and data size to bytes
    S = S * E;
    D = D * E;

    int cli = (int) CL;
    int e = (int) E;

    int mod = (e - 1) % cli;
    double p = mod / CL;
    //double p = find_mod((E - 1), CL) / CL;

    // First Case E >= CL 
    if (E >= CL)
    {
        if ( E < S) 
        {
	    double AE = ceil(E/CL) + p;
            memory_access = floor(D/S) * AE;
        }
        else 
        {
	    memory_access = ceil(D/CL);
        }

    }
    
    // Second Case E < CL <= S
    if (E < CL && CL <= S)
    {
        memory_access = floor(D/S) * (1 + p);
        std::cout << memory_access << " " <<  S  << " " << D  << " " << p << std::endl;
    }

    // for the third case S < CL
    if ( S < CL) 
    {
	memory_access = ceil(D/CL);
    }

    //std::cout << " memory access" << memory_access << "\n"; 

    // memory access is multiplied by CL to convert it to bytes
    return memory_access * CL;
}


double  
analytical_stencil_access(double D, double E, double S, double CL)
{
    double memory_access = 0;
    // converting the stride and data size to bytes
    S = S * E;
    D = D * E;

    int cli = (int) CL;
    int e = (int) E;

    int mod = (e - 1) % cli;
    double p = mod / CL;
    //double p = find_mod((E - 1), CL) / CL;

    // First Case E >= CL 
    if (E >= CL)
    {
        if ( E < S) 
        {
	    double AE = ceil(E/CL) + p;
            memory_access = floor(D/S) * AE;
        }
        else 
        {
	    memory_access = ceil(D/CL);
        }

    }
    
    // Second Case E < CL <= S
    if (E < CL && CL <= S)
    {
        memory_access = floor(D/S) * (1 + p);
        std::cout << memory_access << " " <<  S  << " " << D  << " " << p << std::endl;
    }

    // for the third case S < CL
    if ( S < CL) 
    {
	memory_access = ceil(D/CL);
    }

    //std::cout << " memory access" << memory_access << "\n"; 

    // memory access is multiplied by CL to convert it to bytes
    return memory_access * CL;
}



double  
analytical_streaming_access(double D, double E, double S, double CL)
{
    double memory_access = 0;
    // converting the stride and data size to bytes
    S = S * E;
    D = D * E;

    int cli = (int) CL;
    int e = (int) E;

    int mod = (e - 1) % cli;
    double p = mod / CL;
    //double p = find_mod((E - 1), CL) / CL;

    // First Case E >= CL 
    if (E >= CL)
    {
        if ( E < S) 
        {
	    double AE = ceil(E/CL) + p;
            memory_access = floor(D/S) * AE;
        }
        else 
        {
	    memory_access = ceil(D/CL);
        }

    }
    
    // Second Case E < CL <= S
    if (E < CL && CL <= S)
    {
        memory_access = floor(D/S) * (1 + p);
        std::cout << memory_access << " " <<  S  << " " << D  << " " << p << std::endl;
    }

    // for the third case S < CL
    if ( S < CL) 
    {
	memory_access = ceil(D/CL);
    }

    //std::cout << " memory access" << memory_access << "\n"; 

    // memory access is multiplied by CL to convert it to bytes
    return memory_access * CL;
}


 
double  
execute_block(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    const ASTExecutionBlock *exec,
    double outer_paralellism)
{  
    double total_memory_access = 0;

    double element_size = 0, stride = 0, total_length = 0;
    double cacheline = 0;
    std::string param = "aspen_param_sizeof_";

    element_size = get_application_param(app, param);
    //std::cout << " " << param << " : " << element_size << std::endl;

    //std::string property = "cacheline";
    std::string property = "cacheline";
    std::string component = "cache";
    //std::string socket = "nvidia_k80";
    //socket = "intel_xeon_x5660";

    cacheline = get_any_machine_property(mach, socket, component, property);
    std::cout << " " << socket << " : " << component << " : " << property << " " << cacheline << std::endl; 

    ExpressionBuilder eb;
    Expression* current_expression;
    double paralellism = 1;
    std::cout << " current multiplying factor : " << 
        exec->GetParallelism()->Expanded(app->paramMap)->Evaluate() << "\n";
    paralellism = exec->GetParallelism()->Expanded(app->paramMap)->Evaluate();
    paralellism *= outer_paralellism;
    std::cout << " multiplied factor : " << paralellism << "\n";
    //std::string resource;
	//resource = "loads"; stride = 0; total_length = 0;
    //std::cout << exec->GetResourceRequirementExpression(app,resource)->GetText();
    const vector<ASTExecutionStatement*> statements_exec = exec->GetStatements(); 

     
    for (unsigned int j = 0; j < statements_exec.size(); ++j)
    {
        const ASTExecutionStatement *s = statements_exec[j];
        const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
        if (req) // requires statement
        {
            if (req->GetResource() == "loads") 
            {  
                //Expression *expr = NULL;
                //expr = req->GetQuantity()->Cloned();
                eb = req->GetQuantity()->Cloned();
			    const vector<ASTTrait*> traits = req->GetTraits();
			    for (int k = 0; k < traits.size(); k++)
                {
	                string ttrait = traits[k]->GetName();
                    if (ttrait == "stride") 
                    {
                        //std::cout << "traits Name " << ttrait;
                        //std::cout << " traits value " << 
                        //	traits[k]->GetValue()->Evaluate() << std::endl;
				        stride = traits[k]->GetValue()->Evaluate();
				        //stride = stride * element_size;
                    }
			    }
                std::cout << " Loads : " << eb.GetExpression()->GetText() << std::endl;
			    double temp_total = eb.GetExpression()->Expanded(app->paramMap)->Evaluate();
                total_length = temp_total / element_size;
                std::cout << " Total length=" << total_length << " element size=" 
				    << element_size << " stride=" << stride 
				    << " cacheline=" << cacheline << "\n\n";
			    double memory_access = 
				    analytical_streaming_access(total_length, element_size, stride,
				    cacheline);
                memory_access = memory_access * paralellism; 
        		total_memory_access += memory_access;
			    std::cout << " memory access : " << memory_access << "\n \n";
                          
            }
            else if (req->GetResource() == "stores") 
		    {
                //Expression *expr = NULL;
                //expr = req->GetQuantity()->Cloned();
                eb = req->GetQuantity()->Cloned();
			    const vector<ASTTrait*> traits = req->GetTraits();
			    for (int k = 0; k < traits.size(); k++)
                {
	                string ttrait = traits[k]->GetName();
                    if (ttrait == "stride") 
                    {

                        //std::cout << "traits Name " << ttrait;
                        //std::cout << " traits value " << 
                        //    traits[k]->GetValue()->Evaluate() << std::endl;
				        stride = traits[k]->GetValue()->Evaluate();
				        //stride = stride * element_size;
                    }
			    }
                std::cout << " stores : " << eb.GetExpression()->GetText() << std::endl;
        		double temp_total = eb.GetExpression()->Expanded(app->paramMap)->Evaluate();
                total_length = temp_total / element_size;
			    std::cout << " Total length=" << total_length << " element size=" 
				     << element_size << " stride=" << stride 
				     << " cacheline=" << cacheline << "\n\n";
        		double memory_access = 
				              analytical_streaming_access(total_length, element_size, stride,
					          cacheline);
                memory_access = memory_access * paralellism; 
			    total_memory_access += memory_access;
			    std::cout << " memory access : " << memory_access << "\n \n";
                      
            } 
        }
    } 

    return total_memory_access;

}
    
double  
recursive_block(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    double outer_parallelism,
    const ASTControlStatement *s)
{
    double total_memory_access = 0;
    const ASTExecutionBlock *exec = dynamic_cast<const ASTExecutionBlock*>(s);
    if (exec) // it's a kernel-like requires statement
    {
        std::cout << " Execute block\n";
        total_memory_access += execute_block(app, mach, socket, exec, outer_parallelism);
        std::cout << "\n total memory upto this block: " << total_memory_access << "\n";
        return total_memory_access;
    }
    
    const ASTControlIterateStatement *iter_statement = 
        dynamic_cast<const ASTControlIterateStatement*>(s);

    //const ASTControlSequentialStatement *statements = 
                    //dynamic_cast<const ASTControlSequentialStatement *>(s);
    if (iter_statement){
        std::cout << " Iter block\n";
        double new_parallelism = iter_statement->GetQuantity()->Expanded(app->paramMap)->Evaluate();

        outer_parallelism *= new_parallelism;

        std::cout << "\n from iter: new factor: " << new_parallelism << "\n";
        ASTControlStatement * s_new = iter_statement->GetItem();

        total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s_new);
        return total_memory_access;

    }
  
 
    const ASTControlMapStatement *map_statement = 
        dynamic_cast<const ASTControlMapStatement*>(s);

    //const ASTControlSequentialStatement *statements = 
                    //dynamic_cast<const ASTControlSequentialStatement *>(s);
    if (map_statement){
        std::cout << " Map block\n";
        double new_parallelism = map_statement->GetQuantity()->Expanded(app->paramMap)->Evaluate();

        outer_parallelism *= new_parallelism;

        std::cout << "\n from map: new factor: " << new_parallelism << "\n";
        ASTControlStatement * s_new = map_statement->GetItem();

        total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s_new);
        return total_memory_access;

    }
 
    const ASTControlSequentialStatement *seq_statements 
                = dynamic_cast<const ASTControlSequentialStatement*>(s);

    if (seq_statements){
        std::cout << " Seq block\n";
        std::cout << " size " << seq_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < seq_statements->GetItems().size(); ++i)
        {
            //double outer_paralellism = 1;

            const ASTControlStatement *s_new = seq_statements->GetItems()[i];
            total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s_new);
        }
    } 

 
    const ASTControlParallelStatement *par_statements 
                = dynamic_cast<const ASTControlParallelStatement*>(s);

    if (par_statements){
        std::cout << " Par block\n";
        std::cout << " size " << par_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < par_statements->GetItems().size(); ++i)
        {
            //double outer_paralellism = 1;

            const ASTControlStatement *s_new = par_statements->GetItems()[i];
            total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s_new);
        }
    } 

    const ASTControlKernelCallStatement *call_statements 
                = dynamic_cast<const ASTControlKernelCallStatement*>(s);

    if (call_statements){
        std::cout << " Call block\n";
        const ASTKernel *k = call_statements->GetKernel(app);
        std::cout << " Kernel name: " << k->GetName() << "\n";
        const ASTControlSequentialStatement *seq_statements = k->GetStatements();
 
        const ASTControlStatement *s_new
            = dynamic_cast<const ASTControlStatement*>(seq_statements);

        total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s_new);

    } 


    return total_memory_access;
}

/*
 * Function name: predict_memory_streaming_access
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function predicts the memory traffic for streaming access pattern 
 * 
 * Input: This function takes two arguments, 1. application model, 2. Machine Model
 * 
 * Output: Total memory traffic prediction 
 * 
 * Description: This function finds the kernels and it's load and store commands. 
 * Then for every load store command, it finds the stride traits. with the stride 
 * tratis, and cacheline from machine model, it calls the analytical model function 
 * to calculte the total memory traffic.
 */

double  
predict_memory_streaming_access(ASTAppModel *app, ASTMachModel *mach, std::string socket)
{
    double total_memory_access = 0;

    //std::string size_name;

    for (unsigned int i=0; i<app->GetKernels().size(); ++i)
    {
        try
        {
            ASTKernel *k = app->GetKernels()[i];
            if(k->GetName() == "main")
            {
                const ASTControlSequentialStatement *statements = k->GetStatements();
                double outer_parallelism = 1;
 
                const ASTControlStatement *s 
                    = dynamic_cast<const ASTControlStatement*>(statements);

                total_memory_access += recursive_block(app, mach, socket, outer_parallelism, s);

            }
        
        }
        catch (const AspenException &exc)
        {
            cerr << exc.PrettyString() <<endl;
        }
    }

    //std::cout << " Total memory access: " << total_memory_access << "\n"; 
    return total_memory_access;

} 


/*
 * Function name: get_application_param
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function finds a global param from aplication model 
 * 
 * Input: This function takes four arguments, 1. application model, 2. param name,
 * 
 * Output: If found, value of the param is returned, if not -1 is returned.
 * 
 * Description: This function fetches all the global statements of an application 
 * model and then dynamic cast the statement ot assign statments. assign statements
 * has a name and a value. This value is the value of the parameter. When evaluated
 * the value is found and returned. if not  -1 is returned. 
 *
 * Note: when the param is not found, -1 may not be a good idea. need to improve in 
 * 	future.
 * Note2: partial matching is supported. here.
 * Improvement: remove the partial matching. Ok for now. 
 */


double  
get_application_param(ASTAppModel *app, std::string param)
{
    double param_value = -1;
    if (app)
    {
        //cout << "\n ------  Application model search:param search function called ------\n";
        try
        {
            vector<ASTStatement*> globals = app->GetGlobals();

            //std::cout <<  "Size globals" << globals.size() << std::endl;

            for (unsigned int i=0; i<globals.size(); ++i)
            {
                //const ASTDataStatement *data = dynamic_cast<const ASTDataStatement*>(globals[i]);
                const ASTAssignStatement *data = dynamic_cast<const ASTAssignStatement*>(globals[i]);
                if (!data)
                    continue;
                std::string temp = data->GetName();
                //std::cout << "Identifier name " << data->GetName() << std::endl;
                //std::cout << "Identifier Value " << data->GetValue()->Evaluate() << std::endl;
                if (temp.find(param) != std::string::npos) {
                    param_value = data->GetValue()->Evaluate();
                    //std::cout << "param " << temp << " : " << param_value << std::endl;
                    return param_value; 
                }
            }  

        }
        catch (const AspenException& exc)
        {
            cerr << exc.PrettyString() <<endl;
        }
    }
    std::cout << " Param: "  << param << " not found" << std::endl;
    return param_value; 
} 



/*
 * Function name: get_socket_component
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function finds socket component from a machine model 
 * 
 * Input: This function takes four arguments, 1. machine model, 2. socket name,
 * 
 * Output: If found, socket component is returned, if not NULL is returned.
 * 
 * Description: This function descends down the machine tree, general rule is 
 * it decomposes machine into components and then to subcomponent. 
 * every subcomponent can be brought back to component again.
 * Even though, one would think a sub component might have the property
 * but its  the component that has the list of properties.
 * This function travarses through the hierarchy of the
 * model and finds the component and returns that.
 *
 * Note1: We added some function in the mach/ASTMachComponent.h files.
 */


const ASTMachComponent* 
get_socket_component(ASTMachModel *mach, std::string socket)
{

    if (mach)
    {
        //cout << "\n ------  Machine model search:component function called ------\n";
        //for (unsigned int i=0; i<mach->socketlist.size(); ++i)
        {
          try
          {
            const ASTMachComponent *machine = mach->GetMachine();
            //cout << "  Name = " << machine->GetName() << endl;
            //cout << " Type = " << machine->GetType() << endl;
            const vector<const ASTSubComponent*> subcomponents = machine->GetSubComponent();
	    for (unsigned int i = 0; i < subcomponents.size(); ++i)
    	    {
                const ASTSubComponent *sc = subcomponents[i];
                const NameMap<const ASTMachComponent*> subcomponentmap = machine->GetSubComponentMap();
                const ASTMachComponent *comp = subcomponentmap[sc->GetName()];
                

                //cout << " Name = " << comp->GetName() << endl;
                //cout << " Type = " << comp->GetType() << endl;
		if (comp->GetName() == socket){
		    return comp;
		}
 
                const vector<const ASTSubComponent*> newsub = comp->GetSubComponent();
	        for (unsigned int j = 0; j < newsub.size(); ++j)
    	        {
                    const ASTSubComponent *newsc = newsub[j];
                    const NameMap<const ASTMachComponent*> newsubcomponentmap = comp->GetSubComponentMap();
                    const ASTMachComponent* newcomp = newsubcomponentmap[newsc->GetName()];
                    //cout << " Name = " << newcomp->GetName() << endl;
                    //cout << " Type = " << newcomp->GetType() << endl;
		    if (newcomp->GetName() == socket){
		        return newcomp;
		    }
 
                }
            }
          }
          catch (const AspenException& exc)
          {
              cerr << exc.PrettyString() <<endl;
          }
 
        } 
     }
 
    std::cout << " Socket: "  << socket << " not found" << std::endl;
    return NULL;
} 


/*
 * Function name: get_any_machine_property
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function finds any property from the machine model
 * 
 * Input: This function takes four arguments, 1. machine model, 2. socket name,
 * 3. component name and  4. property 
 * Output: If found, the value of the property, if not "-1" is returned.
 * 
 * Description: This function at first finds the socket component using  
 * return_socket_component function then travarses through all the properties 
 * of the socket component and finds the property and then, returns the value.
 *
 * Note1: This function does not return global params, works only for 
 * 	property. this can be improved in future.
 * Note2: Only exact mach is considered for the string to be matched. A partial
 * 	mach can be done in the future.
 * Note3: The same way resources and tratis can be searched. Can be done in future.
 * Note4: different component can have same property.
 * Note5: We added some function in the mach/ASTMachComponent.h files.
 */


double 
get_any_machine_property(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property)
{
    double property_value = -1;
    if (mach)
    {
        //cout << "\n ------  Machine model search:property function called ------\n";
        try
        {
            const ASTMachComponent* newcomp = get_socket_component(mach, socket);
	    if ( newcomp == NULL) return property_value;
            //cout << " Name = " << newcomp->GetName() << endl;
            //cout << " Type = " << newcomp->GetType() << endl;
            //newcomp->Print(std::cout);
            const vector<const ASTSubComponent*> newnewsub = newcomp->GetSubComponent();
            //std::cout << " size = " << newnewsub.size() << endl;
   	    for (unsigned int k = 0; k < newnewsub.size(); ++k)
    	    {
                 const ASTSubComponent *newnewsc = newnewsub[k];
                 const NameMap<const ASTMachComponent*> newnewsubcomponentmap = newcomp->GetSubComponentMap();
                 const ASTMachComponent* newnewcomp = newnewsubcomponentmap[newnewsc->GetName()];

                 //const vector<const ASTSubComponent*> newnewsub = newcomp->GetSubComponent();
                 if (newnewsc->GetType() == component)
                 {
                     //cout << " Name = " << newnewsc->GetName() << endl;
                     //cout << " Type = " << newnewsc->GetType() << endl;
                     //newnewsc->Print(std::cout);
 
                     const vector<const ASTMachProperty*> newnewproperties =  newnewcomp->GetProperties();
                     //std::cout << "properties" << newnewproperties.size() << std::endl;
                     for (unsigned int l = 0; l < newnewproperties.size(); ++l)
    	             {
 		         //std::cout << property << std::endl;
                         if (newnewproperties[l]->GetName() == property)
                         {
                             //cout << " Name = " << newnewproperties[l]->GetName() << endl;
                             //cout << " Value = " << newnewproperties[l]->GetValue()->Evaluate() << endl;
                             property_value = newnewproperties[l]->GetValue()->Evaluate();
    			             return property_value; 
                         }
                     } 
                 }
            } 
	    std::cout << std::endl; 
          }
          catch (const AspenException& exc)
          {
              cerr << exc.PrettyString() <<endl;
          }
        
    }

    std::cout << " Property: "  << property << " not found" << std::endl;
    return property_value; 
}


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

            double memory = predict_memory_streaming_access(app, mach, socket); 
            //std::cout << " for Double data type : " << memory * 2 << "\n";
            std::cout << " Total bytes accessed : " << memory << "\n";
            //std::cout << " for float data type : " << memory << "\n";
          }
          catch (const AspenException &exc)
          {
            cerr << exc.PrettyString() <<endl;
          } 
        }
    }
    




 
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

    if (app)
        delete app;
    if (mach)
        delete mach;


    exit (0);



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




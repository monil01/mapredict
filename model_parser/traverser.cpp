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
#include "analytical_model.h"



#define Finegrained_RSC_Print

using namespace std;

traverser::traverser(ASTAppModel *app, ASTMachModel *mach):AspenTool(app,mach){
    //app = app;
    //mach = mach;

}

traverser::~traverser(){

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
traverser::analyticalStreamingAccess(double D, double E, double S, double CL)
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


double traverser::predictMemoryStatement(const ASTRequiresStatement *req, std::string socket,
    double inner_parallelism){
    double memory_access = 0;
    int compiler;
    int instruction_type;
    size_t cache_line_size;
    vector<ASTTrait*> traits;
    //bool initialized;
    bool prefetch_enabled;
    bool multithreaded;
    int64_t data_structure_size;
    int element_size;
    std::string property, component;

    // getting compiler
    property = "compiler";
    component = "cache";

    //std::cout << " " << socket << " : " << component << " : " << property << " " << getAnyMachineProperty(mach, socket, component, property) << std::endl;
    compiler = (int) getAnyMachineProperty(mach, socket, component, property);
    std::cout << " " << socket << " : " << component << " : " << property << " " << compiler << std::endl;

    if (req->GetResource().compare("loads")) instruction_type = instructions::LOAD;
    if (req->GetResource().compare("stores")) instruction_type = instructions::STORE;
    std::cout << " " << socket << " :  instruction " << instruction_type << std::endl;

    // getting cacheline size
    property = "cacheline";
    component = "cache";

    cache_line_size = (size_t) getAnyMachineProperty(mach, socket, component, property);
    std::cout << " " << socket << " : " << component << " : " << property << " " << cache_line_size << std::endl; 
    traits = req->GetTraits();
 
    // getting prefetch 
    property = "prefetch";
    component = "cache";

    prefetch_enabled = (bool) getAnyMachineProperty(mach, socket, component, property);
    std::cout << " " << socket << " : " << component << " : " << property << " " << prefetch_enabled << std::endl; 
    
 
    // getting multithreaded 
    property = "multithreaded";
    component = "cache";

    multithreaded = (bool) getAnyMachineProperty(mach, socket, component, property);
    std::cout << " " << socket << " : " << component << " : " << property << " " << multithreaded << std::endl; 
    
    // getting element size
    std::string param = "aspen_param_sizeof_";
    element_size = (int) getApplicationParam(app, param);
    std::cout << " " << socket << " : " << " element size " << element_size << std::endl; 
 
    // getting data size
    ExpressionBuilder eb = req->GetQuantity()->Cloned();
    double temp_total = eb.GetExpression()->Expanded(app->paramMap)->Evaluate();
    //multiplied by parallelism because of the for loop is counted as parallelism
    data_structure_size = (int64_t) temp_total / element_size * inner_parallelism;
    std::cout << " " << socket << " :  data size " << data_structure_size << std::endl; 
    std::cout << std::endl;	

    analytical_model * ana_model = new analytical_model(compiler, instruction_type, cache_line_size,
        traits, prefetch_enabled, multithreaded, data_structure_size, element_size);

    delete ana_model;
    return memory_access;
}


 
double  
traverser::executeBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    const ASTExecutionBlock *exec,
    double outer_parallelism)
{  
    double total_memory_access = 0;

    //double element_size = 0, stride = 0, total_length = 0;
    //double cacheline = 0;
    //std::string param = "aspen_param_sizeof_";

    //element_size = getApplicationParam(app, param);
    //std::cout << " " << param << " : " << element_size << std::endl;

    //std::string property = "cacheline";
    //std::string property = "cacheline";
    //std::string component = "cache";
    //std::string socket = "nvidia_k80";
    //socket = "intel_xeon_x5660";

    //cacheline = getAnyMachineProperty(mach, socket, component, property);
    //std::cout << " " << socket << " : " << component << " : " << property << " " << cacheline << std::endl; 

    ExpressionBuilder eb;
    Expression* current_expression;
    double inner_parallelism = 1;
    //std::cout << " outer multiplying factor : " << 
    //    exec->GetParallelism()->Expanded(app->paramMap)->Evaluate() << "\n";
    inner_parallelism = exec->GetParallelism()->Expanded(app->paramMap)->Evaluate();
    //parallelism *= outer_parallelism;
    std::cout << " current factor : " << inner_parallelism << "\n";
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
            // not entertaining other types of instructions
            if (req->GetResource() != "loads" && req->GetResource() != "stores" ) return 0;
            // calling the analytical model
			double memory_access_statement =  predictMemoryStatement(req, socket, inner_parallelism);
            /*
             
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
				    analyticalStreamingAccess(total_length, element_size, stride,
				    cacheline);
                memory_access = memory_access * parallelism; 
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
				              analyticalStreamingAccess(total_length, element_size, stride,
					          cacheline);
                memory_access = memory_access * parallelism; 
			    total_memory_access += memory_access;
			    std::cout << " memory access : " << memory_access << "\n \n";
                      
            } 
            */
        }
    } 

    return total_memory_access;

}
    
double  
traverser::recursiveBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    double outer_parallelism,
    const ASTControlStatement *s)
{
    double total_memory_access = 0;
    const ASTExecutionBlock *exec = dynamic_cast<const ASTExecutionBlock*>(s);
    if (exec) // it's a kernel-like requires statement
    {
        std::cout << " Execute block\n";
        total_memory_access += executeBlock(app, mach, socket, exec, outer_parallelism);
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

        total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
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

        total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        return total_memory_access;

    }
 
    const ASTControlSequentialStatement *seq_statements 
                = dynamic_cast<const ASTControlSequentialStatement*>(s);

    if (seq_statements){
        std::cout << " Seq block\n";
        std::cout << " size " << seq_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < seq_statements->GetItems().size(); ++i)
        {
            //double outer_parallelism = 1;

            const ASTControlStatement *s_new = seq_statements->GetItems()[i];
            total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        }
    } 

 
    const ASTControlParallelStatement *par_statements 
                = dynamic_cast<const ASTControlParallelStatement*>(s);

    if (par_statements){
        std::cout << " Par block\n";
        std::cout << " size " << par_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < par_statements->GetItems().size(); ++i)
        {
            //double outer_parallelism = 1;

            const ASTControlStatement *s_new = par_statements->GetItems()[i];
            total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
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

        total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);

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
traverser::predictMemoryAccess(ASTAppModel *app, ASTMachModel *mach, std::string socket)
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

                total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s);

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
traverser::getApplicationParam(ASTAppModel *app, std::string param)
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
traverser::getSocketComponent(ASTMachModel *mach, std::string socket)
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
traverser::getAnyMachineProperty(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property)
{
    double property_value = -1;
    if (mach)
    {
        //cout << "\n ------  Machine model search:property function called ------\n";
        try
        {
            const ASTMachComponent* newcomp = getSocketComponent(mach, socket);
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






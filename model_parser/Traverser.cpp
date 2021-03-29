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
#include "Types.h"
#include "Traverser.h"
#include "AnalyticalModelIntel.h"



#define Finegrained_RSC_Print

using namespace std;

Traverser::Traverser(ASTAppModel *app, ASTMachModel *mach):AspenTool(app,mach){
    //app = app;
    //mach = mach;
    _aspen_utility = new AspenUtility(app, mach);
    _total_loads = 0;
    _total_stores = 0;

}

Traverser::~Traverser(){

    if (_aspen_utility)
        delete _aspen_utility;


}




/*
 * Function name: analytical_streaming_access
 * Function Author: Monil, Mohammad Alaul Haque
 *
 * Objective: This function calculates the analytical streaming access 
 * 
 * Input: This function takes four arguments, 
 * 1.  D = total length of the data structure
 * 2.  E = Element size (i.e., size of float/std::int64_t/int)
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


/*
std::int64_t  
Traverser::analyticalStreamingAccess(std::int64_t D, std::int64_t E, std::int64_t S, std::int64_t CL)
{
    std::int64_t memory_access = 0;
    // converting the stride and data size to bytes
    S = S * E;
    D = D * E;

    int cli = (int) CL;
    int e = (int) E;

    int mod = (e - 1) % cli;
    std::int64_t p = mod / CL;
    //std::int64_t p = find_mod((E - 1), CL) / CL;

    // First Case E >= CL 
    if (E >= CL)
    {
        if ( E < S) 
        {
	    std::int64_t AE = ceil(E/CL) + p;
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
        if(DEBUG_MAPMC == true) std::cout << memory_access << " " <<  S  << " " << D  << " " << p << std::endl;
    }

    // for the third case S < CL
    if ( S < CL) 
    {
	memory_access = ceil(D/CL);
    }

    //if(DEBUG_MAPMC == true) std::cout << " memory access" << memory_access << "\n"; 

    // memory access is multiplied by CL to convert it to bytes
    return memory_access * CL;
}
*/
double Traverser::getExecuteBlockReuse( std::string execute_block_name, std::string socket) {

    double reuse_block = 2;

    std::string microarchitecture = _aspen_utility->getStringMicroarchitecture(
                 _aspen_utility->getMicroarchitecture(socket));
 
    // getting prefetch 
    std::string property = "prefetch";
    std::string component = "cache";

    std::string prefetch_enabled = _aspen_utility->getStringPrefetch(
            (int) _aspen_utility->getAnyMachineProperty(mach, socket, component, property));


    std::string variable_name = "aspen_param_reuse_"+execute_block_name+"_"+microarchitecture+"_"+prefetch_enabled;
    
    if(DEBUG_MAPMC == true) std::cout << " Microarchitecture : " << microarchitecture << " : prefetch: " << prefetch_enabled << " variable string : " << variable_name << std::endl; 
    
    reuse_block = _aspen_utility->getApplicationParamDouble(app, variable_name);

    if(DEBUG_MAPMC == true) std::cout << " Reuse block factor : " << reuse_block << std::endl; 

   return reuse_block; 
}

std::int64_t Traverser::predictMemoryStatement(const ASTRequiresStatement *req, std::string socket,
    std::int64_t inner_parallelism){
    std::int64_t memory_access = 0;
    int compiler;
    int instruction_type;
    int cache_line_size;
    vector<ASTTrait*> traits;
    //bool initialized;
    bool prefetch_enabled;
    bool multithreaded;
    int64_t data_structure_size;
    int element_size;
    int microarchitecture;
    std::string property, component;

    //if(DEBUG_MAPMC == true) std::cout << " statement " << req->GetText() << std::endl;
    
    // getting compiler
    property = "compiler";
    component = "cache";

    //if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << getAnyMachineProperty(mach, socket, component, property) << std::endl;
    compiler = (int) _aspen_utility->getAnyMachineProperty(mach, socket, component, property);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << compiler << std::endl;

    if (req->GetResource().compare("loads") == 0) instruction_type = instructions::LOAD;
    if (req->GetResource().compare("stores") == 0) instruction_type = instructions::STORE;
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " :  instruction " 
        << instruction_type << " string " << req->GetResource() 
        << " instruction type " << instructions::LOAD << std::endl;

    // getting cacheline size
    property = "cacheline";
    component = "cache";

    cache_line_size = (int) _aspen_utility->getAnyMachineProperty(mach, socket, component, property);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << cache_line_size << std::endl; 

    // getting microarchitecture
    property = "microarchitecture";
    component = "cache";


    microarchitecture = _aspen_utility->getMicroarchitecture(socket);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << microarchitecture << std::endl; 


    traits = req->GetTraits();
 
    // getting prefetch 
    property = "prefetch";
    component = "cache";

    prefetch_enabled = (bool) _aspen_utility->getAnyMachineProperty(mach, socket, component, property);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << prefetch_enabled << std::endl; 
    
 
    // getting multithreaded 
    property = "multithreaded";
    component = "cache";

    multithreaded = (bool) _aspen_utility->getAnyMachineProperty(mach, socket, component, property);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << multithreaded << std::endl; 
    
    // getting element size
    ExpressionBuilder ebuilder = req->GetQuantity()->Cloned();
    std::string param = _aspen_utility->getNameOfDataType(ebuilder.GetExpression()->GetText());
    element_size = (int) _aspen_utility->getApplicationParam(app, param);
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << " element name: " << param << " element size " << element_size << std::endl; 
 
    // getting data size
    std::int64_t temp_total = ebuilder.GetExpression()->Expanded(app->paramMap)->Evaluate();
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " :  expression " <<  ebuilder.GetExpression()->GetText() << " " << temp_total << std::endl; 
    
    //multiplied by parallelism because of the for loop is counted as parallelism
    data_structure_size = (std::int64_t) ((temp_total / (double)element_size) * inner_parallelism);

    if(DEBUG_MAPMC == true) std::cout << " " << socket << " :  value " << temp_total << " " << element_size << " parallel " << inner_parallelism << std::endl; 
    if(DEBUG_MAPMC == true) std::cout << " " << socket << " :  data size " << data_structure_size << std::endl; 
    if(DEBUG_MAPMC == true) std::cout << std::endl;	

    if(DEBUG_MAPMC == true) std::cout << " Statement: " << req->GetResource() << " " << ebuilder.GetExpression()->GetText() << " " << req->GetToFrom() << " " << std::endl; 

    AnalyticalModelIntel * ana_model = new AnalyticalModelIntel(compiler, instruction_type, cache_line_size,
        traits, prefetch_enabled, multithreaded, data_structure_size, element_size, microarchitecture);
    memory_access = (std::int64_t) ana_model->predictMemoryAccess(); 
    if(DEBUG_MAPMC == true) std::cout << " memory access : " << memory_access << "\n";

    delete ana_model;

    return memory_access;
}


std::int64_t
Traverser::executeBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    const ASTExecutionBlock *exec,
    std::int64_t outer_parallelism)
{  
    std::int64_t total_memory_access = 0;

    //std::int64_t element_size = 0, stride = 0, total_length = 0;
    //std::int64_t cacheline = 0;
    //std::string param = "aspen_param_sizeof_";

    //element_size = getApplicationParam(app, param);
    //if(DEBUG_MAPMC == true) std::cout << " " << param << " : " << element_size << std::endl;

    //std::string property = "cacheline";
    //std::string property = "cacheline";
    //std::string component = "cache";
    //std::string socket = "nvidia_k80";
    //socket = "intel_xeon_x5660";

    //cacheline = getAnyMachineProperty(mach, socket, component, property);
    //if(DEBUG_MAPMC == true) std::cout << " " << socket << " : " << component << " : " << property << " " << cacheline << std::endl; 

    ExpressionBuilder eb;
    Expression* current_expression;
    std::int64_t inner_parallelism = 1;
    //if(DEBUG_MAPMC == true) std::cout << " outer multiplying factor : " << 
    //    exec->GetParallelism()->Expanded(app->paramMap)->Evaluate() << "\n";
    inner_parallelism = exec->GetParallelism()->Expanded(app->paramMap)->Evaluate();
    //parallelism *= outer_parallelism;
    if(DEBUG_MAPMC == true) std::cout << " current factor : " << inner_parallelism << "\n";
    if(DEBUG_MAPMC == true) std::cout << " outer factor : " << outer_parallelism << "\n";
    //std::string resource;
	//resource = "loads"; stride = 0; total_length = 0;
    //if(DEBUG_MAPMC == true) std::cout << exec->GetResourceRequirementExpression(app,resource)->GetText();
    const vector<ASTExecutionStatement*> statements_exec = exec->GetStatements(); 

     
    for (unsigned int j = 0; j < statements_exec.size(); ++j)
    {
        const ASTExecutionStatement *s = statements_exec[j];
        const ASTRequiresStatement *req = dynamic_cast<const ASTRequiresStatement*>(s);
        if (req) // requires statement
        {
            // not entertaining other types of instructions
            if (req->GetResource() != "loads" && req->GetResource() != "stores") continue;
            if (req->GetToFrom().length() < 1) continue;

            double reuse_block = getExecuteBlockReuse(exec->GetName(), socket);
            exit(0);
            if(DEBUG_MAPMC == true) std::cout << " Variable name: To to from: " << req->GetToFrom() << " -- instruction type " << req->GetResource() << "\n";
            // calling the analytical model
			std::int64_t memory_access_statement =  predictMemoryStatement(req, socket, inner_parallelism);
			if(DEBUG_MAPMC == true) std::cout << " memory access statement : " << memory_access_statement << "\n";

            if (req->GetResource() == "loads") _total_loads += memory_access_statement;
            if (req->GetResource() == "stores") _total_stores += memory_access_statement;

            total_memory_access += memory_access_statement;
            if(DEBUG_MAPMC == true) std::cout << " Upto now total program Total loads : " << _total_loads << " Total sotres: " << _total_stores << "\n";
            if(DEBUG_MAPMC == true) std::cout << " Upto now Executive block memory access : " << total_memory_access << "\n \n";
        }
    } 

    total_memory_access *= outer_parallelism;


    if(DEBUG_MAPMC == true) std::cout << " Execute Block name: " << exec->GetName() << "\n";
    if(DEBUG_MAPMC == true) std::cout << " Total program total loads : " << _total_loads << " total stores " << _total_stores << "\n";
    if(DEBUG_MAPMC == true) std::cout << " Total executive block memory access : " << total_memory_access << "\n \n";

    //exit(0);
    return total_memory_access;

}
    
std::int64_t
Traverser::recursiveBlock(ASTAppModel *app, ASTMachModel *mach, std::string socket, 
    std::int64_t outer_parallelism,
    const ASTControlStatement *s)
{
    std::int64_t total_memory_access = 0;
    const ASTExecutionBlock *exec = dynamic_cast<const ASTExecutionBlock*>(s);
    if (exec) // it's a kernel-like requires statement
    {
        if(DEBUG_MAPMC == true) std::cout << " ---------------------- Execute block ------------------\n";
        if(DEBUG_MAPMC == true) std::cout << " Execute Block name: " << exec->GetName() << "\n";
        total_memory_access += executeBlock(app, mach, socket, exec, outer_parallelism);
        //if(DEBUG_MAPMC == true) std::cout << "\n total memory upto this block: " << total_memory_access << "\n";
        return total_memory_access;
    }
    
    const ASTControlIterateStatement *iter_statement = 
        dynamic_cast<const ASTControlIterateStatement*>(s);

    //const ASTControlSequentialStatement *statements = 
                    //dynamic_cast<const ASTControlSequentialStatement *>(s);
    if (iter_statement){
        if(DEBUG_MAPMC == true) std::cout << " Iter block\n";
        std::int64_t new_parallelism = iter_statement->GetQuantity()->Expanded(app->paramMap)->Evaluate();

        outer_parallelism *= new_parallelism;

        if(DEBUG_MAPMC == true) std::cout << "\n from iter: new factor: " << new_parallelism << "\n";
        ASTControlStatement * s_new = iter_statement->GetItem();

        total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        return total_memory_access;

    }
  
 
    const ASTControlMapStatement *map_statement = 
        dynamic_cast<const ASTControlMapStatement*>(s);

    //const ASTControlSequentialStatement *statements = 
                    //dynamic_cast<const ASTControlSequentialStatement *>(s);
    if (map_statement){
        if(DEBUG_MAPMC == true) std::cout << " Map block\n";
        std::int64_t new_parallelism = map_statement->GetQuantity()->Expanded(app->paramMap)->Evaluate();

        outer_parallelism *= new_parallelism;

        if(DEBUG_MAPMC == true) std::cout << "\n from map: new factor: " << new_parallelism << "\n";
        ASTControlStatement * s_new = map_statement->GetItem();

        total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        return total_memory_access;

    }
 
    const ASTControlSequentialStatement *seq_statements 
                = dynamic_cast<const ASTControlSequentialStatement*>(s);

    if (seq_statements){
        if(DEBUG_MAPMC == true) std::cout << " Seq block\n";
        if(DEBUG_MAPMC == true) std::cout << " size " << seq_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < seq_statements->GetItems().size(); ++i)
        {
            //std::int64_t outer_parallelism = 1;

            const ASTControlStatement *s_new = seq_statements->GetItems()[i];
            total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        }
    } 

 
    const ASTControlParallelStatement *par_statements 
                = dynamic_cast<const ASTControlParallelStatement*>(s);

    if (par_statements){
        if(DEBUG_MAPMC == true) std::cout << " Par block\n";
        if(DEBUG_MAPMC == true) std::cout << " size " << par_statements->GetItems().size() << std::endl;

        for (unsigned int i=0; i < par_statements->GetItems().size(); ++i)
        {
            //std::int64_t outer_parallelism = 1;

            const ASTControlStatement *s_new = par_statements->GetItems()[i];
            total_memory_access += recursiveBlock(app, mach, socket, outer_parallelism, s_new);
        }
    } 

    const ASTControlKernelCallStatement *call_statements 
                = dynamic_cast<const ASTControlKernelCallStatement*>(s);

    if (call_statements){
        if(DEBUG_MAPMC == true) std::cout << " Call block\n";
        const ASTKernel *k = call_statements->GetKernel(app);
        if(DEBUG_MAPMC == true) std::cout << " Kernel name: " << k->GetName() << "\n";
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

std::int64_t
Traverser::predictMemoryAccess(ASTAppModel *app, ASTMachModel *mach, std::string socket)
{
    std::int64_t total_memory_access = 0;


    //std::string size_name;

    for (unsigned int i=0; i<app->GetKernels().size(); ++i)
    {
        try
        {
            ASTKernel *k = app->GetKernels()[i];
            if(k->GetName() == "main")
            {
                const ASTControlSequentialStatement *statements = k->GetStatements();
                std::int64_t outer_parallelism = 1;
 
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

    //if(DEBUG_MAPMC == true) std::cout << " Total memory access: " << total_memory_access << "\n"; 
    return total_memory_access;

} 



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
#include "AspenUtility.h"
#include "AnalyticalModelIntel.h"



#define Finegrained_RSC_Print

using namespace std;

AspenUtility::AspenUtility(ASTAppModel *app, ASTMachModel *mach):AspenTool(app,mach){
    //app = app;
    //mach = mach;

}

AspenUtility::~AspenUtility(){

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


std::int64_t  
AspenUtility::getApplicationParam(ASTAppModel *app, std::string param)
{
    std::int64_t param_value = -1;
    if (app)
    {
        //cout << "\n ------  Application model search:param search function called ------\n";
        try
        {
            vector<ASTStatement*> globals = app->GetGlobals();

            //if(DEBUG_MAPMC == true) std::cout <<  "Size globals" << globals.size() << std::endl;

            for (unsigned int i=0; i<globals.size(); ++i)
            {
                //const ASTDataStatement *data = dynamic_cast<const ASTDataStatement*>(globals[i]);
                const ASTAssignStatement *data = dynamic_cast<const ASTAssignStatement*>(globals[i]);
                if (!data)
                    continue;
                std::string temp = data->GetName();
                //if(DEBUG_MAPMC == true) std::cout << "Identifier name " << data->GetName() << std::endl;
                //if(DEBUG_MAPMC == true) std::cout << "Identifier Value " << data->GetValue()->Evaluate() << std::endl;
                if (temp.find(param) != std::string::npos) {
                    param_value = data->GetValue()->Evaluate();
                    //if(DEBUG_MAPMC == true) std::cout << "param " << temp << " : " << param_value << std::endl;
                    return param_value; 
                }
            }  

        }
        catch (const AspenException& exc)
        {
            cerr << exc.PrettyString() <<endl;
        }
    }
    if(DEBUG_MAPMC == true) std::cout << " Param: "  << param << " not found" << std::endl;
    return param_value; 
} 


double  
AspenUtility::getApplicationParamDouble(ASTAppModel *app, std::string param)
{
    double param_value = -1;
    if (app)
    {
        //cout << "\n ------  Application model search:param search function called ------\n";
        try
        {
            vector<ASTStatement*> globals = app->GetGlobals();

            //if(DEBUG_MAPMC == true) std::cout <<  "Size globals" << globals.size() << std::endl;

            for (unsigned int i=0; i<globals.size(); ++i)
            {
                //const ASTDataStatement *data = dynamic_cast<const ASTDataStatement*>(globals[i]);
                const ASTAssignStatement *data = dynamic_cast<const ASTAssignStatement*>(globals[i]);
                if (!data)
                    continue;
                std::string temp = data->GetName();
                //if(DEBUG_MAPMC == true) std::cout << "Identifier name " << data->GetName() << std::endl;
                //if(DEBUG_MAPMC == true) std::cout << "Identifier Value " << data->GetValue()->Evaluate() << std::endl;
                if (temp.find(param) != std::string::npos) {
                    param_value = data->GetValue()->Evaluate();
                    //if(DEBUG_MAPMC == true) std::cout << "param " << temp << " : " << param_value << std::endl;
                    return param_value; 
                }
            }  

        }
        catch (const AspenException& exc)
        {
            cerr << exc.PrettyString() <<endl;
        }
    }
    if(DEBUG_MAPMC == true) std::cout << " Param: "  << param << " not found" << std::endl;
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
AspenUtility::getSocketComponent(ASTMachModel *mach, std::string socket)
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
 
    if(DEBUG_MAPMC == true) std::cout << " Socket: "  << socket << " not found" << std::endl;
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


std::int64_t 
AspenUtility::getAnyMachineProperty(ASTMachModel *mach, 
			std::string socket,
			std::string component, std::string property)
{
    std::int64_t property_value = -1;
    if (mach)
    {
        //cout << "\n ------  Machine model search:property function called ------\n";
        try
        {
            const ASTMachComponent* newcomp = getSocketComponent(mach, socket);
	    if ( newcomp == NULL) return property_value;
            //cout << " Name = " << newcomp->GetName() << endl;
            //cout << " Type = " << newcomp->GetType() << endl;
            //newcomp->Print(if(DEBUG_MAPMC == true) std::cout);
            const vector<const ASTSubComponent*> newnewsub = newcomp->GetSubComponent();
            //if(DEBUG_MAPMC == true) std::cout << " size = " << newnewsub.size() << endl;
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
                     //newnewsc->Print(if(DEBUG_MAPMC == true) std::cout);
 
                     const vector<const ASTMachProperty*> newnewproperties =  newnewcomp->GetProperties();
                     //if(DEBUG_MAPMC == true) std::cout << "properties" << newnewproperties.size() << std::endl;
                     for (unsigned int l = 0; l < newnewproperties.size(); ++l)
    	             {
 		         //if(DEBUG_MAPMC == true) std::cout << property << std::endl;
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
	    if(DEBUG_MAPMC == true) std::cout << std::endl; 
          }
          catch (const AspenException& exc)
          {
              cerr << exc.PrettyString() <<endl;
          }
        
    }

    if(DEBUG_MAPMC == true) std::cout << " Property: "  << property << " not found" << std::endl;
    return property_value; 
}

int AspenUtility::getMicroarchitecture(std::string socket){

    // getting microarchitecture
    std::string property = "microarchitecture";
    std::string component = "cache";

    int micro = (int) getAnyMachineProperty(mach, socket, component, property);
    return micro;

}


std::string  AspenUtility::getNameOfDataType(std::string str_expression){

    std::string type_name;
    if(str_expression.find("aspen_param_sizeof_int") != std::string::npos) {
        if(DEBUG_MAPMC == true) std::cout << " type " << str_expression << " " << str_expression.find("aspen_param_sizeof_int") << std::endl;
        return "aspen_param_sizeof_int";
    }
    if(str_expression.find("aspen_param_sizeof_double") != std::string::npos){
        if(DEBUG_MAPMC == true) std::cout << " type " << str_expression << " " << str_expression.find("aspen_param_sizeof_double") << std::endl;
        return "aspen_param_sizeof_double";
    }
    if(str_expression.find("aspen_param_sizeof_float") != std::string::npos){ 
        if(DEBUG_MAPMC == true) std::cout << " type " << str_expression << " " << str_expression.find("aspen_param_sizeof_float") << std::endl;
       return "aspen_param_sizeof_float";
    }
}

std::string  AspenUtility::getStringMicroarchitecture(int integer_microarchitecture){
    std::string microarchitecture = "";
    
    switch(integer_microarchitecture) {
        case 0: 
            microarchitecture = "BW";
            break;
        case 1: 
            microarchitecture = "SK";
            break;
        case 2: 
            microarchitecture = "CS";
            break;
        case 3: 
            microarchitecture = "CP";
            break;
        defualt: 
            microarchitecture = "NOTHING-ERROR";
            break;
    }
    
    return microarchitecture; 
}


std::string  AspenUtility::getStringPrefetch(int integer_prefetch){
    std::string prefetch = "";
    
    switch(integer_prefetch) {
        case 0: 
            prefetch = "noprefetch";
            break;
        case 1: 
            prefetch = "prefetch";
            break;
        defualt: 
            prefetch = "NOTHING-ERROR";
            break;
    }
    
    return prefetch; 
}


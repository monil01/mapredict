// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef AST_MACH_COMPONENT_H
#define AST_MACH_COMPONENT_H

#include "common/AST.h"
#include "common/MappingRestriction.h"
#include "common/ParameterStack.h"
#include "expr/BinaryExpr.h"
#include "expr/Real.h"
#include "mach/ASTMachPower.h"
#include "mach/ASTMachProperty.h"
#include "mach/ASTMachResource.h"
#include "mach/ASTResourceConflict.h"
#include "model/ASTStatement.h"
#include "mach/ASTSubComponent.h"
#include "app/ASTTrait.h"


class ASTMachModel;


// ****************************************************************************
// Class:  ASTMachComponent
//
// Purpose:
///   A component (node, socket, core, mem, machine, etc.) of a machine model.
//
// Programmer:  Jeremy Meredith
// Creation:    May 21, 2013
//
// Modifications:
// ****************************************************************************

class ASTMachComponent : public ASTNode
{
  protected:
    string type;
    string name;
    vector<const ASTSubComponent*> subcomponents;
    vector<const ASTMachProperty*> properties;
    vector<const ASTMachResource*> resources;
    vector<const ASTMachPower*>    power;
    vector<const ASTResourceConflict*> conflicts;

    // acceleration structures
    const ASTMachModel *mach;
    NameMap<const ASTMachComponent*> subcomponentmap;
    NameMap<const ASTMachResource*> resourceMap;

  public:
    ASTMachComponent(string type,
                     string name,
                     ParseVector<ASTNode*> contents);
    virtual ~ASTMachComponent();
    virtual void Print(ostream &out, int indent = 0) const;
    virtual void CompleteAndCheck(ASTMachModel *mach);
    Expression *GetSelfTimeExpression(const string &resource,
                                      const vector<ASTTrait*> &traits, 
                                      const Expression *value) const;
    Expression *GetSerialTimeExpression(const string &resource,
                                        const vector<ASTTrait*> &traits, 
                                        const Expression *value,
                                        const MappingRestriction &restriction) const;
    Expression *GetIdealizedTimeExpression(const string &resource,
                                           const vector<ASTTrait*> &traits, 
                                           const Expression *value,
                                           const MappingRestriction &restriction) const;
    bool CheckConflict(string ra, string rb,
                       const MappingRestriction &restriction) const;
    Expression *GetTotalQuantityExpression(const string &resource, const MappingRestriction &restriction) const;

    const string &GetName() const { return name; }
    const string &GetType() const { return type; }
    const vector<const ASTSubComponent*> &GetSubComponent() const { return subcomponents;}
    const NameMap<const ASTMachComponent*> &GetSubComponentMap() const { return subcomponentmap;}
    const vector<const ASTMachProperty*> &GetProperties() const { return properties;}

    /*const void printAll() const {
      
      std::cout << " power " << power.size() << std::endl;
      std::cout << " Sub " << subcomponents.size() << std::endl;
      std::cout << " resourses " << resources.size() << std::endl;
      std::cout << " property " << properties.size() << std::endl;
      for(int i = 0; i < subcomponents.size(); i++)
      {
           std::cout << " property " << subcomponents[i]->GetName() << std::endl;

      } 
   

    } */
};

#endif

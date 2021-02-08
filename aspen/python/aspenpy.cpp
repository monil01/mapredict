// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;

#include <model/ASTAppModel.h>
#include <model/ASTMachModel.h>
#include <mach/ASTMachComponent.h>
#include <parser/Parser.h>

template <class T>
boost::python::list Vector_Of_Pointers_To_List(const std::vector<T*> &v)
{
    boost::python::list pylist;
    for (auto &t: v)
    {
        pylist.append(ptr(t));
    }
    return pylist;
}

template <class T>
boost::python::dict NameMap_Of_Pointers_To_Dict(const NameMap<const T*> &m)
{
    boost::python::dict pydict;
    for (auto &t: m)
    {
        //pydict[t.first] = t.second;
        pydict[t.first] = ptr(t.second);
    }
    return pydict;
}

template <class T>
void Fill_NameMap_From_Dict(NameMap<const T*> &nm, boost::python::dict pydict)
{
    boost::python::list keys = pydict.keys();
    int nitems = len(keys);
    for (int i=0; i<nitems; ++i)
    {
        object pykey = keys[i];
        object pyvalue = pydict[pykey];
        string key;

        //
        // get the key as a string
        //
        try
        {
            key = extract<string>(pykey);
        }
        catch (...)
        {
            // keys had better be strings; just ignore this one and continue
            PyErr_Clear();
            continue;
        }

        //
        // successively try some ways to get the value
        //
        try
        {
            const T *value = extract<const T*>(pyvalue);
            nm[key] = value;
            // success!  assign and continue
            PyErr_Clear();
            continue;
        }
        catch (...)
        {
            PyErr_Clear();
            // fall through on failure for next attempted extraction
        }

        try
        {
            double value = extract<double>(pyvalue);
            ///\todo: fix leak:
            nm[key] = new Real(value);
            // success!  assign and continue
            continue;
        }
        catch (...)
        {
            PyErr_Clear();
            // fall through on failure for next attempted extraction
        }

    }
}

class ExpressionWrap : public Expression, public wrapper<Expression>
{
  public:
    void Print(ostream &out, int indent = 0) const
    {
        this->get_override("Print")(out, indent);
    }
    double Evaluate() const
    {
        return this->get_override("Evaluate")();
    }
    std::string GetText(TextStyle style) const
    {
        return this->get_override("GetText")(style);
    }
    Expression *Cloned() const
    {
        return this->get_override("Cloned")();
    }
    void Visit(ModelVisitor *visitor)
    {
        this->get_override("Visit")(visitor);
    }
};

class StatementWrap : public ASTStatement, public wrapper<ASTStatement>
{
  public:
    void Print(ostream &out, int indent = 0) const
    {
        this->get_override("Print")(out, indent);
    }
};

boost::python::list AppModel_GetGlobals(const ASTAppModel *app)
{
    return Vector_Of_Pointers_To_List(app->GetGlobals());
}

boost::python::dict AppModel_GetParamMap(const ASTAppModel *app)
{
    return NameMap_Of_Pointers_To_Dict(app->paramMap);
}

boost::python::dict MachModel_GetParamMap(const ASTMachModel *mach)
{
    return NameMap_Of_Pointers_To_Dict(mach->paramMap);
}

void Expression_Print(const Expression *expr)
{
    expr->Print(cout);
}

Expression *Expression_Expanded(const Expression *expr, boost::python::dict paramMap)
{
    NameMap<const Expression*> p;
    Fill_NameMap_From_Dict(p, paramMap);
    return expr->Expanded(p);
}

string Expression_ToString(const Expression *expr)
{
    if (dynamic_cast<const Value*>(expr))
    {
        return "<Value (" + expr->GetText() + ")>";
    }
    if (dynamic_cast<const Identifier*>(expr))
    {
        return "<Identifier (" + expr->GetText() + ")>";
    }
    else if (expr->GetDepth() < 3)
    {
        return "<Expression (" + expr->GetText() + ")>";
    }
    else
    {
        return "<Expression (...)>";
    }
}

BOOST_PYTHON_MODULE(aspen)
{
    class_<std::vector<ASTStatement*> >("vec_Statement")
        .def(vector_indexing_suite<std::vector<ASTStatement*> >())
    ;
    class_<std::vector<ASTKernel*> >("vec_Kernel")
        .def(vector_indexing_suite<std::vector<ASTKernel*> >())
    ;

    enum_<Expression::TextStyle>("TextStyle")
        .value("ASPEN",   Expression::ASPEN)
        .value("C",       Expression::C)
        .value("GNUPLOT", Expression::GNUPLOT)
        ;

    class_<ExpressionWrap, boost::noncopyable>("Expression")
        .def("GetText", pure_virtual(&Expression::GetText), (arg("style")=Expression::ASPEN) ) 
        .def("Simplified", &Expression::Simplified, return_value_policy<manage_new_object>())
        .def("Expanded", &Expression_Expanded, return_value_policy<manage_new_object>())
        .def("Evaluate", pure_virtual(&Expression::Evaluate))
        .def("Print", &Expression_Print)
        .def("__repr__", &Expression_ToString)
        ;
    class_<ASTAppModel>("AppModel", no_init)
        .def("GetName", &ASTAppModel::GetName, return_value_policy<copy_const_reference>())
        .def("GetResourceRequirementExpression", &ASTAppModel::GetResourceRequirementExpression, return_value_policy<manage_new_object>())
        .def("GetGlobalArraySizeExpression", &ASTAppModel::GetGlobalArraySizeExpression, return_value_policy<manage_new_object>())
        .def("GetMainKernel", &ASTAppModel::GetMainKernel, return_internal_reference<1>())
        .def("GetGlobals", &AppModel_GetGlobals)
        .def("GetParamMap", &AppModel_GetParamMap)
        ;
    class_<ASTKernel>("Kernel", no_init)
        .def("GetName", &ASTKernel::GetName, return_value_policy<copy_const_reference>())
        .def("GetTimeExpression", &ASTKernel::GetTimeExpression, return_value_policy<manage_new_object>());
        ;
    class_<ASTMachModel>("MachModel", no_init)
        .def("GetMachine", &ASTMachModel::GetMachine, return_internal_reference<1>())
        .def("GetParamMap", &MachModel_GetParamMap)
        ;
    class_<StatementWrap, boost::noncopyable>("Statement", no_init)
        ;
    class_<ASTMachComponent>("MachComponent", no_init)
        .def("GetName", &ASTMachComponent::GetName, return_value_policy<copy_const_reference>())
        .def("GetType", &ASTMachComponent::GetType, return_value_policy<copy_const_reference>())
        ;
    def("LoadAppModel", LoadAppModel, return_value_policy<manage_new_object>());
    def("LoadMachineModel", LoadMachineModel, return_value_policy<manage_new_object>());


}

// Copyright 2013-2017 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <deque>
#include <cstdio>
#include <map>

#include "model/ASTAppModel.h"
#include "model/ASTMachModel.h"
#include "parser/Parser.h"
#include "walkers/AspenTool.h"
#include "walkers/RuntimeCounter.h"

#include "InputParser.h"

#include <sys/time.h>
#include <nlopt.hpp>

void PrintUsageAndExit()
{
    cerr << "Usage: blackbox <input.csv> <input.aspen> <output.aspen>\n";
    exit(1);
}

static bool debug = true;

// variable names, ranges, and default values to solve for
int numVars;
vector<string> varNames;
vector<double> upperBounds;
vector<double> lowerBounds;
vector<double> guesses;

// runtime expressions to solve for
int numRows;
vector<Expression*> predictions;
vector<double> measurements;

void
ExtractRangeParamsFromApp(ASTAppModel *app)
{
    if (debug)
        cout << endl << "Solving for variables:" << endl;

    vector<const Identifier*> identifiers;
    app->FindParametersWithRanges(identifiers,
                                  guesses,
                                  lowerBounds,
                                  upperBounds);
    for (const Identifier *id : identifiers)
    {
        string name = id->GetName();
        if (debug)
        {
            int n = varNames.size();
            cout << name << " in range ["
                 << lowerBounds[n] << " .. " << upperBounds[n] << "]"
                 << " (starting guess = " << guesses[n] << ")"
                 << endl;
        }
        varNames.push_back(name);
    }

    numVars = varNames.size();
}

void
EraseVarsFromParameterMap(InputParser &parser, ASTAppModel *app)
{
    if (debug)
        cout << endl << "Erasing unknowns\n";
    for (auto name : varNames)
    {
        if (debug)
            cout << "    erasing variable " << name << " definition\n";
        app->paramMap.Erase(name);
        //paramMap.Erase(name);
    }

    if (debug)
        cout << endl << "Erasing CSV-specified variables\n";
    for (auto name : parser.getSampleNames())
    {
        if (debug)
            cout << "    erasing variable " << name << " definition\n";
        app->paramMap.Erase(name);
        //paramMap.Erase(name);
    }
}

void
GenerateTimeExpressionsForRows(InputParser &parser, ASTAppModel *app)
{
    numRows = parser.getDataVector().size();
    if (debug)
        cout << endl << "Generating " << numRows << " predictions\n";

    for (int row=0; row < numRows; ++row)
    {
        string measurement_type = "runtime";
        if (parser.getRuntimeIdx() != -1)
        {
            double runtime = atof(parser.getDataVector()[row][parser.getRuntimeIdx()].c_str());
            measurements.push_back(runtime);
        }
        else
        {
            double measurement = atof(parser.getDataVector()[row][parser.getMeasurementIdx()].c_str());
            measurements.push_back(measurement);
            measurement_type = parser.getDataVector()[row][parser.getTypeIdx()];
        }

        string machPath = parser.getDataVector()[row][parser.getMachineIdx()];
        string socket = parser.getDataVector()[row][parser.getSocketIdx()];
        ASTMachModel *mach = LoadMachineModel(machPath);

        Expression *expr;
        if (measurement_type == "runtime")
            expr = app->mainKernel->GetTimeExpression(app,mach,socket);
        else // it's some sort of resource
            expr = app->mainKernel->GetResourceRequirementExpression(app,measurement_type);

        /*
        NameMap<const Expression*> expand_n2;
        for (auto i : Parser.getSampleNames())
            expand_n2[i] = new Identifier(i + to_string(idx+1));
        Expression *t2 = expr->Expanded(expand_n2);
        */

        expr->ExpandInPlace(app->paramMap);
        expr->ExpandInPlace(mach->paramMap);

        Expression *e = expr->Simplified();
        predictions.push_back(e);

        delete expr;

        if (debug)
        {
            cout << "  prediction type '"<<measurement_type<<"' for " << machPath<< " " << socket<< ":" << endl;
            cout << "    " << e->GetText() << endl;
        }
    }
}

void SubstituteSolutionIntoModel(ASTAppModel *app)
{
    for (int i=0; i < numVars; ++i)
        app->RedefineParam(varNames[i], new Real(guesses[i]));
}

int iterationcount = 0;
double objective(const vector<double> &x, vector<double> &, void *p)
{
    iterationcount++;
    InputParser &parser = *((InputParser*)p);

    double totalError = 0;
    for (int row=0; row < numRows; ++row)
    {
        ///\todo: inefficient to call "new" so often.
        NameMap<const Expression*> vars;
        for (int i=0; i < numVars; ++i)
        {
            vars[varNames[i]] = new Real(x[i]);
        }
        for (unsigned int i=0; i < parser.getSampleNames().size() ; ++i)
        {
            const string &name = parser.getSampleNames()[i];
            const string &value = parser(row, parser.getSampleIdx()[i]);
            vars[name] = new Real(atof(value.c_str()));
        }

        double prediction = predictions[row]->EvaluateWithExpansion(vars);
        if (debug)
        {
            //cout << "prediction " << row << " = " << prediction << endl;
        }
        double measurement = measurements[row];

        double error = prediction - measurement;
        totalError += error*error;
    }
    return sqrt(totalError);
}

void Solve(InputParser parser, ASTAppModel *app)
{
    //nlopt::opt optimizer(nlopt::GN_ISRES, numVars);
    nlopt::opt optimizer(nlopt::GN_CRS2_LM, numVars);
    //nlopt::opt optimizer(nlopt::G_MLSL, numVars);
    optimizer.set_lower_bounds(lowerBounds);
    optimizer.set_upper_bounds(upperBounds);
    optimizer.set_min_objective(objective, &parser);
    optimizer.set_maxtime(4);

    double error = 0;

    /*nlopt::result result =*/ optimizer.optimize(guesses, error);

    double magnitude = pow(10.0, int(log(double(iterationcount))/log(10.)));
    double approx_iter = int(double(iterationcount)/magnitude + 0.5)*magnitude;

    cout << endl;
    cout << "Final Result:" << endl;
    cout << "   RMS error:" << error << endl;
    cout << "   iterations (approx):" << approx_iter << endl;

    for (int i=0; i < numVars; ++i)
    {
        cout << varNames[i] << " = " << guesses[i] << endl;
    }
}

int main(int argc, char *argv[])
{
    try {
        if (argc != 4)
            PrintUsageAndExit();

        ASTAppModel *app = LoadAppModel(argv[2]);
        if (!app)
            PrintUsageAndExit();

        InputParser parser(argv[1]);

        ExtractRangeParamsFromApp(app);
        EraseVarsFromParameterMap(parser, app);
        GenerateTimeExpressionsForRows(parser, app);
        Solve(parser, app);
        SubstituteSolutionIntoModel(app);

        ofstream out(argv[3]);
        app->Export(out);
        out.close();
    }
    catch (const AspenException &exc)
    {
        cerr << exc.PrettyString() << endl;
        return -1;
    }

    return 0;
}

// Copyright 2013-2017 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include "InputParser.h"
#include "common/Exception.h"

InputParser::InputParser(string path) : CSVParser(path, true)
{
    machineIdx = -1;
    socketIdx = -1;
    measurementIdx = -1;
    typeIdx = -1;
    runtimeIdx = -1;

    // Get machine, runtine, and socket index, and 
    // assume the rest are problem parameters.
    int a = 0;
    for (auto i : formatVec)
    {
        if (i == "machine")
            machineIdx = a;
        else if (i == "socket")
            socketIdx = a;
        else if (i == "measurement")
            measurementIdx = a;
        else if (i == "type")
            typeIdx = a;
        else if (i == "runtime")
            runtimeIdx = a;
        else
        {
            //Assume anything else remaining is the sample varible.
            sampleIdx.push_back(a);
            sampleNames.push_back(i);
        }
        a++;
    }

    if (machineIdx == -1 || socketIdx == -1)
        THROW(InputError, "Input file must have a 'machine' and 'socket' column.");
    if (runtimeIdx == -1 && (typeIdx == -1 || measurementIdx == -1))
        THROW(InputError, "Input file must have either a 'runtime' column or a 'measurement' and 'type' column.");
    if (runtimeIdx != -1 && (typeIdx != -1 || measurementIdx != -1))
        THROW(InputError, "Input file cannot have both a 'runtime' column and either a 'measurement' or 'type' column.");
}

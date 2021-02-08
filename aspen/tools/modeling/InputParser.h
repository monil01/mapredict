// Copyright 2013-2017 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef INPUTPARSER_HPP
#define INPUTPARSER_HPP

#include "CSVParser.h"

class InputParser : public CSVParser
{
  private:
    string socket;
    string machine;
    int machineIdx;
    int socketIdx;
    int runtimeIdx;
    int measurementIdx;
    int typeIdx;
    vector<int> sampleIdx;
    vector<string> sampleNames;
    int i = -1;
    int nIdx = -1;
    int sIdx = -1;
  public:
    InputParser(string path);
    ~InputParser() {}
    vector<string>& getFormatVector() { return formatVec; }
    vector<vector<string> >& getDataVector() { return data; }
    vector<string> getNextLine() { i++; return data[i]; }
    int getMachineIdx() { return machineIdx; }
    int getSocketIdx() { return socketIdx; }
    const vector<int> &getSampleIdx() { return sampleIdx; }
    const vector<string> &getSampleNames() { return sampleNames; }
    int getNextSampleIdx() { sIdx++; return sampleIdx[sIdx]; }
    string getNextSampleName() { nIdx++; return sampleNames[nIdx]; }
    int getRuntimeIdx() { return runtimeIdx; }
    int getMeasurementIdx() { return measurementIdx; }
    int getTypeIdx() { return typeIdx; }
    string operator() (int row, int col) { return data[row][col]; }
};

#endif

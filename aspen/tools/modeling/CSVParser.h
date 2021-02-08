// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#ifndef CSVPARSER_HPP
#define CSVPARSER_HPP
#include<iostream>
#include<vector>
#include<string>
#include<fstream>
using namespace std;

class CSVParser
{
  protected:
    string Path;
    string fileName;
    unsigned xidx, yidx;
    vector<string> formatVec;
    vector<vector<string> > data;
  public:
    vector<string> getLineVec(istream&);
    CSVParser(string path, bool header);
    ~CSVParser() {}
    vector<vector<string>>& getData() { return data; }
    string operator() (int row, int col) { return data[row][col]; }
    string getNextElement();
};

#endif

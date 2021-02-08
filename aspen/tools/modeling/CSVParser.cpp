// Copyright 2013-2015 UT-Battelle, LLC.  See LICENSE.txt for more information.
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "CSVParser.h"
using namespace std;

vector<string> 
CSVParser::getLineVec(istream &cfstream)
{
    string line;
    vector<string> lineVec;
    getline(cfstream,line,'\n');
    stringstream ss(line);
    string tok;
    while (getline(ss,tok,','))
        lineVec.push_back(tok);
    return lineVec;
}

CSVParser::CSVParser(string path, bool header) : Path(path)
{ 
    ifstream csvFStream(path);
    while (true)
    {
        vector<string> tmp = getLineVec(csvFStream);
        if (csvFStream.eof() || tmp.size() == 0)
            break;
        if (tmp.empty())
            continue;
        if (header && formatVec.empty())
            formatVec = tmp;
        else 
            data.push_back(tmp);
    }
    xidx = 0; 
    yidx = 0;
}

string 
CSVParser::getNextElement()
{
    if (xidx == data.size())
        return "";

    string tmp = data[xidx][yidx];

    if (yidx == data[0].size()-1)
    {
        yidx = 0;
        xidx++;
    }
    else
        yidx++;
    return tmp;
}

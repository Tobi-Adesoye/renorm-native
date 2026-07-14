#pragma once

#include <string>

namespace renorm
{

struct BenchmarkResult
{
    std::string algorithm;

    int graphNodes=0;
    int graphEdges=0;

    int exploredNodes=0;

    int exploredEdges=0;

    int pathLength=0;

    double compileTimeMS=0.0;

    double executionTimeMS=0.0;

    double totalTimeMS=0.0;

    long long matrixOperations=0;

    long long floatingPointOperations=0;

    double memoryMB=0.0;

    bool solved=false;
};

}
#pragma once

#include "renorm/benchmark/bench_metrics.h"

#include <fstream>
#include <string>

namespace renorm
{
namespace benchmark
{

class CSVWriter
{
public:

    CSVWriter();

    ~CSVWriter();

    bool open(
        const std::string& filename);

    void close();

    //----------------------------------------------------------
    // Force buffered data onto disk.
    //----------------------------------------------------------

    void flush();

    //----------------------------------------------------------
    // CSV
    //----------------------------------------------------------

    void write_header();

    void write_row(
        const BenchMetrics& metrics);

private:

    std::ofstream m_file;
};

}
}
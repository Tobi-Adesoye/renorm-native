#include "renorm/benchmark/csv_writer.h"

namespace renorm
{
namespace benchmark
{

CSVWriter::CSVWriter()
{
}

CSVWriter::~CSVWriter()
{
    close();
}

bool CSVWriter::open(
    const std::string& filename)
{
    m_file.open(filename);

    return m_file.is_open();
}

void CSVWriter::close()
{
    if (m_file.is_open())
    {
        m_file.flush();
        m_file.close();
    }
}

void CSVWriter::flush()
{
    if (m_file.is_open())
    {
        m_file.flush();
    }
}

void CSVWriter::write_header()
{
    if (!m_file.is_open())
        return;

    m_file
        << "Algorithm,"
        << "Graph,"
        << "Nodes,"
        << "Edges,"
        << "CompileMs,"
        << "ExecutionMs,"
        << "TotalMs,"
        << "Visited,"
        << "PathLength,"
        << "Reduction,"
        << "FLOPs,"
        << "Blocks,"
        << "NodesPerSecond,"
        << "EdgesPerSecond,"
        << "Speedup,"
        << "Confidence,"
        << "HeuristicScore,"
        << "FrontierQuality,"
        << "GraphComplexity,"
        << "PredictionConfidence,"
        << "DecisionCost,"
        << "ExpansionReduction,"
        << "ExecutionPriority,"
        << "ExpansionPolicy\n";
}

void CSVWriter::write_row(
    const BenchMetrics& m)
{
    if (!m_file.is_open())
        return;

    m_file
        << m.algorithm << ","
        << m.graphFamily << ","
        << m.nodes << ","
        << m.edges << ","
        << m.compileTimeMs << ","
        << m.executionTimeMs << ","
        << m.totalTimeMs << ","
        << m.visitedNodes << ","
        << m.pathLength << ","
        << m.reductionPercent << ","
        << m.floatingPointOperations << ","
        << m.blocksExecuted << ","
        << m.nodesPerSecond << ","
        << m.edgesPerSecond << ","
        << m.speedup << ','
        << m.confidence << ','
        << m.heuristicScore << ','
        << m.frontierQuality << ','
        << m.graphComplexity << ','
        << m.predictionConfidence << ','
        << m.decisionCost << ','
        << m.expansionReduction << ','
        << m.executionPriority << ','
        << m.expansionPolicy
        << "\n";

    //
    // Flush every benchmark row.
    //
    // If the benchmark later crashes or is interrupted,
    // all previous results are already safely written.
    //

    m_file.flush();
}

} // namespace benchmark
} // namespace renorm
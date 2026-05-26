#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

class ThreadPool;

/**
 * @brief Partial minimizer graph built by one worker thread.
 *
 * `kmers` stores this thread's minimizer occurrences in generation order, which follows genomic position.
 * `idx` groups those occurrences by minimizer hash: each value in `idx` is an index into `kmers`.
 */
struct ThreadGraph {
    std::vector<Kmer> kmers; // Needs push_back() support (exact kmer count is not known)
    NoInitArray<std::uint64_t> idx;
    NoInitArray<Node> nodes;
    NoInitArray<Edge> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::size_t n_kmers = 0;
    std::size_t start_assembly = 0;
};

/**
 * @brief Emit a message through Python's logging module.
 *
 * @param message Message to log.
 * @param level Logging level: `debug`, `info`, `warning`, `error`, or `critical`.
 */
void log_python(
    const std::string& message,
    const std::string& level = "info"
);

/**
 * @brief Merge thread-local minimizer graphs into a single graph.
 *
 * @param graphs Thread-local graphs produced during parallel graph construction.
 * @param n_assemblies Total number of input assemblies.
 * @param pool Thread pool used for parallel merge steps.
 * @return Fully merged minimizer graph.
 */
Graph merge_thread_graphs(
    std::vector<ThreadGraph>& graphs,
    std::size_t n_assemblies,
    ThreadPool& pool
);

Graph filter_kmers(
    const Kmer* kmers,
    const std::uint64_t* idx,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
);

} // namespace seqwin

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

/**
 * ThreadPool.parallel_for(n_items, fn) splits [0, n_items) into contiguous
 * chunks and calls fn(start, end, worker_id) once per non-empty chunk.
 * Callers should let parallel_for() choose chunk boundaries.
 */
class ThreadPool;

void log_python(
    const std::string& message,
    const std::string& level = "info"
);

Graph merge_thread_graphs(
    std::vector<Graph>& graphs,
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

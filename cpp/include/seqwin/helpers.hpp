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

struct ThreadResult {
    std::vector<Kmer> kmers; // Ordered by genomic positions (the original order)
    std::vector<std::uint64_t> idx; // Original k-mer indices, grouped by hash
    std::vector<Node> nodes; // Unsorted; start and stop point to k-mer groups in idx
    std::vector<Edge> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::size_t n_kmers = 0;
    std::size_t start_assembly = 0;
};

struct FilterResult {
    std::vector<Kmer> kmers;
    std::vector<std::uint64_t> idx;
    std::vector<Node> nodes;
};

void log_python(
    const std::string& message,
    const std::string& level = "info"
);

BuildResult merge_thread_results(
    std::vector<ThreadResult>& results,
    std::size_t n_assemblies,
    ThreadPool& pool
);

FilterResult filter_kmers(
    const Kmer* kmers,
    const std::uint64_t* idx,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
);

} // namespace seqwin

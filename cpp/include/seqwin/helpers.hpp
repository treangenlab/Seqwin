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

struct ThreadNode {
    std::uint64_t hash;
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    std::uint64_t thread_id;
    std::uint64_t start;
    std::uint64_t stop;
};

struct ThreadResult {
    std::vector<Kmer> kmers; // Ordered by genomic positions (the original order)
    std::vector<std::uint64_t> idx; // Original k-mer indices, grouped by hash
    std::vector<ThreadNode> nodes; // Unsorted; start and stop point to k-mer groups in idx
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::uint64_t n_kmers = 0;
    std::size_t start_assembly = 0;
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

} // namespace seqwin

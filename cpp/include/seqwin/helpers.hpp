#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

struct ThreadNode {
    std::uint64_t hash;
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    std::uint64_t thread_id;
    std::uint64_t start;
    std::uint64_t stop;
};

struct ThreadResult {
    std::vector<Kmer> kmers;
    std::vector<ThreadNode> nodes;
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::vector<std::uint64_t> idx; // Flat local k-mer indices grouped by local node
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
    std::size_t n_workers
);

} // namespace seqwin

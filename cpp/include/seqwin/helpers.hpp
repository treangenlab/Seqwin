#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

struct ThreadResult {
    std::vector<std::uint8_t> kmers;
    std::vector<std::uint64_t> nodes; // 4-D array: {n_tar, n_neg, last_seen_assembly, idx_ptr}; idx_ptr points into all_idx
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::vector<std::vector<std::uint64_t>> all_idx; // K-mer indices of each node
    std::uint64_t n_kmers = 0;
    std::size_t start_assembly = 0;
};

BuildResult merge_thread_results(
    std::vector<ThreadResult>& results,
    std::size_t n_assemblies,
    std::size_t n_workers
);

} // namespace seqwin

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/no_init_array.hpp"

namespace seqwin {

struct Kmer {
    std::uint32_t pos;
    std::uint16_t record_idx;
    std::uint16_t assembly_idx;
};

struct Node {
    std::uint64_t hash;
    std::uint64_t start;
    std::uint64_t stop;
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    double penalty; // Used as thread_id before merging from different threads
};

struct Edge {
    std::uint64_t first;
    std::uint64_t second;
    std::uint64_t weight;
};

struct Graph {
    NoInitArray<Kmer> kmers;
    NoInitArray<std::uint64_t> idx;
    NoInitArray<Node> nodes;
    NoInitArray<Edge> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
};

Graph build(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu
);

} // namespace seqwin

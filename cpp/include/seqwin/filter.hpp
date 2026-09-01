#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

using Subgraphs = std::vector<std::vector<std::uint64_t>>;

void get_penalty(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t n_cpu = 1
);

Graph filter_kmers(
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
);

std::pair<Subgraphs, std::vector<std::uint64_t>> get_subgraphs(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::vector<std::uint64_t>& seeds,
    double penalty_th,
    std::size_t min_nodes,
    std::size_t max_nodes
);

} // namespace seqwin

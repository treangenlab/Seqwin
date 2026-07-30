#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

void get_penalty(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const std::size_t* record_offsets,
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

} // namespace seqwin

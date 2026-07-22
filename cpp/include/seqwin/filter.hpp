#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

Graph filter_kmers(
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
);

} // namespace seqwin

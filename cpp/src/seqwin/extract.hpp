#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "seqwin/filter.hpp"

namespace seqwin::internal {

struct ConsecutiveKmers {
    SeqLocation location;
    bool is_target;
    std::vector<std::uint64_t> order;
};

std::optional<Signature> extract_worker(
    std::size_t subgraph_idx,
    const std::vector<std::size_t>& subgraph,
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    std::size_t total_tar
);

void fetch_signature_sequences(
    std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    std::size_t n_cpu
);

} // namespace seqwin::internal

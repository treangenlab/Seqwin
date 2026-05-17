#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace seqwin {

struct Kmer {
    std::uint32_t pos;
    std::uint16_t record_idx;
    std::uint16_t assembly_idx;
};

struct Node {
    std::uint64_t hash;
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    double penalty;
    std::uint64_t start;
    std::uint64_t stop;
};

struct BuildResult {
    std::vector<Kmer> kmers;
    std::vector<std::uint64_t> idx;
    std::vector<Node> nodes;
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
};

BuildResult build_impl(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu
);

} // namespace seqwin

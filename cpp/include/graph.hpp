#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct IndexlrResult {
    std::vector<std::uint8_t> kmers;
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
};

IndexlrResult indexlr_impl(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets
);

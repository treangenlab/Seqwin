#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

using Subgraphs = std::vector<std::vector<std::uint64_t>>;

/**
 * @brief Part of Seqwin configurations.
 */
struct FilterConfig {
    std::optional<double> penalty_th;
    double stringency;
    double penalty_th_cap;
    double edge_w_th_mul;
    std::size_t windowsize;
    std::size_t min_len;
    std::optional<std::size_t> max_len;
    std::size_t min_nodes_floor;
    std::optional<std::size_t> max_nodes_cap;
    std::size_t n_cpu;
};

/**
 * @brief Includes filtered graph arrays, subgraphs and calculated values.
 */
struct FilterResult {
    NoInitArray<Kmer> kmers;
    NoInitArray<Node> nodes;
    std::vector<Edge> edges;
    Subgraphs subgraphs;
    std::size_t total_tar;
    std::size_t total_neg;
    double e_absence_tar;
    double e_presence_neg;
    double penalty_th;
    double edge_weight_th;
    std::size_t min_nodes;
    std::optional<std::size_t> max_nodes;
};

/**
 * @brief Filter the minimizer graph and extract low-penalty subgraphs.
 */
FilterResult filter(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
);

} // namespace seqwin

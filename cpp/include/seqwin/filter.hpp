#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "seqwin/graph.hpp"
#include "seqwin/signature.hpp"

namespace seqwin {

/** @brief Represented by indices into `FilteredGraph.nodes`. */
using Subgraph = std::vector<std::size_t>;

/**
 * @brief Part of Seqwin configurations.
 */
struct FilterConfig {
    std::size_t kmerlen;
    std::size_t windowsize;
    std::optional<double> penalty_th;
    double stringency;
    std::size_t min_len;
    std::optional<std::size_t> max_len;
    double penalty_th_cap;
    double edge_w_th_mul;
    std::size_t min_nodes_floor;
    std::optional<std::size_t> max_nodes_cap;
    double consec_kmer_mul;
    std::size_t n_cpu;
};

/**
 * @brief Includes filtered graph arrays, low-penalty subgraphs, and calculated values.
 *
 * Filtered nodes and edges follow their original order.
 */
struct FilteredGraph {
    NoInitArray<Node> nodes;
    NoInitArray<Edge> edges;
    std::vector<Subgraph> subgraphs;
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
 * @brief Filter the minimizer graph and extract signatures from low-penalty subgraphs.
 */
std::pair<FilteredGraph, std::vector<Signature>> filter(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const std::vector<std::string>& assembly_paths,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
);

} // namespace seqwin

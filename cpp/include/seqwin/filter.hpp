#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "seqwin/graph_view.hpp"
#include "seqwin/signature.hpp"

namespace seqwin {

/**
 * @brief Represented by node indices of the original graph.
 */
using Subgraph = std::vector<std::size_t>;

/**
 * @brief Part of Seqwin configurations.
 */
struct FilterConfig {
    std::size_t kmerlen;
    std::size_t windowsize;
    double tar_weight;
    double neg_rank;
    std::size_t min_len;
    std::optional<std::size_t> max_len;
    double edge_w_th_mul;
    std::optional<std::size_t> n_tar_th;
    std::optional<std::size_t> n_neg_th;
    std::size_t min_nodes_floor;
    std::optional<std::size_t> max_nodes_cap;
    double consec_kmer_mul;
    std::size_t n_cpu;
    std::optional<std::string> fasta_path;
    std::optional<std::string> csv_path;
};

/**
 * @brief Graph node retained by edge filtering.
 */
struct FilteredNode {
    /** Node index in the original graph. */
    std::size_t idx;
    /** Number of target assemblies containing this node's k-mer. */
    std::uint32_t n_tar;
    /** Number of non-target assemblies containing this node's k-mer. */
    std::uint32_t n_neg;
};

/**
 * @brief Includes filtered graph arrays, subgraphs, and calculated values.
 *
 * Filtered nodes and edges follow their original order.
 */
struct FilteredGraph {
    /** Nodes retained by edge filtering. */
    NoInitArray<FilteredNode> nodes;
    /** Edges passing weight and target-support thresholds. */
    std::vector<Edge> edges;
    /** Subgraphs represented by node indices of the original graph. */
    std::vector<Subgraph> subgraphs;
    /** Number of target assemblies. */
    std::size_t total_tar;
    /** Number of non-target assemblies. */
    std::size_t total_neg;
    /** Minimum mean target prevalence in a subgraph. */
    std::size_t n_tar_th;
    /** Maximum mean non-target prevalence in a subgraph. */
    std::size_t n_neg_th;
    /** Graph edge weight threshold. */
    double edge_weight_th;
    /** Minimum number of nodes for a subgraph. */
    std::size_t min_nodes;
    /** Maximum number of nodes for a subgraph. */
    std::optional<std::size_t> max_nodes;
};

/**
 * @brief Filter the k-mer graph and extract signatures from subgraphs.
 */
std::pair<FilteredGraph, std::vector<Signature>> filter(
    const GraphView& graph,
    const std::vector<std::string>& assembly_paths,
    ArrayView<bool> is_targets,
    const FilterConfig& config
);

} // namespace seqwin

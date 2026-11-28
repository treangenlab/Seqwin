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
    double edge_weight_th_mul;
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
 *
 * A node is retained if it is an endpoint of at least one retained edge.
 */
struct RetainedNode {
    /** Node index in the original graph. */
    std::size_t idx;
    /** Target prevalence (number of target assemblies containing this node's k-mer). */
    std::uint32_t n_tar;
    /** Non-target prevalence (number of non-target assemblies containing this node's k-mer). */
    std::uint32_t n_neg;
};

/**
 * @brief Retained graph nodes and edges, generated subgraphs, and filtering thresholds.
 *
 * Retained nodes and edges follow their original order.
 */
struct RetainedGraph {
    /** Nodes that are endpoints of at least one retained edge. */
    NoInitArray<RetainedNode> nodes;
    /** Edges with sufficient weight and target prevalence at either endpoint. */
    std::vector<Edge> edges;
    /** Subgraphs represented by node indices of the original graph. */
    std::vector<Subgraph> subgraphs;
    /** Number of target assemblies. */
    std::size_t total_tar;
    /** Number of non-target assemblies. */
    std::size_t total_neg;
    /** Graph edge weight threshold. */
    double edge_weight_th;
    /** Minimum mean target prevalence of a subgraph. */
    std::size_t n_tar_th;
    /** Maximum mean non-target prevalence of a subgraph. */
    std::size_t n_neg_th;
    /** Minimum number of nodes for a subgraph. */
    std::size_t min_nodes;
    /** Maximum number of nodes for a subgraph. */
    std::optional<std::size_t> max_nodes;
};

/**
 * @brief Filter the k-mer graph and extract signatures from subgraphs.
 */
std::pair<RetainedGraph, std::vector<Signature>> filter(
    const GraphView& graph,
    const std::vector<std::string>& assembly_paths,
    ArrayView<bool> is_targets,
    const FilterConfig& config
);

} // namespace seqwin

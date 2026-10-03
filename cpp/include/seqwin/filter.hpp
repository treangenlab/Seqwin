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

/**
 * @brief Represented by indices into `FilteredGraph.nodes`.
 */
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
 * @brief Graph node retained by edge filtering.
 */
struct FilteredNode {
    /** Index into the original `Graph.nodes` array. */
    std::size_t idx;
    /** Number of target assemblies containing this node's minimizer. */
    std::uint32_t n_tar;
    /** Number of non-target assemblies containing this node's minimizer. */
    std::uint32_t n_neg;
    /** Node penalty score. */
    double penalty;
};

/**
 * @brief Includes filtered graph arrays, low-penalty subgraphs, and calculated values.
 *
 * Filtered nodes and edges follow their original order.
 */
struct FilteredGraph {
    /** Nodes retained by edge filtering. */
    NoInitArray<FilteredNode> nodes;
    /** Edges passed the weight threshold. Endpoints are indices into the retained nodes. */
    NoInitArray<Edge> edges;
    /** Low-penalty subgraphs represented by indices of retained nodes. */
    std::vector<Subgraph> subgraphs;
    /** Number of target assemblies. */
    std::size_t total_tar;
    /** Number of non-target assemblies. */
    std::size_t total_neg;
    /** Expected k-mer absence in target assemblies. */
    std::optional<double> e_absence_tar;
    /** Expected k-mer presence in non-target assemblies. */
    std::optional<double> e_presence_neg;
    /** Node penalty threshold (user input or auto-computed). */
    double penalty_th;
    /** Graph edge weight threshold. */
    double edge_weight_th;
    /** Minimum number of nodes for a low-penalty subgraph. */
    std::size_t min_nodes;
    /** Maximum number of nodes for a low-penalty subgraph. */
    std::optional<std::size_t> max_nodes;
};

/**
 * @brief Filter the minimizer graph and extract signatures from low-penalty subgraphs.
 */
std::pair<FilteredGraph, std::vector<Signature>> filter(
    const Kmer* kmers,
    std::size_t n_kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const std::size_t* assembly_nodes,
    std::size_t n_assembly_nodes,
    const std::size_t* node_offsets,
    std::size_t n_node_offsets,
    const std::vector<std::string>& assembly_paths,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
);

} // namespace seqwin

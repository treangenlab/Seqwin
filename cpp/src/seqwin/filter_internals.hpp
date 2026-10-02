#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "seqwin/filter.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {

/**
 * @brief Dense assembly occurrence counts for every graph node.
 *
 * It stores whichever assembly group (target/non-target) is smaller using the
 * narrowest safe counter type. Consumers access target counts without needing
 * to know which group is stored.
 */
class TargetCounts {
public:
    TargetCounts(NoInitArray<std::uint8_t>&& counts, bool counts_targets)
        : counts_(std::move(counts))
        , counts_targets_(counts_targets)
    {}
    TargetCounts(NoInitArray<std::uint16_t>&& counts, bool counts_targets)
        : counts_(std::move(counts))
        , counts_targets_(counts_targets)
    {}
    TargetCounts(NoInitArray<std::uint32_t>&& counts, bool counts_targets)
        : counts_(std::move(counts))
        , counts_targets_(counts_targets)
    {}

    /** Dispatch once on the counter type, then invoke `fn` with an `n_tar` accessor. */
    template <typename Fn> void visit(const Node* nodes, Fn&& fn) const
    {
        std::visit(
            [&](const auto& counts) {
                fn([&](std::size_t node_idx) -> std::size_t {
                    const std::size_t stored = counts[node_idx];
                    if (counts_targets_) {
                        return stored;
                    }
                    if (stored > nodes[node_idx].prevalence) {
                        throw std::invalid_argument("Node n_neg exceeds prevalence");
                    }
                    return nodes[node_idx].prevalence - stored;
                });
            },
            counts_
        );
    }

private:
    std::variant<NoInitArray<std::uint8_t>, NoInitArray<std::uint16_t>, NoInitArray<std::uint32_t>>
        counts_;
    bool counts_targets_;
};

/**
 * @brief Undirected graph stored as contiguous adjacency lists.
 */
class GraphTopology {
public:
    class NeighborRange {
    public:
        using Iterator = std::vector<std::size_t>::const_iterator;

        NeighborRange(Iterator begin, Iterator end)
            : begin_(begin)
            , end_(end)
        {}

        Iterator begin() const { return begin_; }
        Iterator end() const { return end_; }

    private:
        Iterator begin_;
        Iterator end_;
    };

    GraphTopology(std::size_t n_nodes, const NoInitArray<Edge>& edges)
        : offsets_(n_nodes + 1, 0)
    {
        for (const auto& edge : edges) {
            if (edge.first >= n_nodes || edge.second >= n_nodes) {
                throw std::invalid_argument("Edge endpoint does not correspond to a node");
            }
            ++offsets_[edge.first + 1];
            ++offsets_[edge.second + 1];
        }

        for (std::size_t i = 1; i < offsets_.size(); ++i) {
            offsets_[i] += offsets_[i - 1];
        }
        neighbors_.resize(offsets_.back());
        auto cursors = offsets_;
        for (const auto& edge : edges) {
            neighbors_[cursors[edge.first]++] = edge.second;
            neighbors_[cursors[edge.second]++] = edge.first;
        }
    }

    NeighborRange neighbors(std::size_t node_index) const
    {
        return {
            neighbors_.cbegin() + offsets_[node_index],
            neighbors_.cbegin() + offsets_[node_index + 1]
        };
    }

private:
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbors_;
};

/**
 * @brief For each graph node, count its occurrences in target assemblies (`n_tar`).
 *
 * Also calculate `total_tar`, `total_neg`, `e_absence_tar` and `e_presence_neg`
 * and add them to `FilteredGraph`.
 */
std::pair<FilteredGraph, TargetCounts> collect_target_counts(
    const Node* nodes,
    std::size_t n_nodes,
    const std::size_t* assembly_nodes,
    std::size_t n_assembly_nodes,
    const std::size_t* node_offsets,
    std::size_t n_node_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    ThreadPool& pool
);

/**
 * @brief Remove low-weight edges and isolated nodes.
 * Filtered nodes and edges are stored directly in `filtered`.
 */
void prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    double edge_weight_th,
    const TargetCounts& target_counts,
    FilteredGraph& filtered,
    ThreadPool& pool
);

/**
 * @brief Grow disjoint low-penalty subgraphs from eligible seeds.
 * Generated subgraphs are stored directly in `filtered`.
 */
void get_subgraphs(
    const NoInitArray<FilteredNode>& nodes,
    const NoInitArray<Edge>& edges,
    double penalty_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilteredGraph& filtered
);

/**
 * @brief Extract signatures from low-penalty subgraphs.
 */
std::vector<Signature> extract_signatures(
    const std::vector<Subgraph>& subgraphs,
    const NoInitArray<FilteredNode>& filtered_nodes,
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const std::vector<std::string>& assembly_paths,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    double consec_kmer_mul,
    std::size_t total_tar,
    ThreadPool& pool
);

} // namespace seqwin::internal

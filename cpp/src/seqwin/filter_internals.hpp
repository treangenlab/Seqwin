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
 * @brief Undirected graph stored as CSR adjacency lists.
 */
class GraphTopology {
public:
    class NeighborRange {
    public:
        using Iterator = const std::size_t*;

        NeighborRange(Iterator begin, Iterator end)
            : begin_(begin)
            , end_(end)
        {}

        Iterator begin() const noexcept { return begin_; }
        Iterator end() const noexcept { return end_; }

    private:
        Iterator begin_;
        Iterator end_;
    };

    GraphTopology(
        std::vector<std::size_t>&& offsets,
        NoInitArray<std::size_t>&& neighbors
    )
        : offsets_(std::move(offsets))
        , neighbors_(std::move(neighbors))
    {
        if (offsets_.empty()) {
            throw std::invalid_argument("Graph topology offsets must not be empty");
        }
        if (offsets_.front() != 0) {
            throw std::invalid_argument("Graph topology offsets must start with 0");
        }
        if (offsets_.back() != neighbors_.size()) {
            throw std::invalid_argument(
                "Final graph topology offset must equal the number of neighbors"
            );
        }
    }

    std::size_t size() const noexcept
    {
        return offsets_.size() - 1;
    }

    NeighborRange neighbors(std::size_t node_index) const
    {
        if (neighbors_.empty()) {
            return {nullptr, nullptr};
        }
        return {
            neighbors_.data() + offsets_[node_index],
            neighbors_.data() + offsets_[node_index + 1]
        };
    }

private:
    std::vector<std::size_t> offsets_;
    NoInitArray<std::size_t> neighbors_;
};

/**
 * @brief For each graph node, count its occurrences in target assemblies (`n_tar`).
 */
TargetCounts count_target_nodes(
    std::size_t n_nodes,
    const std::size_t* assembly_nodes,
    std::size_t n_assembly_nodes,
    const std::size_t* node_offsets,
    std::size_t n_node_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
);

/**
 * @brief For k-mers in target assemblies, calculate their expected absence
 * in target assemblies, and expected presence in non-target assemblies.
 */
std::pair<double, double> expected_presence(
    const Node* nodes,
    std::size_t n_nodes,
    const TargetCounts& target_counts,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
);

/**
 * @brief Remove low-weight edges and isolated nodes, calculate penalty scores,
 * and construct CSR adjacency lists of the retained graph.
 *
 * Retained nodes and edges are stored directly in `filtered`.
 */
GraphTopology prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const TargetCounts& target_counts,
    std::size_t total_tar,
    std::size_t total_neg,
    double edge_weight_th,
    FilteredGraph& filtered,
    ThreadPool& pool
);

/**
 * @brief Grow disjoint low-penalty subgraphs from eligible seeds.
 * Generated subgraphs are stored directly in `filtered`.
 */
void get_subgraphs(
    const GraphTopology& graph,
    const NoInitArray<FilteredNode>& nodes,
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
    std::size_t n_kmers,
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

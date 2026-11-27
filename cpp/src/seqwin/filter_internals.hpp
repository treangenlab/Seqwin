#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "seqwin/filter.hpp"
#include "seqwin/graph_view.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {

/**
 * @brief Memory-efficient storage for accessing node target prevalence.
 *
 * Store the prevalence of each node in the smaller assembly group using
 * the narrowest safe counter type. Derive target prevalence from total
 * and non-target prevalence when necessary.
 */
class TargetCounts {
public:
    template <typename Counter>
    TargetCounts(
        NoInitArray<Counter>&& counts,
        bool counts_targets,
        ArrayView<std::uint32_t> prevalences
    )
        : counts_(std::move(counts))
        , counts_targets_(counts_targets)
        , prevalences_(prevalences)
    {}

    /**
     * @brief Accept a function `fn` that needs access to node `n_tar`.
     *
     * The caller provides `fn`, which takes the `n_tar` accessor as its sole argument.
     * The accessor is selected once based on the counter type and stored assembly group.
     */
    template <typename Fn>
    void with(Fn&& fn) const {
        std::visit(
            [&](const auto& counts) {
                if (counts_targets_) {
                    fn([&](std::size_t node_idx) -> std::size_t {
                        return counts[node_idx];
                    });
                } else {
                    fn([&](std::size_t node_idx) -> std::size_t {
                        const std::size_t n_neg = counts[node_idx];
                        if (n_neg > prevalences_[node_idx]) {
                            throw std::invalid_argument("Node n_neg exceeds prevalence");
                        }
                        return prevalences_[node_idx] - n_neg;
                    });
                }
            },
            counts_
        );
    }

    void reset() noexcept {
        std::visit([](auto& counts) { counts.reset(); }, counts_);
    }

private:
    std::variant<
        NoInitArray<std::uint8_t>,
        NoInitArray<std::uint16_t>,
        NoInitArray<std::uint32_t>
    > counts_;
    bool counts_targets_;
    ArrayView<std::uint32_t> prevalences_;
};

/**
 * @brief Undirected graph stored as CSR adjacency lists.
 * Nodes are represented by compact indices.
 */
class GraphTopology {
public:
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

    std::size_t size() const noexcept {
        return offsets_.empty() ? 0 : offsets_.size() - 1;
    }

    ArrayView<std::size_t> neighbors(std::size_t node_index) const {
        if (neighbors_.empty()) {
            return {nullptr, 0};
        }
        return {
            neighbors_.data() + offsets_[node_index],
            offsets_[node_index + 1] - offsets_[node_index]
        };
    }

    void reset() noexcept {
        std::vector<std::size_t>().swap(offsets_);
        neighbors_.reset();
    }

private:
    std::vector<std::size_t> offsets_;
    NoInitArray<std::size_t> neighbors_;
};

/**
 * @brief Calculate the target prevalence of each graph node.
 *
 * Count nodes in the smaller assembly group using the narrowest safe
 * counter type to reduce memory usage.
 */
TargetCounts count_target_nodes(
    ArrayView<std::uint32_t> prevalences,
    ArrayView<std::size_t> assembly_nodes,
    ArrayView<std::size_t> node_offsets,
    ArrayView<bool> is_targets,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
);

/**
 * @brief Filter edges, determine the target prevalence threshold,
 * and construct CSR adjacency lists.
 *
 * An edge is retained only if its weight and the target prevalence
 * of at least one endpoint exceed `edge_weight_th`.
 *
 * If `n_tar_th` is not provided, interpolate between the mean target
 * presence of retained nodes and `total_tar`, using `tar_weight`.
 *
 * Retained nodes, edges, and the threshold are stored in `filtered`.
 */
GraphTopology prune_graph(
    ArrayView<std::uint32_t> prevalences,
    ArrayView<Edge> edges,
    const TargetCounts& target_counts,
    std::size_t total_tar,
    std::size_t total_neg,
    double edge_weight_th,
    double tar_weight,
    std::optional<std::size_t> n_tar_th,
    FilteredGraph& filtered,
    ThreadPool& pool
);

/**
 * @brief Greedily grow disjoint subgraphs from eligible seeds.
 *
 * A seed must satisfy both target and non-target prevalence thresholds. During expansion,
 * prioritize nodes with high target prevalence and low non-target prevalence.
 *
 * A node is accepted only if it keeps the subgraph's mean prevalences within the limits.
 *
 * Subgraphs satisfying the size limits are stored in `filtered`.
 */
void grow_subgraphs(
    const GraphTopology& graph,
    const NoInitArray<FilteredNode>& nodes,
    std::size_t total_tar,
    std::int64_t n_tar_th,
    std::int64_t n_neg_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilteredGraph& filtered
);

/**
 * @brief Extract a signature from each subgraph in parallel.
 *
 * Return valid signatures in subgraph order. Write signature sequences
 * and metadata to FASTA and CSV if output paths are provided.
 */
std::vector<Signature> extract_signatures(
    const std::vector<Subgraph>& subgraphs,
    ArrayView<Kmer> kmers,
    ArrayView<std::size_t> offsets,
    ArrayView<std::uint32_t> record_offsets,
    const std::vector<std::string>& assembly_paths,
    ArrayView<bool> is_targets,
    std::size_t total_tar,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    double consec_kmer_mul,
    const std::optional<std::string>& fasta_path,
    const std::optional<std::string>& csv_path,
    ThreadPool& pool
);

} // namespace seqwin::internal

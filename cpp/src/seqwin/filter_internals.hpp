#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "seqwin/filter.hpp"

namespace seqwin::internal {

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

    GraphTopology(const std::vector<Node>& nodes, const std::vector<Edge>& edges)
        : offsets_(nodes.size() + 1, 0)
    {
        for (const auto& edge : edges) {
            if (edge.first >= nodes.size() || edge.second >= nodes.size()) {
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
 * @brief Nodes and edges follow their original order.
*/
struct PrunedGraph {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
};

/**
 * @brief Node ranges are rewritten to index the compacted `kmers`.
 * `edges` contains only edges with two selected endpoints.
 */
struct CompactedGraph {
    NoInitArray<Kmer> kmers;
    NoInitArray<Node> nodes;
    std::vector<Edge> edges;
};

/**
 * @brief Calculate `n_tar`, `n_neg` and `penalty` for each node,
 * and update `nodes` in place.
 *
 * Also calculate `total_tar`, `total_neg`, `e_absence_tar` and `e_presence_neg`,
 * and add them to `FilterResult`.
 */
FilterResult get_penalty(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t n_cpu
);

/**
 * @brief Remove low-weight edges and isolated nodes.
 */
PrunedGraph prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    double edge_weight_th
);

/**
 * @brief Grow disjoint low-penalty subgraphs from eligible seeds.
 *
 * @return Subgraphs represented by node hashes;
 * indices of all accepted nodes in `PrunedGraph.nodes`.
 */
std::pair<Subgraphs, std::vector<std::size_t>> get_subgraphs(
    const std::vector<Node>& nodes,
    const std::vector<Edge>& edges,
    double penalty_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes
);

/**
 * @brief Restrict a pruned graph to nodes used by accepted subgraphs.
 */
CompactedGraph compact_graph(
    const Kmer* kmers,
    const std::vector<Node>& nodes,
    const std::vector<Edge>& edges,
    std::vector<std::size_t> used_nodes
);

} // namespace seqwin::internal

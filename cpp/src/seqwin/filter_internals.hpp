#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "seqwin/filter.hpp"

namespace seqwin::internal {

/** Undirected graph stored as contiguous adjacency lists. */
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
        ankerl::unordered_dense::map<std::uint64_t, std::size_t> node_indices;
        node_indices.reserve(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            node_indices.emplace(nodes[i].hash, i);
        }

        std::vector<std::pair<std::size_t, std::size_t>> endpoints;
        endpoints.reserve(edges.size());
        for (const auto& edge : edges) {
            const auto first = node_indices.find(edge.first);
            const auto second = node_indices.find(edge.second);
            if (first == node_indices.end() || second == node_indices.end()) {
                throw std::invalid_argument("Edge endpoint does not correspond to a node");
            }
            endpoints.emplace_back(first->second, second->second);
            ++offsets_[first->second + 1];
            ++offsets_[second->second + 1];
        }

        for (std::size_t i = 1; i < offsets_.size(); ++i) {
            offsets_[i] += offsets_[i - 1];
        }
        neighbors_.resize(offsets_.back());
        auto cursors = offsets_;
        for (const auto& endpoint : endpoints) {
            neighbors_[cursors[endpoint.first]++] = endpoint.second;
            neighbors_[cursors[endpoint.second]++] = endpoint.first;
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
 * @brief Nodes and edges follow their original order (sorted by hash).
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
 * @brief Populate node target counts and penalty scores in place,
 * and calculate `total_tar` and `total_neg`.
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
    std::size_t max_nodes
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

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "seqwin/graph.hpp"

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

    GraphTopology(
        const Node* nodes,
        std::size_t n_nodes,
        const Edge* edges,
        std::size_t n_edges
    )
        : offsets_(n_nodes + 1, 0)
    {
        node_indices_.reserve(n_nodes);
        for (std::size_t i = 0; i < n_nodes; ++i) {
            node_indices_.emplace(nodes[i].hash, i);
        }

        std::vector<std::pair<std::size_t, std::size_t>> endpoints;
        endpoints.reserve(n_edges);
        for (std::size_t i = 0; i < n_edges; ++i) {
            const auto first = node_indices_.find(edges[i].first);
            const auto second = node_indices_.find(edges[i].second);
            if (first == node_indices_.end() || second == node_indices_.end()) {
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

    std::size_t node_index(std::uint64_t hash) const
    {
        const auto it = node_indices_.find(hash);
        if (it == node_indices_.end()) {
            throw std::invalid_argument("Seed hash does not correspond to a node");
        }
        return it->second;
    }

    NeighborRange neighbors(std::size_t node_index) const
    {
        return {
            neighbors_.cbegin() + offsets_[node_index],
            neighbors_.cbegin() + offsets_[node_index + 1]
        };
    }

private:
    ankerl::unordered_dense::map<std::uint64_t, std::size_t> node_indices_;
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> neighbors_;
};

} // namespace seqwin::internal

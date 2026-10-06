#include "seqwin/filter_internals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include <ankerl/unordered_dense.h>

#include "seqwin/filter.hpp"
#include "seqwin/shared_internals.hpp"
#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {
namespace {

/**
 * @brief One endpoint occurrence of a retained edge.
 */
struct Endpoint {
    /** Original graph node represented by this endpoint. */
    std::size_t node;
    /**
     * Index in the flattened edge array.
     * The first endpoint has even index; the second endpoint has odd index.
     */
    std::size_t idx;
};

/**
 * @brief For each graph node, count its occurrences in target or non-target assemblies.
 *
 * Partitions the node index range across worker threads. Each worker scans target/non-target
 * assemblies and updates counts only for nodes in its assigned range. Because node indices in
 * each assembly slice of `assembly_nodes` is sorted, this range can be determined with binary
 * search on each assembly slice.
 */
template <typename Counter>
TargetCounts build_target_counts(
    std::size_t n_nodes,
    const std::size_t* assembly_nodes,
    const std::size_t* node_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    bool count_targets,
    ThreadPool& pool
) {
    NoInitArray<Counter> counts(n_nodes);
    pool.parallel_for_chunks(n_nodes, [&](std::size_t start, std::size_t end, std::size_t) {
        std::fill(
            counts.begin() + start,
            counts.begin() + end,
            Counter{0}
        );
        for (std::size_t assembly_idx = 0; assembly_idx < n_assemblies; ++assembly_idx) {
            if (is_targets[assembly_idx] != count_targets) {
                continue;
            }
            const auto* assembly_start = assembly_nodes + node_offsets[assembly_idx];
            const auto* assembly_end = assembly_nodes + node_offsets[assembly_idx + 1];
            const auto* first = std::lower_bound(assembly_start, assembly_end, start);
            const auto* last = std::lower_bound(first, assembly_end, end);
            for (auto it = first; it != last; ++it) {
                ++counts[*it];
            }
        }
    });
    return TargetCounts(std::move(counts), count_targets);
}

} // namespace

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
) {
    if (n_node_offsets != n_assemblies + 1) {
        throw std::invalid_argument("len(node_offsets) must equal len(is_targets) + 1");
    }
    if (n_node_offsets == 0 || node_offsets[0] != 0) {
        throw std::invalid_argument("node_offsets must start with 0");
    }
    for (std::size_t i = 0; i < n_assemblies; ++i) {
        if (node_offsets[i + 1] < node_offsets[i]) {
            throw std::invalid_argument("node_offsets must be nondecreasing");
        }
    }
    if (node_offsets[n_assemblies] != n_assembly_nodes) {
        throw std::invalid_argument("Final node offset must equal len(assembly_nodes)");
    }

    // The smaller assembly group is counted using the narrowest safe counter type
    const bool count_targets = total_tar <= total_neg;
    const std::size_t counted_assemblies = count_targets ? total_tar : total_neg;
    if (counted_assemblies <= std::numeric_limits<std::uint8_t>::max()) {
        return build_target_counts<std::uint8_t>(
            n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    if (counted_assemblies <= std::numeric_limits<std::uint16_t>::max()) {
        return build_target_counts<std::uint16_t>(
            n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    return build_target_counts<std::uint32_t>(
        n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
    );
}

std::pair<double, double> expected_presence(
    const Node* nodes,
    std::size_t n_nodes,
    const TargetCounts& target_counts,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
) {
    /**
     * Sums over all nodes, for penalty threshold calculation.
     * For all k-mers in targets, calculate their total presence in targets or non-targets.
     */
    struct NodeSums {
        std::size_t n_tar = 0;
        double presence_tar = 0.0; // `(node.n_tar / total_tar) * node.n_tar`
        double presence_neg = 0.0; // `(node.n_neg / total_neg) * node.n_tar`
    };
    std::vector<NodeSums> node_sums(pool.size());
    const double total_tar_inv = 1.0 / total_tar;
    const double total_neg_inv = 1.0 / total_neg;

    target_counts.visit(nodes, [&](const auto& n_tar_at) {
        pool.parallel_for_chunks(n_nodes, [&](
            std::size_t start, std::size_t end, std::size_t chunk_id
        ) {
            NodeSums sums;

            for (std::size_t node_idx = start; node_idx < end; ++node_idx) {
                const std::size_t n_tar = n_tar_at(node_idx);
                if (n_tar == 0) {
                    continue;
                }
                if (n_tar > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument("Node n_tar exceeds uint32 range");
                }
                const auto prevalence = nodes[node_idx].prevalence;
                if (n_tar > prevalence) {
                    throw std::invalid_argument("Node n_tar exceeds prevalence");
                }
                const std::size_t n_neg = prevalence - n_tar;
                if (n_neg > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::invalid_argument("Node n_neg exceeds uint32 range");
                }
                sums.n_tar += n_tar;
                sums.presence_tar += n_tar * total_tar_inv * n_tar;
                sums.presence_neg += n_neg * total_neg_inv * n_tar;
            }
            node_sums[chunk_id] = sums;
        });
    });
    NodeSums totals;
    for (const auto& sums : node_sums) {
        totals.n_tar += sums.n_tar;
        totals.presence_tar += sums.presence_tar;
        totals.presence_neg += sums.presence_neg;
    }
    if (totals.n_tar == 0) {
        throw std::invalid_argument("No target minimizers are available for threshold estimation");
    }

    return {
        1.0 - totals.presence_tar / totals.n_tar,
        totals.presence_neg / totals.n_tar
    };
}

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
) {
    // Find edges that pass the weight threshold
    const std::size_t th = edge_weight_th;
    std::size_t retained_count = 0;
    if (n_edges != 0) {
        const auto* retained_end = std::lower_bound(
            edges,
            edges + n_edges,
            th,
            [](const Edge& edge, std::size_t threshold) {
                return edge.weight > threshold;
            }
        );
        retained_count = static_cast<std::size_t>(retained_end - edges);
    }

    // Store each retained edge as two endpoint records
    // Used for building the CSR adjacency lists (offsets and neighbors)
    NoInitArray<Endpoint> endpoints(retained_count * 2);
    pool.parallel_for(retained_count, [&](std::size_t i) {
        if (edges[i].first >= n_nodes || edges[i].second >= n_nodes) {
            throw std::invalid_argument("Edge endpoint does not correspond to a node");
        }
        endpoints[i * 2] = Endpoint{edges[i].first, i * 2};
        endpoints[i * 2 + 1] = Endpoint{edges[i].second, i * 2 + 1};
    });
    lsd_radix_sort(endpoints, &Endpoint::node, pool);

    // Unique node indices in endpoints
    std::vector<std::size_t> connected;
    connected.reserve(endpoints.size());
    // Offsets into the neighbors array
    std::vector<std::size_t> offsets;
    offsets.reserve(endpoints.size() + 1);
    // Map the original node indices to indices of retained nodes
    ankerl::unordered_dense::map<std::size_t, std::size_t> node_indices;
    node_indices.reserve(endpoints.size());
    for (std::size_t i = 0; i < endpoints.size(); ++i) {
        if (i == 0 || endpoints[i].node != endpoints[i - 1].node) {
            node_indices.emplace(endpoints[i].node, connected.size());
            connected.push_back(endpoints[i].node);
            offsets.push_back(i);
        }
    }
    offsets.push_back(endpoints.size());

    // Calculate the penalty of each retained node
    filtered.nodes = NoInitArray<FilteredNode>(connected.size());
    const double total_tar_inv = 1.0 / total_tar;
    const double total_neg_inv = 1.0 / total_neg;

    target_counts.visit(nodes, [&](const auto& n_tar_at) {
        pool.parallel_for(connected.size(), [&](std::size_t i) {
            const auto node_idx = connected[i];
            const std::size_t n_tar = n_tar_at(node_idx);
            if (n_tar > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("Filtered node n_tar exceeds uint32 range");
            }
            const auto prevalence = nodes[node_idx].prevalence;
            if (n_tar > prevalence) {
                throw std::invalid_argument("Filtered node n_tar exceeds prevalence");
            }
            const std::size_t n_neg = prevalence - n_tar;
            if (n_neg > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("Filtered node n_neg exceeds uint32 range");
            }

            const double frac_tar = n_tar * total_tar_inv;
            const double frac_neg = n_neg * total_neg_inv;
            filtered.nodes[i] = FilteredNode{
                node_idx,
                static_cast<std::uint32_t>(n_tar),
                static_cast<std::uint32_t>(n_neg),
                std::sqrt((1.0 - frac_tar) * (1.0 - frac_tar) + frac_neg * frac_neg)
            };
        });
    });

    // Update edge endpoints to indices of retained nodes
    filtered.edges = NoInitArray<Edge>(retained_count);
    const auto& node_indices_read = node_indices;
    pool.parallel_for(retained_count, [&](std::size_t i) {
        filtered.edges[i] = Edge{
            node_indices_read.at(edges[i].first),
            node_indices_read.at(edges[i].second),
            edges[i].weight
        };
    });

    // The sorted endpoints array is parallel to neighbors
    // For each endpoint, recover the opposite endpoint with Endpoint::idx
    NoInitArray<std::size_t> neighbors(endpoints.size());
    pool.parallel_for(endpoints.size(), [&](std::size_t i) {
        const auto endpoint_idx = endpoints[i].idx;
        const auto& edge = filtered.edges[endpoint_idx / 2];
        neighbors[i] = endpoint_idx % 2 == 0
            ? edge.second
            : edge.first;
    });

    return GraphTopology(
        std::move(offsets),
        std::move(neighbors)
    );
}

void get_subgraphs(
    const GraphTopology& graph,
    const NoInitArray<FilteredNode>& nodes,
    double penalty_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilteredGraph& filtered
) {
    // Graph nodes are represented by indices, instead of hashes
    std::vector<std::size_t> seeds;
    seeds.reserve(nodes.size());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (nodes[node].penalty <= penalty_th) {
            seeds.push_back(node);
        }
    }
    log_python(
        " - Expanding subgraphs from " + std::to_string(seeds.size()) +
        " seed nodes (penalty<=" + std::to_string(penalty_th) + ")..."
    );

    struct FrontierNode {
        double penalty;
        std::size_t index;
    };
    const auto lower_priority = [](const FrontierNode& left, const FrontierNode& right) {
        if (left.penalty != right.penalty) {
            return left.penalty > right.penalty;
        }
        return left.index > right.index;
    };

    // Marks nodes accepted in any of the subgraphs
    std::vector<std::uint8_t> used(nodes.size(), 0);
    // Marks nodes in the frontier or accepted into the current subgraph
    // Cleared for each seed
    std::vector<std::uint8_t> seen(nodes.size(), 0);

    const std::size_t max_nodes_value = max_nodes.value_or(
        std::numeric_limits<std::size_t>::max()
    );
    for (const auto seed : seeds) {
        if (used[seed]) {
            continue;
        }

        Subgraph subgraph{seed};
        seen[seed] = 1;
        double sum_penalty = nodes[seed].penalty;
        std::priority_queue<
            FrontierNode, std::vector<FrontierNode>, decltype(lower_priority)
        > frontier(lower_priority);

        const auto add_neighbors = [&](std::size_t node) {
            for (const auto neighbor : graph.neighbors(node)) {
                if (!used[neighbor] && !seen[neighbor]) {
                    frontier.push({nodes[neighbor].penalty, neighbor});
                    seen[neighbor] = 1;
                }
            }
        };
        add_neighbors(seed);

        while (!frontier.empty() && subgraph.size() < max_nodes_value) {
            const auto candidate = frontier.top();
            frontier.pop();
            const double new_sum_penalty = sum_penalty + candidate.penalty;
            if (new_sum_penalty / (subgraph.size() + 1) <= penalty_th) {
                subgraph.push_back(candidate.index);
                sum_penalty = new_sum_penalty;
                add_neighbors(candidate.index);
            } else {
                // All remaining candidates have at least this penalty, so they
                // cannot lower the subgraph average below the threshold.
                seen[candidate.index] = 0;
                break;
            }
        }

        // Clear node states for the next seed
        while (!frontier.empty()) {
            seen[frontier.top().index] = 0;
            frontier.pop();
        }
        for (const auto node : subgraph) {
            seen[node] = 0;
        }

        if (subgraph.size() >= min_nodes) {
            for (const auto node : subgraph) {
                used[node] = 1;
            }
            filtered.subgraphs.push_back(std::move(subgraph));
        }
    }
}

} // namespace seqwin::internal

#include "seqwin/filter_internals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "seqwin/filter.hpp"
#include "seqwin/shared_internals.hpp"
#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {
namespace {

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
    const std::uint32_t* prevalence,
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
    return TargetCounts(std::move(counts), count_targets, prevalence);
}

} // namespace

TargetCounts count_target_nodes(
    const std::uint32_t* prevalence,
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
            prevalence, n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    if (counted_assemblies <= std::numeric_limits<std::uint16_t>::max()) {
        return build_target_counts<std::uint16_t>(
            prevalence, n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    return build_target_counts<std::uint32_t>(
        prevalence, n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
    );
}

std::pair<double, double> expected_presence(
    const std::uint32_t* prevalence,
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

    target_counts.with([&](const auto& n_tar_at) {
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
                const auto node_prevalence = prevalence[node_idx];
                if (n_tar > node_prevalence) {
                    throw std::invalid_argument("Node n_tar exceeds prevalence");
                }
                const std::size_t n_neg = node_prevalence - n_tar;
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
        throw std::invalid_argument("No target minimizer available for threshold estimation");
    }

    return {
        1.0 - totals.presence_tar / totals.n_tar,
        totals.presence_neg / totals.n_tar
    };
}

GraphTopology prune_graph(
    const std::uint32_t* prevalence,
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
    // One endpoint occurrence of a retained edge
    // Used for building the CSR adjacency lists (offsets and neighbors)
    struct Endpoint {
        // Original graph node represented by this endpoint
        std::size_t node;
        // Index in the flattened edge array
        // The first endpoint has even index; the second endpoint has odd index
        std::size_t idx;
    };

    // Scan the prefix of edges (sorted by weight)
    // Keep edges with sufficient weight and target support
    filtered.edges.clear();
    const std::size_t th = edge_weight_th;
    // Store each retained edge as two endpoint records
    std::vector<Endpoint> endpoints;
    target_counts.with([&](const auto& n_tar_at) {
        for (std::size_t i = 0; i < n_edges; ++i) {
            const auto& edge = edges[i];
            if (edge.weight <= th) {
                break;
            }
            if (edge.first >= n_nodes || edge.second >= n_nodes) {
                throw std::invalid_argument("Edge endpoint does not correspond to a node");
            }
            if (n_tar_at(edge.first) > th || n_tar_at(edge.second) > th) {
                const auto n_retained = filtered.edges.size();
                filtered.edges.push_back(edge);
                endpoints.push_back({edge.first, n_retained * 2});
                endpoints.push_back({edge.second, n_retained * 2 + 1});
            }
        }
    });
    lsd_radix_sort(endpoints, &Endpoint::node, pool);

    // Offsets into the neighbors array
    std::vector<std::size_t> offsets;
    offsets.reserve(endpoints.size() + 1);
    // Unique node indices in endpoints
    std::vector<std::size_t> connected;
    connected.reserve(endpoints.size());
    // Map endpoint indices to indices of retained nodes
    NoInitArray<std::size_t> node_indices(endpoints.size());
    for (std::size_t i = 0; i < endpoints.size(); ++i) {
        if (i == 0 || endpoints[i].node != endpoints[i - 1].node) {
            offsets.push_back(i);
            connected.push_back(endpoints[i].node);
        }
        node_indices[endpoints[i].idx] = connected.size() - 1;
    }
    offsets.push_back(endpoints.size());

    // The sorted endpoints array is parallel to neighbors
    // For each endpoint, recover the index of the opposite endpoint
    NoInitArray<std::size_t> neighbors(endpoints.size());
    pool.parallel_for(endpoints.size(), [&](std::size_t i) {
        neighbors[i] = node_indices[endpoints[i].idx ^ 1];
    });

    // Calculate the penalty of each retained node
    filtered.nodes = NoInitArray<FilteredNode>(connected.size());
    const double total_tar_inv = 1.0 / total_tar;
    const double total_neg_inv = 1.0 / total_neg;

    target_counts.with([&](const auto& n_tar_at) {
        pool.parallel_for(connected.size(), [&](std::size_t i) {
            const auto node_idx = connected[i];
            const std::size_t n_tar = n_tar_at(node_idx);
            if (n_tar > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("Filtered node n_tar exceeds uint32 range");
            }
            const auto node_prevalence = prevalence[node_idx];
            if (n_tar > node_prevalence) {
                throw std::invalid_argument("Filtered node n_tar exceeds prevalence");
            }
            const std::size_t n_neg = node_prevalence - n_tar;
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
    // GraphTopology represents nodes by indices into the retained nodes
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
        std::size_t node;
    };
    const auto lower_priority = [](const FrontierNode& left, const FrontierNode& right) {
        if (left.penalty != right.penalty) {
            return left.penalty > right.penalty;
        }
        return left.node > right.node;
    };
    // Reuse heap storage across seeds
    std::vector<FrontierNode> frontier;

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

        const auto add_neighbors = [&](std::size_t node) {
            for (const auto neighbor : graph.neighbors(node)) {
                if (!used[neighbor] && !seen[neighbor]) {
                    frontier.push_back({nodes[neighbor].penalty, neighbor});
                    std::push_heap(frontier.begin(), frontier.end(), lower_priority);
                    seen[neighbor] = 1;
                }
            }
        };
        add_neighbors(seed);

        while (!frontier.empty() && subgraph.size() < max_nodes_value) {
            std::pop_heap(frontier.begin(), frontier.end(), lower_priority);
            const auto candidate = frontier.back();
            frontier.pop_back();

            const double new_sum_penalty = sum_penalty + candidate.penalty;
            if (new_sum_penalty / (subgraph.size() + 1) <= penalty_th) {
                subgraph.push_back(candidate.node);
                sum_penalty = new_sum_penalty;
                add_neighbors(candidate.node);
            } else {
                // All remaining candidates have at least this penalty, so they
                // cannot lower the subgraph average below the threshold.
                seen[candidate.node] = 0;
                break;
            }
        }

        // Clear node states for the next seed
        for (const auto& candidate : frontier) {
            seen[candidate.node] = 0;
        }
        for (const auto node : subgraph) {
            seen[node] = 0;
        }
        frontier.clear();

        // Output subgraphs using node indices from the original graph
        if (subgraph.size() >= min_nodes) {
            for (auto& node : subgraph) {
                used[node] = 1;
                node = nodes[node].idx;
            }
            filtered.subgraphs.push_back(std::move(subgraph));
        }
    }
}

} // namespace seqwin::internal

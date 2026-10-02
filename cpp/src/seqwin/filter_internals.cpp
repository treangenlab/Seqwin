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

template <typename Counter>
TargetCounts count_assembly_group(
    std::size_t n_nodes,
    const std::size_t* assembly_nodes,
    const std::size_t* node_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    bool count_targets,
    ThreadPool& pool
) {
    NoInitArray<Counter> counts(n_nodes);
    pool.parallel_for(n_nodes, [&](std::size_t node_begin, std::size_t node_end, std::size_t) {
        std::fill(
            counts.begin() + node_begin,
            counts.begin() + node_end,
            Counter{0}
        );
        for (std::size_t assembly_idx = 0; assembly_idx < n_assemblies; ++assembly_idx) {
            if (is_targets[assembly_idx] != count_targets) {
                continue;
            }
            const auto* begin = assembly_nodes + node_offsets[assembly_idx];
            const auto* end = assembly_nodes + node_offsets[assembly_idx + 1];
            const auto* first = std::lower_bound(begin, end, node_begin);
            const auto* last = std::lower_bound(first, end, node_end);
            for (auto it = first; it != last; ++it) {
                ++counts[*it];
            }
        }
    });
    return TargetCounts(std::move(counts), count_targets);
}

TargetCounts build_target_counts(
    std::size_t n_nodes,
    const std::size_t* assembly_nodes,
    const std::size_t* node_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
) {
    const bool count_targets = total_tar <= total_neg;
    const std::size_t counted_assemblies = count_targets ? total_tar : total_neg;

    if (counted_assemblies <= std::numeric_limits<std::uint8_t>::max()) {
        return count_assembly_group<std::uint8_t>(
            n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    if (counted_assemblies <= std::numeric_limits<std::uint16_t>::max()) {
        return count_assembly_group<std::uint16_t>(
            n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
        );
    }
    return count_assembly_group<std::uint32_t>(
        n_nodes, assembly_nodes, node_offsets, is_targets, n_assemblies, count_targets, pool
    );
}

} // namespace

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

    const std::size_t total_tar = std::count(is_targets, is_targets + n_assemblies, true);
    const std::size_t total_neg = n_assemblies - total_tar;
    if (total_tar == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (total_neg == 0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }

    auto target_counts = build_target_counts(
        n_nodes,
        assembly_nodes,
        node_offsets,
        is_targets,
        n_assemblies,
        total_tar,
        total_neg,
        pool
    );

    // Calculate e_absence_tar and e_presence_neg
    // Only nodes present in at least on target assembly contribute to the calculation
    std::size_t sum_n_tar = 0;
    double sum_presence_tar = 0.0;
    double sum_presence_neg = 0.0;
    const auto total_tar_double = static_cast<double>(total_tar);
    const auto total_neg_double = static_cast<double>(total_neg);

    target_counts.visit(nodes, [&](const auto& n_tar_at) {
        for (std::size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
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
            const double frac_tar = n_tar / total_tar_double;
            const double frac_neg = n_neg / total_neg_double;
            sum_n_tar += n_tar;
            sum_presence_tar += frac_tar * n_tar;
            sum_presence_neg += frac_neg * n_tar;
        }
    });
    if (sum_n_tar == 0) {
        throw std::invalid_argument("No target minimizers are available for threshold estimation");
    }

    FilteredGraph filtered;
    filtered.total_tar = total_tar;
    filtered.total_neg = total_neg;
    filtered.e_absence_tar = 1.0 - sum_presence_tar / sum_n_tar;
    filtered.e_presence_neg = sum_presence_neg / sum_n_tar;
    return {std::move(filtered), std::move(target_counts)};
}

void prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    double edge_weight_th,
    const TargetCounts& target_counts,
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

    // Indices of retained nodes after edge filtering
    NoInitArray<std::size_t> connected_multi(retained_count * 2);
    pool.parallel_for(retained_count, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i) {
            if (edges[i].first >= n_nodes || edges[i].second >= n_nodes) {
                throw std::invalid_argument("Edge endpoint does not correspond to a node");
            }
            connected_multi[i * 2] = edges[i].first;
            connected_multi[i * 2 + 1] = edges[i].second;
        }
    });
    lsd_radix_sort(connected_multi, pool);
    // These are the nodes output to FilteredGraph.nodes
    std::vector<std::size_t> connected;
    connected.reserve(connected_multi.size());
    for (std::size_t i = 0; i < connected_multi.size(); ++i) {
        if (i == 0 || connected_multi[i] != connected_multi[i - 1]) {
            connected.push_back(connected_multi[i]);
        }
    }
    connected_multi.reset();

    // Calculate the penalty of each retained node
    filtered.nodes = NoInitArray<FilteredNode>(connected.size());
    const double total_tar = filtered.total_tar;
    const double total_neg = filtered.total_neg;
    // Map the original node indices to indices of retained nodes
    ankerl::unordered_dense::map<std::size_t, std::size_t> node_indices;
    node_indices.reserve(connected.size());

    target_counts.visit(nodes, [&](const auto& n_tar_at) {
        for (std::size_t i = 0; i < connected.size(); ++i) {
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

            const double frac_tar = n_tar / total_tar;
            const double frac_neg = n_neg / total_neg;
            filtered.nodes[i] = FilteredNode{
                node_idx,
                static_cast<std::uint32_t>(n_tar),
                static_cast<std::uint32_t>(n_neg),
                std::sqrt((1.0 - frac_tar) * (1.0 - frac_tar) + frac_neg * frac_neg)
            };
            node_indices.emplace(node_idx, i);
        }
    });

    // Update edge endpoints to indices of retained nodes
    filtered.edges = NoInitArray<Edge>(retained_count);
    for (std::size_t i = 0; i < retained_count; ++i) {
        filtered.edges[i] = Edge{
            node_indices.at(edges[i].first),
            node_indices.at(edges[i].second),
            edges[i].weight
        };
    }
}

void get_subgraphs(
    const NoInitArray<FilteredNode>& nodes,
    const NoInitArray<Edge>& edges,
    double penalty_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilteredGraph& filtered
) {
    // Graph nodes are represented by indices, instead of hashes
    const GraphTopology graph(nodes.size(), edges);

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

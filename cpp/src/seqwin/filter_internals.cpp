#include "seqwin/filter_internals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
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
    ArrayView<std::uint32_t> prevalences,
    ArrayView<std::size_t> assembly_nodes,
    ArrayView<std::size_t> node_offsets,
    ArrayView<bool> is_targets,
    bool count_targets,
    ThreadPool& pool
) {
    NoInitArray<Counter> counts(prevalences.size());
    pool.parallel_for_chunks(
        prevalences.size(), [&](std::size_t start, std::size_t end, std::size_t
    ) {
        std::fill(
            counts.begin() + start,
            counts.begin() + end,
            Counter{0}
        );
        for (std::size_t assembly_idx = 0; assembly_idx < is_targets.size(); ++assembly_idx) {
            if (is_targets[assembly_idx] != count_targets) {
                continue;
            }
            if (node_offsets[assembly_idx] == node_offsets[assembly_idx + 1]) {
                continue;
            }
            const auto* assembly_start = assembly_nodes.data() + node_offsets[assembly_idx];
            const auto* assembly_end = assembly_nodes.data() + node_offsets[assembly_idx + 1];
            const auto* first = std::lower_bound(assembly_start, assembly_end, start);
            const auto* last = std::lower_bound(first, assembly_end, end);
            for (auto it = first; it != last; ++it) {
                ++counts[*it];
            }
        }
    });
    return TargetCounts(std::move(counts), count_targets, prevalences);
}

} // namespace

TargetCounts count_target_nodes(
    ArrayView<std::uint32_t> prevalences,
    ArrayView<std::size_t> assembly_nodes,
    ArrayView<std::size_t> node_offsets,
    ArrayView<bool> is_targets,
    std::size_t total_tar,
    std::size_t total_neg,
    ThreadPool& pool
) {
    // The smaller assembly group is counted using the narrowest safe counter type
    const bool count_targets = total_tar <= total_neg;
    const std::size_t counted_assemblies = count_targets ? total_tar : total_neg;
    if (counted_assemblies <= std::numeric_limits<std::uint8_t>::max()) {
        return build_target_counts<std::uint8_t>(
            prevalences, assembly_nodes, node_offsets, is_targets, count_targets, pool
        );
    }
    if (counted_assemblies <= std::numeric_limits<std::uint16_t>::max()) {
        return build_target_counts<std::uint16_t>(
            prevalences, assembly_nodes, node_offsets, is_targets, count_targets, pool
        );
    }
    return build_target_counts<std::uint32_t>(
        prevalences, assembly_nodes, node_offsets, is_targets, count_targets, pool
    );
}

GraphTopology prune_graph(
    ArrayView<std::uint32_t> prevalences,
    ArrayView<Edge> edges,
    const TargetCounts& target_counts,
    const FilterConfig& config,
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
    const std::size_t th = filtered.edge_weight_th;
    // Store each retained edge as two endpoint records
    std::vector<Endpoint> endpoints;
    target_counts.with([&](const auto& n_tar_at) {
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            if (edge.weight <= th) {
                break;
            }
            if (edge.first >= prevalences.size() || edge.second >= prevalences.size()) {
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

    // Populate FilteredGraph::nodes and calculate n_tar_th
    std::size_t sum_n_tar = 0;
    double sum_presence = 0.0;
    filtered.nodes = NoInitArray<FilteredNode>(connected.size());
    target_counts.with([&](const auto& n_tar_at) {
        for (std::size_t i = 0; i < connected.size(); ++i) {
            const auto node_idx = connected[i];
            const std::size_t n_tar = n_tar_at(node_idx);
            const auto prevalence = prevalences[node_idx];
            if (n_tar > prevalence || n_tar > filtered.total_tar) {
                throw std::invalid_argument("Invalid target prevalence for retained node");
            }
            const std::size_t n_neg = prevalence - n_tar;
            if (n_neg > filtered.total_neg) {
                throw std::invalid_argument("Invalid non-target prevalence for retained node");
            }
            filtered.nodes[i] = FilteredNode{
                node_idx,
                static_cast<std::uint32_t>(n_tar),
                static_cast<std::uint32_t>(n_neg)
            };
            sum_n_tar += n_tar;
            sum_presence += n_tar * n_tar;
        }
    });

    if (config.n_tar_th) {
        filtered.n_tar_th = *config.n_tar_th;
    } else if (sum_n_tar == 0) {
        filtered.n_tar_th = 0;
    } else {
        const double mean_presence = sum_presence / sum_n_tar;
        const double total = filtered.total_tar;
        const double threshold = total - (1.0 - config.stringency / 10.0) * (total - mean_presence);
        filtered.n_tar_th = static_cast<std::size_t>(std::clamp(std::ceil(threshold), 0.0, total));
    }

    return GraphTopology(
        std::move(offsets),
        std::move(neighbors)
    );
}

void get_subgraphs(
    const GraphTopology& graph,
    FilteredGraph& filtered
) {
    // GraphTopology represents nodes by indices into the retained nodes
    const auto& nodes = filtered.nodes;
    const std::int64_t n_tar_th = filtered.n_tar_th;
    const std::int64_t n_neg_th = filtered.n_neg_th;
    const auto allowed_absences = filtered.total_tar - filtered.n_tar_th;
    std::vector<std::size_t> seeds;
    seeds.reserve(nodes.size());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (nodes[node].n_tar >= n_tar_th && nodes[node].n_neg <= n_neg_th) {
            seeds.push_back(node);
        }
    }
    log_python(
        " - Expanding subgraphs from " + std::to_string(seeds.size()) + " seed nodes..."
    );

    struct FrontierNode {
        double max_score;
        double sum_score;
        std::size_t node;
    };
    const auto lower_priority = [](const FrontierNode& left, const FrontierNode& right) {
        return std::tie(left.max_score, left.sum_score, left.node) >
               std::tie(right.max_score, right.sum_score, right.node);
    };
    // Reuse heap storage across seeds
    std::vector<FrontierNode> frontier;

    // Marks nodes accepted in any of the subgraphs
    std::vector<std::uint8_t> used(nodes.size(), 0);
    // Includes rejected nodes until the current seed expansion finishes
    std::vector<std::uint8_t> seen(nodes.size(), 0);

    std::vector<std::size_t> encountered;
    const std::size_t max_nodes_value = filtered.max_nodes.value_or(
        std::numeric_limits<std::size_t>::max()
    );
    for (const auto seed : seeds) {
        if (used[seed]) {
            continue;
        }

        Subgraph subgraph{seed};
        seen[seed] = 1;
        encountered.push_back(seed);
        std::int64_t tar_budget = nodes[seed].n_tar - n_tar_th;
        std::int64_t neg_budget = n_neg_th - nodes[seed].n_neg;

        const auto add_neighbors = [&](std::size_t node) {
            for (const auto neighbor : graph.neighbors(node)) {
                if (used[neighbor] || seen[neighbor]) {
                    continue;
                }
                seen[neighbor] = 1;
                encountered.push_back(neighbor);
                const double absences = filtered.total_tar - nodes[neighbor].n_tar;
                const double presences = nodes[neighbor].n_neg;
                if (
                    (allowed_absences == 0 && absences != 0.0) ||
                    (n_neg_th == 0 && presences != 0.0)
                ) {
                    continue;
                }
                const double a = allowed_absences == 0 ? 0.0 : absences / allowed_absences;
                const double b = n_neg_th == 0 ? 0.0 : presences / n_neg_th;
                frontier.push_back({std::max(a, b), a + b, neighbor});
                std::push_heap(frontier.begin(), frontier.end(), lower_priority);
            }
        };
        add_neighbors(seed);

        while (!frontier.empty() && subgraph.size() < max_nodes_value) {
            std::pop_heap(frontier.begin(), frontier.end(), lower_priority);
            const auto candidate = frontier.back();
            frontier.pop_back();

            const auto& node = nodes[candidate.node];
            const auto new_tar_budget = tar_budget + (node.n_tar - n_tar_th);
            const auto new_neg_budget = neg_budget + (n_neg_th - node.n_neg);
            if (new_tar_budget < 0 || new_neg_budget < 0) {
                continue;
            }
            subgraph.push_back(candidate.node);
            tar_budget = new_tar_budget;
            neg_budget = new_neg_budget;
            add_neighbors(candidate.node);
        }

        // Clear node states for the next seed
        for (const auto node : encountered) {
            seen[node] = 0;
        }
        encountered.clear();
        frontier.clear();

        // Output subgraphs using node indices from the original graph
        if (subgraph.size() >= filtered.min_nodes) {
            for (auto& node : subgraph) {
                used[node] = 1;
                node = nodes[node].idx;
            }
            filtered.subgraphs.push_back(std::move(subgraph));
        }
    }
}

} // namespace seqwin::internal

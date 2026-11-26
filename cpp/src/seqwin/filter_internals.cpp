#include "seqwin/filter_internals.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
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
    std::vector<std::pair<std::size_t, std::size_t>> assembly_ranges;
    for (std::size_t i = 0; i < is_targets.size(); ++i) {
        if (is_targets[i] == count_targets) {
            if (assembly_ranges.empty() || assembly_ranges.back().second != i) {
                assembly_ranges.emplace_back(i, i + 1);
            } else {
                assembly_ranges.back().second = i + 1;
            }
        }
    }
    NoInitArray<Counter> counts(prevalences.size());
    pool.parallel_for_chunks(
        prevalences.size(), [&](std::size_t start, std::size_t end, std::size_t
    ) {
        std::fill(
            counts.begin() + start,
            counts.begin() + end,
            Counter{0}
        );
        for (const auto& [begin, stop] : assembly_ranges) {
            for (std::size_t assembly_idx = begin; assembly_idx < stop; ++assembly_idx) {
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
    std::size_t total_tar,
    std::size_t total_neg,
    double edge_weight_th,
    double tar_weight,
    std::optional<std::size_t> n_tar_th,
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
    offsets.reserve(std::min(prevalences.size(), endpoints.size()) + 1);
    // Map endpoint indices to indices of retained nodes
    NoInitArray<std::size_t> node_indices(endpoints.size());
    for (std::size_t i = 0; i < endpoints.size(); ++i) {
        if (i == 0 || endpoints[i].node != endpoints[i - 1].node) {
            offsets.push_back(i);
        }
        node_indices[endpoints[i].idx] = offsets.size() - 1;
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
    filtered.nodes = NoInitArray<FilteredNode>(offsets.size() - 1);
    target_counts.with([&](const auto& n_tar_at) {
        for (std::size_t i = 0; i < filtered.nodes.size(); ++i) {
            const auto node_idx = endpoints[offsets[i]].node;
            const std::size_t n_tar = n_tar_at(node_idx);
            const auto prevalence = prevalences[node_idx];
            if (n_tar > prevalence || n_tar > total_tar) {
                throw std::invalid_argument("Filtered node n_tar is invalid");
            }
            const std::size_t n_neg = prevalence - n_tar;
            if (n_neg > total_neg) {
                throw std::invalid_argument("Filtered node n_neg exceeds total_neg");
            }
            filtered.nodes[i] = FilteredNode{
                node_idx,
                static_cast<std::uint32_t>(n_tar),
                static_cast<std::uint32_t>(n_neg)
            };
            if (!n_tar_th) {
                sum_n_tar += n_tar;
                sum_presence += n_tar * n_tar;
            }
        }
    });

    if (n_tar_th) {
        filtered.n_tar_th = *n_tar_th;
    } else if (sum_n_tar == 0) {
        filtered.n_tar_th = 0;
    } else {
        const double mean_presence = sum_presence / sum_n_tar;
        const double total = total_tar;
        const double threshold = (1.0 - tar_weight) * mean_presence + tar_weight * total;
        filtered.n_tar_th = static_cast<std::size_t>(
            std::clamp(std::ceil(threshold), 0.0, total)
        );
    }

    return GraphTopology(
        std::move(offsets),
        std::move(neighbors)
    );
}

void get_subgraphs(
    const GraphTopology& graph,
    const NoInitArray<FilteredNode>& nodes,
    std::size_t total_tar,
    std::int64_t n_tar_th,
    std::int64_t n_neg_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilteredGraph& filtered
) {
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

    // Nodes encountered in the current seed expansion are labeled with seed_i
    // Nodes accepted in any of the subgraphs are labeled with used
    std::vector<std::uint16_t> seen(nodes.size(), 0);
    // Identifier for the current seed, which may wrap around on overflow
    std::uint16_t seed_i = 0;
    constexpr auto used = std::numeric_limits<std::uint16_t>::max();

    std::size_t n_seeds = 0;
    const auto n_absence_th = total_tar - n_tar_th;
    const double n_absence_th_inv = n_absence_th == 0 ? 0.0 : 1.0 / n_absence_th;
    const double n_neg_th_inv = n_neg_th == 0 ? 0.0 : 1.0 / n_neg_th;
    const std::size_t max_nodes_value = max_nodes.value_or(
        std::numeric_limits<std::size_t>::max()
    );
    for (std::size_t seed = 0; seed < nodes.size(); ++seed) {
        if (nodes[seed].n_tar < n_tar_th || nodes[seed].n_neg > n_neg_th) {
            continue;
        }
        ++n_seeds;
        if (seen[seed] == used) {
            continue;
        }

        if (++seed_i == used) {
            for (auto& state : seen) {
                if (state != used) {
                    state = 0;
                }
            }
            seed_i = 1;
        }
        Subgraph subgraph{seed};
        seen[seed] = seed_i;
        std::int64_t tar_budget = nodes[seed].n_tar - n_tar_th;
        std::int64_t neg_budget = n_neg_th - nodes[seed].n_neg;

        const auto add_neighbors = [&](std::size_t node) {
            for (const auto neighbor : graph.neighbors(node)) {
                if (seen[neighbor] == used || seen[neighbor] == seed_i) {
                    continue;
                }
                seen[neighbor] = seed_i;
                const double n_absence = total_tar - nodes[neighbor].n_tar;
                const double n_neg = nodes[neighbor].n_neg;
                if (
                    (n_absence_th == 0 && n_absence != 0.0) ||
                    (n_neg_th == 0 && n_neg != 0.0)
                ) {
                    continue;
                }
                const double a = n_absence * n_absence_th_inv;
                const double b = n_neg * n_neg_th_inv;
                frontier.push_back({std::max(a, b), a + b, neighbor});
                std::push_heap(frontier.begin(), frontier.end(), lower_priority);
            }
        };
        if (max_nodes_value > 1) {
            add_neighbors(seed);
        }

        while (!frontier.empty()) {
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
            if (subgraph.size() == max_nodes_value) {
                break;
            }
            tar_budget = new_tar_budget;
            neg_budget = new_neg_budget;
            add_neighbors(candidate.node);
        }
        frontier.clear();

        // Output subgraphs using node indices from the original graph
        if (subgraph.size() >= min_nodes) {
            for (auto& node : subgraph) {
                seen[node] = used;
                node = nodes[node].idx;
            }
            filtered.subgraphs.push_back(std::move(subgraph));
        }
    }
    log_python(" - Found " + std::to_string(n_seeds) + " seed nodes");
}

} // namespace seqwin::internal

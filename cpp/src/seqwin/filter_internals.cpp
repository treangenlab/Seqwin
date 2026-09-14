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
#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {

FilterResult get_penalty(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t n_cpu
) {
    /** Metadata shared by all FASTA records in one assembly. */
    struct RecordInfo {
        /** Inclusive global index of the assembly's final FASTA record. */
        std::uint32_t last_record_idx;
        /** Whether the assembly belongs to the target set, as 0 or 1. */
        std::uint32_t is_target;
    };
    /**
     * Sums over all nodes, for penalty threshold calculation.
     * For all k-mers in targets, calculate their total presence in targets or non-targets.
     */
    struct NodeSums {
        std::size_t n_tar = 0;
        double presence_tar = 0.0; // `(node.n_tar / total_tar) * node.n_tar`
        double presence_neg = 0.0; // `(node.n_neg / total_neg) * node.n_tar`
    };

    if (n_record_offsets != n_assemblies + 1) {
        throw std::invalid_argument("len(record_offsets) must equal len(is_targets) + 1");
    }
    if (n_record_offsets == 0 || record_offsets[0] != 0) {
        throw std::invalid_argument("record_offsets must start with 0");
    }
    if (n_assemblies > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Number of assemblies exceeds uint32 range");
    }
    for (std::size_t i = 0; i < n_assemblies; ++i) {
        if (record_offsets[i + 1] < record_offsets[i]) {
            throw std::invalid_argument("record_offsets must be nondecreasing");
        }
    }

    // Use double for downstream calculation
    const double total_tar = std::count(is_targets, is_targets + n_assemblies, true);
    const auto total_neg = n_assemblies - total_tar;
    if (total_tar == 0.0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (total_neg == 0.0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }

    std::size_t n_workers = std::max<std::size_t>(1, n_cpu);
    if (n_nodes > 0) {
        n_workers = std::min(n_workers, n_nodes);
    }
    ThreadPool pool(n_workers);

    const std::uint32_t n_records = record_offsets[n_assemblies];
    NoInitArray<RecordInfo> record_info(n_records);
    pool.parallel_for(n_assemblies, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t assembly_idx = start; assembly_idx < end; ++assembly_idx) {
            const std::uint32_t record_start = record_offsets[assembly_idx];
            const std::uint32_t record_stop = record_offsets[assembly_idx + 1];
            if (record_start == record_stop) {
                continue;
            }
            const RecordInfo info{
                record_stop - 1,
                is_targets[assembly_idx] ? 1U : 0U
            };
            std::fill(
                record_info.begin() + record_start,
                record_info.begin() + record_stop,
                info
            );
        }
    });

    std::vector<NodeSums> node_sums(n_workers);
    pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t worker_i) {
        auto& sums = node_sums[worker_i];

        for (std::size_t node_i = start; node_i < end; ++node_i) {
            auto& node = nodes[node_i];
            if (node.start == node.stop) {
                node.n_tar = 0;
                node.n_neg = 0;
                node.penalty = 1.0;
                continue;
            }

            auto previous_record_idx = kmers[node.start].record_idx;
            if (previous_record_idx >= n_records) {
                throw std::invalid_argument("record_idx is outside record_offsets range");
            }
            auto info = record_info[previous_record_idx];
            auto last_record_idx = info.last_record_idx;
            std::uint32_t n_tar = info.is_target;
            std::uint32_t n_neg = 1U - info.is_target;

            for (std::size_t kmer_i = node.start + 1; kmer_i < node.stop; ++kmer_i) {
                const std::uint32_t record_idx = kmers[kmer_i].record_idx;
                if (record_idx < previous_record_idx) {
                    throw std::invalid_argument("record_idx must be nondecreasing within each node range");
                }
                previous_record_idx = record_idx;

                if (record_idx <= last_record_idx) {
                    continue;
                }
                if (record_idx >= n_records) {
                    throw std::invalid_argument("record_idx is outside record_offsets range");
                }
                info = record_info[record_idx];
                last_record_idx = info.last_record_idx;
                n_tar += info.is_target;
                n_neg += 1U - info.is_target;
            }

            node.n_tar = n_tar;
            node.n_neg = n_neg;
            const double frac_tar = n_tar / total_tar;
            const double frac_neg = n_neg / total_neg;
            node.penalty = std::sqrt((1.0 - frac_tar) * (1.0 - frac_tar) + frac_neg * frac_neg);

            sums.n_tar += n_tar;
            sums.presence_tar += frac_tar * n_tar;
            sums.presence_neg += frac_neg * n_tar;
        }
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

    FilterResult result;
    result.total_tar = total_tar;
    result.total_neg = total_neg;
    result.e_absence_tar = 1.0 - totals.presence_tar / totals.n_tar;
    result.e_presence_neg = totals.presence_neg / totals.n_tar;
    return result;
}

void prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    double edge_weight_th,
    FilterResult& result
) {
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

    std::vector<std::size_t> connected;
    connected.reserve(retained_count * 2);
    for (std::size_t i = 0; i < retained_count; ++i) {
        if (edges[i].first >= n_nodes || edges[i].second >= n_nodes) {
            throw std::invalid_argument("Edge endpoint does not correspond to a node");
        }
        connected.push_back(edges[i].first);
        connected.push_back(edges[i].second);
    }
    std::sort(connected.begin(), connected.end());
    connected.erase(std::unique(connected.begin(), connected.end()), connected.end());

    result.nodes = NoInitArray<Node>(connected.size());
    ankerl::unordered_dense::map<std::size_t, std::size_t> node_indices;
    node_indices.reserve(connected.size());
    for (std::size_t i = 0; i < connected.size(); ++i) {
        result.nodes[i] = nodes[connected[i]];
        node_indices.emplace(connected[i], i);
    }

    result.edges = NoInitArray<Edge>(retained_count);
    for (std::size_t i = 0; i < retained_count; ++i) {
        result.edges[i] = Edge{
            node_indices.at(edges[i].first),
            node_indices.at(edges[i].second),
            edges[i].weight
        };
    }
}

void get_subgraphs(
    const NoInitArray<Node>& nodes,
    const NoInitArray<Edge>& edges,
    double penalty_th,
    std::size_t min_nodes,
    std::optional<std::size_t> max_nodes,
    FilterResult& result
) {
    // Graph nodes are represented by indices, instead of hashes
    const GraphTopology graph(nodes, edges);

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

        std::vector<std::size_t> subgraph{seed};
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
            result.subgraphs.push_back(std::move(subgraph));
        }
    }
}

} // namespace seqwin::internal

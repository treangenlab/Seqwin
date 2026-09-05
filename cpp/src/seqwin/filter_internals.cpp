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

    const std::size_t total_tar = std::count(is_targets, is_targets + n_assemblies, true);
    const auto total_neg = n_assemblies - total_tar;
    if (total_tar == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (total_neg == 0) {
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

    pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t) {
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
            const double frac_tar = static_cast<double>(n_tar) / total_tar;
            const double frac_neg = static_cast<double>(n_neg) / total_neg;
            node.penalty = std::sqrt((1.0 - frac_tar) * (1.0 - frac_tar) + frac_neg * frac_neg);
        }
    });

    FilterResult result;
    result.total_tar = total_tar;
    result.total_neg = total_neg;
    return result;
}

PrunedGraph prune_graph(
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    double edge_weight_th
) {
    ankerl::unordered_dense::set<std::uint64_t> connected;
    connected.reserve(n_nodes);

    PrunedGraph graph;
    graph.edges.reserve(n_edges);
    const std::size_t th = edge_weight_th;
    for (std::size_t i = 0; i < n_edges; ++i) {
        if (edges[i].weight > th) {
            graph.edges.push_back(edges[i]);
            connected.insert(edges[i].first);
            connected.insert(edges[i].second);
        }
    }

    graph.nodes.reserve(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (connected.count(nodes[i].hash)) {
            graph.nodes.push_back(nodes[i]);
        }
    }
    return graph;
}

std::pair<Subgraphs, std::vector<std::size_t>> get_subgraphs(
    const std::vector<Node>& nodes,
    const std::vector<Edge>& edges,
    double penalty_th,
    std::size_t min_nodes,
    std::size_t max_nodes
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
    Subgraphs subgraphs;
    std::vector<std::size_t> used_nodes;

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

        while (!frontier.empty() && subgraph.size() < max_nodes) {
            const auto candidate = frontier.top();
            frontier.pop();
            seen[candidate.index] = 0; // Rejected candidate can be discovered again later
            const double new_sum_penalty = sum_penalty + candidate.penalty;
            if (
                new_sum_penalty / static_cast<double>(subgraph.size() + 1) <= penalty_th
            ) {
                subgraph.push_back(candidate.index);
                seen[candidate.index] = 1;
                sum_penalty = new_sum_penalty;
                add_neighbors(candidate.index);
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
            auto& hashes = subgraphs.emplace_back();
            hashes.reserve(subgraph.size());
            for (const auto node : subgraph) {
                hashes.push_back(nodes[node].hash);
                used_nodes.push_back(node);
                used[node] = 1;
            }
        }
    }
    return {std::move(subgraphs), std::move(used_nodes)};
}

CompactedGraph compact_graph(
    const Kmer* kmers,
    const std::vector<Node>& nodes,
    const std::vector<Edge>& edges,
    std::vector<std::size_t> used_nodes // Node indices
) {
    // Restore node order (sorted by hash)
    std::sort(used_nodes.begin(), used_nodes.end());

    CompactedGraph graph;
    graph.nodes = NoInitArray<Node>(used_nodes.size());
    std::size_t n_kmers = 0;
    for (std::size_t i = 0; i < used_nodes.size(); ++i) {
        graph.nodes[i] = nodes[used_nodes[i]];
        n_kmers += graph.nodes[i].stop - graph.nodes[i].start;
    }
    graph.kmers = NoInitArray<Kmer>(n_kmers);

    std::size_t new_start = 0;
    for (auto& node : graph.nodes) {
        const auto old_start = node.start;
        const auto old_stop = node.stop;
        const auto size = old_stop - old_start;
        node.start = new_start;
        node.stop = new_start + size;
        std::copy(kmers + old_start, kmers + old_stop, graph.kmers.begin() + new_start);
        new_start += size;
    }

    ankerl::unordered_dense::set<std::uint64_t> used_hashes;
    used_hashes.reserve(graph.nodes.size());
    for (const auto& node : graph.nodes) used_hashes.insert(node.hash);

    graph.edges.reserve(edges.size());
    for (const auto& edge : edges) {
        if (used_hashes.count(edge.first) && used_hashes.count(edge.second)) {
            graph.edges.push_back(edge);
        }
    }
    return graph;
}

} // namespace seqwin::internal

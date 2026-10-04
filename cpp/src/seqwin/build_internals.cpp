#include "seqwin/build_internals.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "seqwin/shared_internals.hpp"
#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {
namespace {

/**
 * @brief Final k-mer output position for one `WorkerNode`.
 * Used in low-memory mode for constructing `KmerMaps` in parallel.
 */
struct KmerMapEntry {
    std::uint64_t hash;
    std::size_t out_start;
};

template <typename T, typename MemberPtr>
NoInitArray<T> concat(
    std::vector<WorkerGraph>& graphs,
    MemberPtr member,
    ThreadPool& pool
) {
    if (graphs.empty()) {
        return {};
    }

    std::vector<std::size_t> offsets(graphs.size(), 0);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < graphs.size(); ++i) {
        offsets[i] = cursor;
        cursor += (graphs[i].*member).size();
    }
    NoInitArray<T> out(cursor);

    pool.parallel_for(graphs.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t i = start; i < end; ++i) {
            auto& local = graphs[i].*member;
            const auto local_size = local.size();
            if (local_size != 0) {
                std::copy(
                    local.begin(),
                    local.end(),
                    out.begin() + static_cast<std::ptrdiff_t>(offsets[i])
                );
            }
            local.reset();
        }
    });

    return out;
}

NoInitArray<WorkerNode> concat_nodes(
    std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    return concat<WorkerNode>(graphs, &WorkerGraph::nodes, pool);
}

NoInitArray<WorkerEdge> concat_edges(
    std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    return concat<WorkerEdge>(graphs, &WorkerGraph::edges, pool);
}

/**
 * @brief Sort worker-local nodes by hash and collect unique hashes in ascending order.
 *
 * The sorted `WorkerNode` array is subsequently consumed by a node/k-mer merge helper.
 */
static std::pair<std::vector<std::uint64_t>, std::size_t> sort_nodes(
    NoInitArray<WorkerNode>& worker_nodes,
    ThreadPool& pool
) {
    const std::size_t n_worker_nodes = worker_nodes.size();
    if (n_worker_nodes == 0) {
        return {};
    }

    lsd_radix_sort(worker_nodes, &WorkerNode::hash, pool);

    std::vector<std::uint64_t> node_hashes;
    node_hashes.reserve(n_worker_nodes);
    std::size_t i = 0;
    while (i < n_worker_nodes) {
        const auto hash = worker_nodes[i].hash;
        node_hashes.push_back(hash);
        while (i < n_worker_nodes && worker_nodes[i].hash == hash) {
            ++i;
        }
    }
    node_hashes.shrink_to_fit();

    const auto n_nodes = node_hashes.size();
    return {std::move(node_hashes), n_nodes};
}

/**
 * @brief Merge sorted worker nodes and their k-mers into the final arrays.
 *
 * Each `WorkerNode::hash` is repurposed to store its final k-mer output start.
 * The underlying memory of `worker_nodes` and `WorkerGraph::kmers` are released before returning.
 */
static std::pair<NoInitArray<Node>, NoInitArray<Kmer>> merge_standard(
    NoInitArray<WorkerNode>& worker_nodes,
    std::size_t n_nodes,
    std::vector<WorkerGraph>& graphs,
    const std::vector<std::uint32_t>& worker_record_offsets,
    ThreadPool& pool
) {
    const std::size_t n_worker_nodes = worker_nodes.size();
    if (n_worker_nodes == 0) {
        worker_nodes.reset();
        for (auto& graph : graphs) {
            graph.kmers.reset();
        }
        return {};
    }

    // Aggregate nodes and track final k-mer output starts
    NoInitArray<Node> nodes(n_nodes);
    std::size_t n_kmers = 0;
    std::size_t write_i = 0;
    std::size_t i = 0;
    while (i < n_worker_nodes) {
        const auto hash = worker_nodes[i].hash;
        const auto start = n_kmers;

        while (i < n_worker_nodes && worker_nodes[i].hash == hash) {
            const auto count = worker_nodes[i].count();
            worker_nodes[i].hash = n_kmers;
            n_kmers += count;
            ++i;
        }
        nodes[write_i++] = Node{hash, start, 0};
    }
    if (write_i != n_nodes) {
        throw std::logic_error("Merged node count does not match unique node count");
    }

    NoInitArray<Kmer> kmers(n_kmers);
    pool.parallel_for(n_worker_nodes, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t i = start; i < end; ++i) {
            const auto& node = worker_nodes[i];
            const auto worker_id = node.worker_id();
            const auto& local_kmers = graphs[worker_id].kmers;
            const auto offset = worker_record_offsets[worker_id];
            const auto out_start = static_cast<std::size_t>(node.hash);
            const auto count = node.count();

            for (std::size_t k = 0; k < count; ++k) {
                auto kmer = local_kmers[node.start + k];
                kmer.record_idx += offset;
                kmers[out_start + k] = kmer;
            }
        }
    });

    worker_nodes.reset();
    for (auto& graph : graphs) {
        graph.kmers.reset();
    }
    return {std::move(nodes), std::move(kmers)};
}

/**
 * @brief Merge sorted worker nodes and build `KmerMaps` for low-memory recomputation.
 *
 * The underlying memory of `worker_nodes` is released before returning.
 */
static std::pair<NoInitArray<Node>, KmerMaps> merge_low_memory(
    NoInitArray<WorkerNode>& worker_nodes,
    std::size_t n_nodes,
    const std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    KmerMaps kmer_maps(graphs.size());

    const std::size_t n_worker_nodes = worker_nodes.size();
    if (n_worker_nodes == 0) {
        worker_nodes.reset();
        return {NoInitArray<Node>{}, std::move(kmer_maps)};
    }

    // KmerMap entries grouped by worker
    NoInitArray<KmerMapEntry> map_entries(n_worker_nodes);

    // Each worker's contiguous range in map_entries
    std::vector<std::size_t> worker_offsets(graphs.size() + 1);
    for (std::size_t worker_id = 0; worker_id < graphs.size(); ++worker_id) {
        worker_offsets[worker_id + 1] = worker_offsets[worker_id] + graphs[worker_id].n_nodes;
    }
    if (worker_offsets.back() != n_worker_nodes) {
        throw std::logic_error("Worker-node count does not match worker graphs");
    }
    // Write cursors for scattering entries into each worker's range
    auto worker_cursors = worker_offsets;

    // Aggregate nodes and track final k-mer output starts
    NoInitArray<Node> nodes(n_nodes);
    std::size_t n_kmers = 0;
    std::size_t write_i = 0;
    std::size_t i = 0;
    while (i < n_worker_nodes) {
        const auto hash = worker_nodes[i].hash;
        const auto start = n_kmers;

        while (i < n_worker_nodes && worker_nodes[i].hash == hash) {
            const auto worker_id = worker_nodes[i].worker_id();
            const auto count = worker_nodes[i].count();
            map_entries[worker_cursors[worker_id]++] = KmerMapEntry{hash, n_kmers};
            n_kmers += count;
            ++i;
        }
        nodes[write_i++] = Node{hash, start, 0};
    }
    if (write_i != n_nodes) {
        throw std::logic_error("Merged node count does not match unique node count");
    }
    for (std::size_t worker_id = 0; worker_id < graphs.size(); ++worker_id) {
        if (worker_cursors[worker_id] != worker_offsets[worker_id + 1]) {
            throw std::logic_error("Worker-node scatter did not fill worker range");
        }
    }
    worker_nodes.reset();

    pool.parallel_for(graphs.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t worker_id = start; worker_id < end; ++worker_id) {
            KmerMap map;
            map.reserve(graphs[worker_id].n_nodes);
            for (
                std::size_t i = worker_offsets[worker_id];
                i < worker_offsets[worker_id + 1];
                ++i
            ) {
                const auto& entry = map_entries[i];
                map.emplace(entry.hash, entry.out_start);
            }
            kmer_maps[worker_id] = std::move(map);
        }
    });
    return {std::move(nodes), std::move(kmer_maps)};
}

/**
 * @brief Convert worker-local edge endpoints from hashes to node indices, and merge duplicates.
 *
 * `node_hashes` must contain the unique node hashes in ascending order (the final node order).
 * The underlying memory of `worker_edges` and `node_hashes` is released before returning.
 */
static NoInitArray<Edge> finalize_edges(
    NoInitArray<WorkerEdge>& worker_edges,
    std::vector<std::uint64_t>& node_hashes,
    ThreadPool& pool
) {
    const std::size_t n_worker_edges = worker_edges.size();
    const std::size_t n_nodes = node_hashes.size();
    if (n_worker_edges == 0 || n_nodes == 0) {
        worker_edges.reset();
        std::vector<std::uint64_t>().swap(node_hashes);
        return {};
    }

    // Convert the second endpoints to node indices
    lsd_radix_sort(worker_edges, &WorkerEdge::second, pool);
    pool.parallel_for(n_worker_edges, [&](std::size_t start, std::size_t end, std::size_t) {
        if (start == end) {
            return;
        }
        auto node_start = std::lower_bound(
            node_hashes.begin(), node_hashes.end(), worker_edges[start].second
        );
        std::size_t node_i = static_cast<std::size_t>(node_start - node_hashes.begin());
        for (std::size_t i = start; i < end; ++i) {
            while (node_i < n_nodes && node_hashes[node_i] < worker_edges[i].second) {
                ++node_i;
            }
            if (node_i == n_nodes || node_hashes[node_i] != worker_edges[i].second) {
                throw std::logic_error("Edge endpoint does not correspond to a node");
            }
            worker_edges[i].second = node_i;
        }
    });

    // Convert the first endpoints to node indices, and aggregate weights
    lsd_radix_sort(worker_edges, &WorkerEdge::first, pool);
    std::size_t node_i = 0;
    std::size_t n_edges = 0;
    for (auto& edge : worker_edges) {
        while (node_i < n_nodes && node_hashes[node_i] < edge.first) {
            ++node_i;
        }
        if (node_i == n_nodes || node_hashes[node_i] != edge.first) {
            throw std::logic_error("Edge endpoint does not correspond to a node");
        }

        const WorkerEdge converted{node_i, edge.second, edge.weight};
        if (
            n_edges != 0 &&
            worker_edges[n_edges - 1].first == converted.first &&
            worker_edges[n_edges - 1].second == converted.second
        ) {
            worker_edges[n_edges - 1].weight += converted.weight;
        } else {
            worker_edges[n_edges++] = converted;
        }
    }
    std::vector<std::uint64_t>().swap(node_hashes);

    NoInitArray<Edge> edges(n_edges);
    pool.parallel_for(n_edges, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t i = start; i < end; ++i) {
            edges[i] = Edge{
                static_cast<std::size_t>(worker_edges[i].first),
                static_cast<std::size_t>(worker_edges[i].second),
                worker_edges[i].weight
            };
        }
    });
    worker_edges.reset();

    // Sort by descending weight
    lsd_radix_sort(edges, &Edge::weight, false, pool);
    return edges;
}

} // namespace

std::pair<Graph, KmerMaps> merge_worker_graphs(
    std::vector<WorkerGraph>& graphs,
    std::size_t n_assemblies,
    ThreadPool& pool,
    bool low_memory
) {
    if (graphs.size() == 1) {
        auto& graph = graphs[0];

        auto [node_hashes, n_nodes] = sort_nodes(graph.nodes, pool);
        auto edges = finalize_edges(graph.edges, node_hashes, pool);

        NoInitArray<Node> nodes;
        NoInitArray<Kmer> kmers;
        KmerMaps kmer_maps;
        if (low_memory) {
            std::tie(nodes, kmer_maps) = merge_low_memory(
                graph.nodes, n_nodes, graphs, pool
            );
        } else {
            std::tie(nodes, kmers) = merge_standard(
                graph.nodes,
                n_nodes,
                graphs,
                std::vector<std::uint32_t>{0},
                pool
            );
        }

        return {
            Graph{
                std::move(kmers),
                std::move(nodes),
                std::move(edges),
                std::move(graph.record_offsets),
                std::move(graph.record_ids)
            },
            std::move(kmer_maps)
        };
    }

    log_python(" - Merging from " + std::to_string(graphs.size()) + " workers...");

    // Record index offsets in each worker
    std::vector<std::uint32_t> worker_record_offsets(graphs.size());
    // Record index offsets in each assembly
    std::vector<std::uint32_t> record_offsets;
    record_offsets.reserve(n_assemblies + 1);
    record_offsets.push_back(0);

    std::uint32_t total_records = 0;
    for (std::size_t t = 0; t < graphs.size(); ++t) {
        auto& local_offsets = graphs[t].record_offsets;

        const auto base = total_records;
        worker_record_offsets[t] = base;
        if (local_offsets.back() > std::numeric_limits<std::uint32_t>::max() - total_records) {
            throw std::runtime_error("Total number of FASTA records exceeds uint32 range");
        }
        total_records += local_offsets.back();

        for (std::size_t i = 1; i < local_offsets.size(); ++i) {
            record_offsets.push_back(base + local_offsets[i]);
        }
        std::vector<std::uint32_t>().swap(local_offsets);
    }

    auto worker_nodes = concat_nodes(graphs, pool);
    auto [node_hashes, n_nodes] = sort_nodes(worker_nodes, pool);
    auto worker_edges = concat_edges(graphs, pool);
    auto edges = finalize_edges(worker_edges, node_hashes, pool);

    NoInitArray<Node> nodes;
    NoInitArray<Kmer> kmers;
    KmerMaps kmer_maps;
    if (low_memory) {
        std::tie(nodes, kmer_maps) = merge_low_memory(
            worker_nodes, n_nodes, graphs, pool
        );
    } else {
        std::tie(nodes, kmers) = merge_standard(
            worker_nodes, n_nodes, graphs, worker_record_offsets, pool
        );
    }

    std::vector<std::string> record_ids;
    record_ids.reserve(total_records);
    for (auto& graph : graphs) {
        record_ids.insert(
            record_ids.end(),
            std::make_move_iterator(graph.record_ids.begin()),
            std::make_move_iterator(graph.record_ids.end())
        );
        std::vector<std::string>().swap(graph.record_ids);
    }

    return {
        Graph{
            std::move(kmers),
            std::move(nodes),
            std::move(edges),
            std::move(record_offsets),
            std::move(record_ids)
        },
        std::move(kmer_maps)
    };
}

} // namespace seqwin::internal

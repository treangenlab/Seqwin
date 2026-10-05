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
 * @brief Final k-mer output position for one worker-local node.
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

    pool.parallel_for(graphs.size(), [&](std::size_t i) {
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
    });

    return out;
}

NoInitArray<WorkerNode> concat_nodes(
    std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    return concat<WorkerNode>(graphs, &WorkerGraph::nodes, pool);
}

NoInitArray<WorkerNodeLM> concat_nodes_lm(
    std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    return concat<WorkerNodeLM>(graphs, &WorkerGraph::nodes_lm, pool);
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
 * The sorted worker-node array is subsequently consumed by a node/k-mer merge helper.
 */
template <typename WorkerNodeT>
static std::vector<std::uint64_t> sort_nodes(
    NoInitArray<WorkerNodeT>& worker_nodes,
    ThreadPool& pool
) {
    const std::size_t n_worker_nodes = worker_nodes.size();
    if (n_worker_nodes == 0) {
        return {};
    }

    lsd_radix_sort(
        worker_nodes,
        [](const WorkerNodeT& node) {
            return node.hash;
        },
        pool
    );

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
    return node_hashes;
}

/**
 * @brief Merge sorted worker nodes and their k-mers into the final arrays.
 *
 * Each `WorkerNode::hash` is repurposed to store its final k-mer output start.
 * The memory of `worker_nodes` and `WorkerGraph::kmers` is released before returning.
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
        for (auto& worker_graph : graphs) {
            worker_graph.kmers.reset();
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
    pool.parallel_for(n_worker_nodes, [&](std::size_t i) {
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
    });

    worker_nodes.reset();
    for (auto& worker_graph : graphs) {
        worker_graph.kmers.reset();
    }
    return {std::move(nodes), std::move(kmers)};
}

/**
 * @brief Merge sorted worker nodes and build `KmerMaps` for low-memory recomputation.
 *
 * The memory of `worker_nodes` is released before returning.
 */
static std::pair<NoInitArray<Node>, KmerMaps> merge_low_memory(
    NoInitArray<WorkerNodeLM>& worker_nodes,
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

    pool.parallel_for(graphs.size(), [&](std::size_t worker_id) {
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
    });
    return {std::move(nodes), std::move(kmer_maps)};
}

/**
 * @brief Convert worker-local edge endpoints from hashes to node indices, and merge duplicates.
 *
 * `node_hashes` must contain the unique node hashes in ascending order (the final node order).
 * The memory of `worker_edges` and `node_hashes` is released before returning.
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
    pool.parallel_for_chunks(n_worker_edges, [&](
        std::size_t start, std::size_t end, std::size_t
    ) {
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
    std::size_t write_i = 0;
    for (auto& edge : worker_edges) {
        while (node_i < n_nodes && node_hashes[node_i] < edge.first) {
            ++node_i;
        }
        if (node_i == n_nodes || node_hashes[node_i] != edge.first) {
            throw std::logic_error("Edge endpoint does not correspond to a node");
        }

        const WorkerEdge converted{node_i, edge.second, edge.weight};
        if (
            write_i != 0 &&
            worker_edges[write_i - 1].first == converted.first &&
            worker_edges[write_i - 1].second == converted.second
        ) {
            worker_edges[write_i - 1].weight += converted.weight;
        } else {
            worker_edges[write_i++] = converted;
        }
    }
    std::vector<std::uint64_t>().swap(node_hashes);

    NoInitArray<Edge> edges(write_i);
    pool.parallel_for(write_i, [&](std::size_t i) {
        edges[i] = Edge{
            static_cast<std::size_t>(worker_edges[i].first),
            static_cast<std::size_t>(worker_edges[i].second),
            worker_edges[i].weight
        };
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
        auto& worker_graph = graphs[0];

        std::vector<std::uint64_t> node_hashes;
        if (low_memory) {
            node_hashes = sort_nodes(worker_graph.nodes_lm, pool);
        } else {
            node_hashes = sort_nodes(worker_graph.nodes, pool);
        }
        const auto n_nodes = node_hashes.size();
        auto edges = finalize_edges(worker_graph.edges, node_hashes, pool);

        NoInitArray<Node> nodes;
        NoInitArray<Kmer> kmers;
        KmerMaps kmer_maps;
        if (low_memory) {
            std::tie(nodes, kmer_maps) = merge_low_memory(
                worker_graph.nodes_lm, n_nodes, graphs, pool
            );
        } else {
            std::tie(nodes, kmers) = merge_standard(
                worker_graph.nodes,
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
                std::move(worker_graph.record_offsets),
                std::move(worker_graph.record_ids)
            },
            std::move(kmer_maps)
        };
    }

    log_python(" - Merging from " + std::to_string(graphs.size()) + " workers...");

    // Record index offsets in each worker
    std::vector<std::uint32_t> worker_record_offsets(graphs.size());
    // Record index offsets in each assembly
    std::vector<std::uint32_t> record_offsets{0};
    record_offsets.reserve(n_assemblies + 1);

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

    NoInitArray<WorkerNode> worker_nodes;
    NoInitArray<WorkerNodeLM> worker_nodes_lm;
    std::vector<std::uint64_t> node_hashes;
    if (low_memory) {
        worker_nodes_lm = concat_nodes_lm(graphs, pool);
        node_hashes = sort_nodes(worker_nodes_lm, pool);
    } else {
        worker_nodes = concat_nodes(graphs, pool);
        node_hashes = sort_nodes(worker_nodes, pool);
    }
    const auto n_nodes = node_hashes.size();
    auto worker_edges = concat_edges(graphs, pool);
    auto edges = finalize_edges(worker_edges, node_hashes, pool);

    NoInitArray<Node> nodes;
    NoInitArray<Kmer> kmers;
    KmerMaps kmer_maps;
    if (low_memory) {
        std::tie(nodes, kmer_maps) = merge_low_memory(
            worker_nodes_lm, n_nodes, graphs, pool
        );
    } else {
        std::tie(nodes, kmers) = merge_standard(
            worker_nodes, n_nodes, graphs, worker_record_offsets, pool
        );
    }

    std::vector<std::string> record_ids;
    record_ids.reserve(total_records);
    for (auto& worker_graph : graphs) {
        record_ids.insert(
            record_ids.end(),
            std::make_move_iterator(worker_graph.record_ids.begin()),
            std::make_move_iterator(worker_graph.record_ids.end())
        );
        std::vector<std::string>().swap(worker_graph.record_ids);
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

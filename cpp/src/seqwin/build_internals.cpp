#include "seqwin/build_internals.hpp"

#include <algorithm>
#include <atomic>
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
 * @brief Metadata for one contiguous chunk of sorted worker nodes.
 */
struct MergeChunk {
    /**
     * Start of the chunk in the sorted worker nodes.
     * Might be in the middle of a hash run.
     */
    std::size_t start = 0;
    /** End of the chunk in the sorted worker nodes. */
    std::size_t end = 0;
    /** Output offset into `Graph::nodes`. */
    std::size_t node_start = 0;
    /** Number of final nodes whose hash runs start in this chunk. */
    std::size_t node_count = 0;
    /** Output offset into `Graph::kmers`. */
    std::size_t kmer_start = 0;
    /** Number of k-mers represented by the nodes in this chunk. */
    std::size_t kmer_count = 0;
    /** True if the first worker node starts a new hash run. */
    bool first_is_new_node = false;
};

/**
 * @brief Plan for parallel merging of sorted worker nodes.
 */
struct MergePlan {
    /** Unique hashes in ascending order (final node order). */
    NoInitArray<std::uint64_t> node_hashes;
    /** Merge metadata for each chunk. */
    std::vector<MergeChunk> chunks;
    /** Size of `Graph::nodes`. */
    std::size_t n_nodes = 0;
    /** Size of `Graph::kmers`. */
    std::size_t n_kmers = 0;
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
 * @brief Sort worker-local nodes and prepare `MergePlan` for parallel merging.
 */
template <typename WorkerNodeT>
static MergePlan prepare_merge(
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

    const auto n_chunks = std::min(pool.size(), n_worker_nodes);
    MergePlan plan;
    plan.chunks.resize(n_chunks);
    // Hashes whose runs start in each chunk
    std::vector<std::vector<std::uint64_t>> chunk_hashes(n_chunks);

    pool.parallel_for_chunks(n_worker_nodes, [&](
        std::size_t start, std::size_t end, std::size_t chunk_id
    ) {
        MergeChunk chunk;
        chunk.start = start;
        chunk.end = end;
        // A chunk may start in the middle of a hash run
        // Compare its first hash with the preceding global worker node
        chunk.first_is_new_node =
            start == 0 ||
            worker_nodes[start - 1].hash != worker_nodes[start].hash;
        std::vector<std::uint64_t> hashes;
        hashes.reserve(end - start);

        std::uint64_t previous_hash = 0;
        for (std::size_t i = start; i < end; ++i) {
            const auto& worker_node = worker_nodes[i];
            const auto hash = worker_node.hash;

            const bool is_new_node = i == start
                ? chunk.first_is_new_node
                : hash != previous_hash;
            if (is_new_node) {
                ++chunk.node_count;
                hashes.push_back(hash);
            }
            chunk.kmer_count += worker_node.count();
            previous_hash = hash;
        }
        plan.chunks[chunk_id] = chunk;
        chunk_hashes[chunk_id] = std::move(hashes);
    });

    // Prefixing k-mer counts keeps kmer_start correct even when a hash run crosses chunks
    for (auto& chunk : plan.chunks) {
        chunk.node_start = plan.n_nodes;
        chunk.kmer_start = plan.n_kmers;
        plan.n_nodes += chunk.node_count;
        plan.n_kmers += chunk.kmer_count;
    }

    plan.node_hashes = NoInitArray<std::uint64_t>(plan.n_nodes);
    pool.parallel_for(n_chunks, [&](std::size_t chunk_id) {
        const auto& hashes = chunk_hashes[chunk_id];
        const auto& chunk = plan.chunks[chunk_id];
        std::copy(
            hashes.begin(),
            hashes.end(),
            plan.node_hashes.begin() + chunk.node_start
        );
    });
    return plan;
}

/**
 * @brief Apply `MergePlan` to build final nodes and process each worker node.
 */
template <typename WorkerNodeT, typename VisitWorkerNode>
static NoInitArray<Node> apply_merge_plan(
    const NoInitArray<WorkerNodeT>& worker_nodes,
    const MergePlan& plan,
    VisitWorkerNode&& visit_worker_node,
    ThreadPool& pool
) {
    NoInitArray<Node> nodes(plan.n_nodes);
    pool.parallel_for(plan.chunks.size(), [&](std::size_t chunk_id) {
        const auto& chunk = plan.chunks[chunk_id];
        std::size_t node_cursor = chunk.node_start;
        std::size_t kmer_cursor = chunk.kmer_start;

        std::uint64_t previous_hash = 0;
        for (std::size_t i = chunk.start; i < chunk.end; ++i) {
            const auto& worker_node = worker_nodes[i];
            const auto hash = worker_node.hash;

            const bool is_new_node = i == chunk.start
                ? chunk.first_is_new_node
                : hash != previous_hash;
            if (is_new_node) {
                nodes[node_cursor++] = Node{hash, kmer_cursor, 0};
            }
            visit_worker_node(worker_node, hash, kmer_cursor);
            kmer_cursor += worker_node.count();
            previous_hash = hash;
        }
        if (node_cursor != chunk.node_start + chunk.node_count) {
            throw std::logic_error("Merge chunk produced unexpected node count");
        }
        if (kmer_cursor != chunk.kmer_start + chunk.kmer_count) {
            throw std::logic_error("Merge chunk produced unexpected k-mer count");
        }
    });
    return nodes;
}

/**
 * @brief Merge sorted worker nodes and their k-mers into the final arrays.
 *
 * The memory of `worker_nodes` and `WorkerGraph::kmers` is released before returning.
 */
static std::pair<NoInitArray<Node>, NoInitArray<Kmer>> merge_standard(
    NoInitArray<WorkerNode>& worker_nodes,
    const MergePlan& plan,
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

    NoInitArray<Kmer> kmers(plan.n_kmers);
    auto nodes = apply_merge_plan(
        worker_nodes,
        plan,
        [&](const WorkerNode& worker_node, std::uint64_t, std::size_t kmer_cursor) {
            const auto worker_id = worker_node.worker_id();
            const auto& local_kmers = graphs[worker_id].kmers;
            const auto offset = worker_record_offsets[worker_id];
            const auto count = worker_node.count();

            for (std::size_t i = 0; i < count; ++i) {
                auto kmer = local_kmers[worker_node.start + i];
                kmer.record_idx += offset;
                kmers[kmer_cursor + i] = kmer;
            }
        },
        pool
    );

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
    const MergePlan& plan,
    const std::vector<WorkerGraph>& graphs,
    ThreadPool& pool
) {
    // Final k-mer output position for one worker-local node
    struct KmerMapEntry {
        std::uint64_t hash;
        std::size_t out_start;
    };
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
    std::vector<std::atomic<std::size_t>> worker_cursors(graphs.size());
    for (std::size_t worker_id = 0; worker_id < graphs.size(); ++worker_id) {
        worker_cursors[worker_id].store(
            worker_offsets[worker_id], std::memory_order_relaxed
        );
    }

    auto nodes = apply_merge_plan(
        worker_nodes,
        plan,
        [&](const WorkerNodeLM& worker_node, std::uint64_t hash, std::size_t kmer_cursor) {
            const auto worker_id = worker_node.worker_id();
            const auto pos = worker_cursors[worker_id].fetch_add(
                1, std::memory_order_relaxed
            );
            map_entries[pos] = KmerMapEntry{hash, kmer_cursor};
        },
        pool
    );
    for (std::size_t worker_id = 0; worker_id < graphs.size(); ++worker_id) {
        if (
            worker_cursors[worker_id].load(std::memory_order_relaxed) !=
            worker_offsets[worker_id + 1]
        ) {
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
    NoInitArray<std::uint64_t>& node_hashes,
    ThreadPool& pool
) {
    const std::size_t n_worker_edges = worker_edges.size();
    const std::size_t n_nodes = node_hashes.size();
    if (n_worker_edges == 0 || n_nodes == 0) {
        worker_edges.reset();
        node_hashes.reset();
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
    node_hashes.reset();

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

        MergePlan plan;
        if (low_memory) {
            plan = prepare_merge(worker_graph.nodes_lm, pool);
        } else {
            plan = prepare_merge(worker_graph.nodes, pool);
        }
        auto edges = finalize_edges(worker_graph.edges, plan.node_hashes, pool);

        NoInitArray<Node> nodes;
        NoInitArray<Kmer> kmers;
        KmerMaps kmer_maps;
        if (low_memory) {
            std::tie(nodes, kmer_maps) = merge_low_memory(
                worker_graph.nodes_lm, plan, graphs, pool
            );
        } else {
            std::tie(nodes, kmers) = merge_standard(
                worker_graph.nodes, plan, graphs, std::vector<std::uint32_t>{0}, pool
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
    MergePlan plan;
    if (low_memory) {
        worker_nodes_lm = concat_nodes_lm(graphs, pool);
        plan = prepare_merge(worker_nodes_lm, pool);
    } else {
        worker_nodes = concat_nodes(graphs, pool);
        plan = prepare_merge(worker_nodes, pool);
    }
    auto worker_edges = concat_edges(graphs, pool);
    auto edges = finalize_edges(worker_edges, plan.node_hashes, pool);

    NoInitArray<Node> nodes;
    NoInitArray<Kmer> kmers;
    KmerMaps kmer_maps;
    if (low_memory) {
        std::tie(nodes, kmer_maps) = merge_low_memory(
            worker_nodes_lm, plan, graphs, pool
        );
    } else {
        std::tie(nodes, kmers) = merge_standard(
            worker_nodes, plan, graphs, worker_record_offsets, pool
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

#include "seqwin/build_internals.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
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
struct NodeChunk {
    /**
     * Start of this chunk in the sorted worker nodes.
     * Might be in the middle of a hash run.
     */
    std::size_t start = 0;
    /** End of this chunk in the sorted worker nodes. */
    std::size_t end = 0;
    /** Output offset into `Nodes::offsets`. */
    std::size_t node_start = 0;
    /** Number of final nodes whose hash runs start in this chunk. */
    std::size_t node_count = 0;
    /** Output offset into `Nodes::kmers`. */
    std::size_t kmer_start = 0;
    /** Number of k-mers represented by the nodes in this chunk. */
    std::size_t kmer_count = 0;
    /** Number of worker nodes from each worker in this chunk. */
    std::vector<std::size_t> worker_counts;
};

/**
 * @brief Metadata for one contiguous chunk of sorted worker edges.
 */
struct EdgeChunk {
    /** Start of this chunk in `worker_edges`. Might be in the middle of a run. */
    std::size_t start = 0;
    /**
     * Number of valid compacted edges stored starting at `start`.
     * Entries later in the chunk are stale after compaction.
     */
    std::size_t count = 0;
    /** Output offset into the final edge array. */
    std::size_t edge_start = 0;
    /** True if the first compacted edge was merged into an earlier chunk. */
    bool skip_first = false;
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

    std::vector<std::size_t> offsets(graphs.size());
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
 * @brief Sort worker-local nodes, build `Nodes::hashes`, and prepare merge chunks.
 *
 * `n_workers` is used for preparing low-memory metadata. Pass 0 for standard mode.
 */
template <typename WorkerNodeT>
static std::vector<NodeChunk> prepare_merge(
    NoInitArray<WorkerNodeT>& worker_nodes,
    std::size_t n_workers,
    Nodes& nodes,
    ThreadPool& pool
) {
    const std::size_t n_worker_nodes = worker_nodes.size();
    if (n_worker_nodes == 0) {
        nodes.hashes.reset();
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
    std::vector<NodeChunk> chunks(n_chunks);

    pool.parallel_for_chunks(n_worker_nodes, [&](
        std::size_t start, std::size_t end, std::size_t chunk_id
    ) {
        NodeChunk chunk;
        chunk.start = start;
        chunk.end = end;
        if (n_workers != 0) {
            chunk.worker_counts.resize(n_workers, 0);
        }
        for (std::size_t i = start; i < end; ++i) {
            const auto& worker_node = worker_nodes[i];
            if (n_workers != 0) {
                ++chunk.worker_counts[worker_node.worker_id()];
            }
            if (i == 0 || worker_nodes[i - 1].hash != worker_node.hash) {
                ++chunk.node_count;
            }
            chunk.kmer_count += worker_node.count();
        }
        chunks[chunk_id] = std::move(chunk);
    });

    // Prefixing k-mer counts keeps kmer_start correct even when a hash run crosses chunks
    std::size_t node_cursor = 0;
    std::size_t kmer_cursor = 0;
    for (auto& chunk : chunks) {
        chunk.node_start = node_cursor;
        chunk.kmer_start = kmer_cursor;
        node_cursor += chunk.node_count;
        kmer_cursor += chunk.kmer_count;
    }

    nodes.hashes = NoInitArray<std::uint64_t>(node_cursor);
    pool.parallel_for(n_chunks, [&](std::size_t chunk_id) {
        const auto& chunk = chunks[chunk_id];
        std::size_t node_cursor = chunk.node_start;
        for (std::size_t i = chunk.start; i < chunk.end; ++i) {
            if (i == 0 || worker_nodes[i - 1].hash != worker_nodes[i].hash) {
                nodes.hashes[node_cursor++] = worker_nodes[i].hash;
            }
        }
        if (node_cursor != chunk.node_start + chunk.node_count) {
            throw std::logic_error("Merge chunk produced unexpected node count");
        }
    });
    return chunks;
}

/**
 * @brief Build `Nodes::offsets` and process each worker node.
 *
 * Requires non-empty chunks. Callers handle empty graphs.
 */
template <typename WorkerNodeT, typename VisitWorkerNode>
static void merge_node_chunks(
    const NoInitArray<WorkerNodeT>& worker_nodes,
    const std::vector<NodeChunk>& chunks,
    Nodes& nodes,
    ThreadPool& pool,
    VisitWorkerNode&& visit_worker_node
) {
    const std::size_t n_nodes = chunks.back().node_start + chunks.back().node_count;
    nodes.offsets = NoInitArray<std::size_t>(n_nodes + 1);

    pool.parallel_for(chunks.size(), [&](std::size_t chunk_id) {
        const auto& chunk = chunks[chunk_id];
        std::size_t node_cursor = chunk.node_start;
        std::size_t kmer_cursor = chunk.kmer_start;

        for (std::size_t i = chunk.start; i < chunk.end; ++i) {
            const auto& worker_node = worker_nodes[i];
            const auto hash = worker_node.hash;

            if (i == 0 || worker_nodes[i - 1].hash != hash) {
                nodes.offsets[node_cursor++] = kmer_cursor;
            }
            visit_worker_node(worker_node, hash, kmer_cursor, chunk_id);
            kmer_cursor += worker_node.count();
        }
        if (node_cursor != chunk.node_start + chunk.node_count) {
            throw std::logic_error("Merge chunk produced unexpected node count");
        }
        if (kmer_cursor != chunk.kmer_start + chunk.kmer_count) {
            throw std::logic_error("Merge chunk produced unexpected k-mer count");
        }
    });
    nodes.offsets[n_nodes] = chunks.back().kmer_start + chunks.back().kmer_count;
}

/**
 * @brief Merge sorted worker nodes and their k-mers into the final arrays.
 *
 * The memory of `worker_nodes` and `WorkerGraph::kmers` is released before returning.
 */
static void merge_standard(
    NoInitArray<WorkerNode>& worker_nodes,
    const std::vector<NodeChunk>& chunks,
    std::vector<WorkerGraph>& graphs,
    const std::vector<std::uint32_t>& worker_record_offsets,
    Nodes& nodes,
    ThreadPool& pool
) {
    if (worker_nodes.empty()) {
        nodes.kmers.reset();
        nodes.offsets = NoInitArray<std::size_t>(1);
        nodes.offsets[0] = 0;
        return;
    }

    nodes.kmers = NoInitArray<Kmer>(chunks.back().kmer_start + chunks.back().kmer_count);
    merge_node_chunks(worker_nodes, chunks, nodes, pool, [&](
        const WorkerNode& worker_node,
        std::uint64_t,
        std::size_t kmer_cursor,
        std::size_t
    ) {
        const auto worker_id = worker_node.worker_id();
        const auto& local_kmers = graphs[worker_id].kmers;
        const auto offset = worker_record_offsets[worker_id];
        const auto count = worker_node.count();

        for (std::size_t i = 0; i < count; ++i) {
            auto kmer = local_kmers[worker_node.start + i];
            kmer.record_idx += offset;
            nodes.kmers[kmer_cursor + i] = kmer;
        }
    });

    worker_nodes.reset();
    for (auto& worker_graph : graphs) {
        worker_graph.kmers.reset();
    }
}

/**
 * @brief Merge sorted worker nodes and build `KmerMaps` for low-memory recomputation.
 *
 * The memory of `worker_nodes` is released before returning.
 */
static KmerMaps merge_low_memory(
    NoInitArray<WorkerNodeLM>& worker_nodes,
    const std::vector<NodeChunk>& chunks,
    const std::vector<WorkerGraph>& graphs,
    Nodes& nodes,
    ThreadPool& pool
) {
    using MapEntries = KmerMap::value_container_type;

    const std::size_t n_chunks = chunks.size();
    const std::size_t n_workers = graphs.size();
    KmerMaps kmer_maps(n_workers);
    if (worker_nodes.empty()) {
        nodes.offsets = NoInitArray<std::size_t>(1);
        nodes.offsets[0] = 0;
        return kmer_maps;
    }

    // Dense key/value arrays for each KmerMap
    std::vector<MapEntries> map_entries(n_workers);
    pool.parallel_for(n_workers, [&](std::size_t worker_id) {
        MapEntries entries;
        entries.resize(graphs[worker_id].n_nodes);
        map_entries[worker_id] = std::move(entries);
    });

    // Convert per-chunk worker counts to write cursors into map_entries
    NoInitArray<std::size_t> entry_cursors(n_chunks * n_workers);
    for (std::size_t worker_id = 0; worker_id < n_workers; ++worker_id) {
        std::size_t cursor = 0;
        for (std::size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
            entry_cursors[chunk_id * n_workers + worker_id] = cursor;
            cursor += chunks[chunk_id].worker_counts[worker_id];
        }
        if (cursor != graphs[worker_id].n_nodes) {
            throw std::logic_error("Worker-node scatter count does not match worker graph");
        }
    }

    merge_node_chunks(worker_nodes, chunks, nodes, pool, [&](
        const WorkerNodeLM& worker_node,
        std::uint64_t hash,
        std::size_t kmer_cursor,
        std::size_t chunk_id
    ) {
        const auto worker_id = worker_node.worker_id();
        auto& cursor = entry_cursors[chunk_id * n_workers + worker_id];
        map_entries[worker_id][cursor++] = {hash, kmer_cursor};
    });
    worker_nodes.reset();

    // Adopt the populated dense arrays and build bucket metadata
    pool.parallel_for(n_workers, [&](std::size_t worker_id) {
        KmerMap map;
        map.replace(std::move(map_entries[worker_id]));
        kmer_maps[worker_id] = std::move(map);
    });
    return kmer_maps;
}

/**
 * @brief Convert worker-local edge endpoints from hashes to node indices, and merge duplicates.
 *
 * `hashes` must contain the unique node hashes in ascending order (the final node order).
 * The memory of `worker_edges` is released before returning.
 */
static NoInitArray<Edge> finalize_edges(
    NoInitArray<WorkerEdge>& worker_edges,
    const NoInitArray<std::uint64_t>& hashes,
    ThreadPool& pool
) {
    const std::size_t n_worker_edges = worker_edges.size();
    const std::size_t n_nodes = hashes.size();
    if (n_worker_edges == 0) {
        return {};
    }
    if (n_nodes == 0) {
        throw std::logic_error("Edge endpoint does not correspond to a node");
    }

    // Convert second endpoints to node indices
    lsd_radix_sort(worker_edges, &WorkerEdge::second, pool);
    pool.parallel_for_chunks(n_worker_edges, [&](
        std::size_t start, std::size_t end, std::size_t
    ) {
        auto node_start = std::lower_bound(
            hashes.begin(), hashes.end(), worker_edges[start].second
        );
        std::size_t node_idx = static_cast<std::size_t>(node_start - hashes.begin());
        for (std::size_t i = start; i < end; ++i) {
            while (node_idx < n_nodes && hashes[node_idx] < worker_edges[i].second) {
                ++node_idx;
            }
            if (node_idx == n_nodes || hashes[node_idx] != worker_edges[i].second) {
                throw std::logic_error("Edge endpoint does not correspond to a node");
            }
            worker_edges[i].second = node_idx;
        }
    });

    // Duplicate edges are contiguous after the second sort
    lsd_radix_sort(worker_edges, &WorkerEdge::first, pool);
    // Convert first endpoints and compact duplicate runs in each chunk, in parallel
    // Runs crossing chunk boundaries are handled later
    std::vector<EdgeChunk> chunks(std::min(pool.size(), n_worker_edges));
    pool.parallel_for_chunks(n_worker_edges, [&](
        std::size_t start, std::size_t end, std::size_t chunk_id
    ) {
        auto node_start = std::lower_bound(
            hashes.begin(), hashes.end(), worker_edges[start].first
        );
        std::size_t node_idx = static_cast<std::size_t>(node_start - hashes.begin());
        // Compacted edges are written to the prefix of this chunk
        std::size_t write_i = start;
        for (std::size_t i = start; i < end; ++i) {
            auto& worker_edge = worker_edges[i];
            while (node_idx < n_nodes && hashes[node_idx] < worker_edge.first) {
                ++node_idx;
            }
            if (node_idx == n_nodes || hashes[node_idx] != worker_edge.first) {
                throw std::logic_error("Edge endpoint does not correspond to a node");
            }

            if (
                write_i != start &&
                worker_edges[write_i - 1].first == node_idx &&
                worker_edges[write_i - 1].second == worker_edge.second
            ) {
                worker_edges[write_i - 1].weight += worker_edge.weight;
            } else {
                worker_edges[write_i++] = WorkerEdge{
                    node_idx, worker_edge.second, worker_edge.weight
                };
            }
        }
        auto& chunk = chunks[chunk_id];
        chunk.start = start;
        chunk.count = write_i - start;
    });

    // Merge duplicate runs crossing chunk boundaries
    // Now we only care about the first EdgeChunk::count edges of each chunk
    // Entries after that are stale
    std::size_t n_edges = 0;
    std::size_t previous_i = 0;
    bool have_previous = false;
    for (auto& chunk : chunks) {
        auto& first = worker_edges[chunk.start];
        if (
            have_previous &&
            worker_edges[previous_i].first == first.first &&
            worker_edges[previous_i].second == first.second
        ) {
            // If the first edge in this chunk is a duplicate,
            // fold its weight into the last edge of an earlier chunk
            worker_edges[previous_i].weight += first.weight;
            chunk.skip_first = true;
        }

        chunk.edge_start = n_edges;
        const std::size_t skip = chunk.skip_first;
        n_edges += chunk.count - skip;

        // If this chunk contains only a continuation of a duplicate run, keep
        // previous_i at the earlier owner so the run can span many chunks.
        if (chunk.count > skip) {
            previous_i = chunk.start + chunk.count - 1;
            have_previous = true;
        }
    }

    NoInitArray<Edge> edges(n_edges);
    pool.parallel_for(chunks.size(), [&](std::size_t chunk_id) {
        const auto& chunk = chunks[chunk_id];
        const std::size_t skip = chunk.skip_first;

        for (std::size_t i = skip; i < chunk.count; ++i) {
            const auto& edge = worker_edges[chunk.start + i];
            edges[chunk.edge_start + i - skip] = Edge{
                static_cast<std::size_t>(edge.first),
                static_cast<std::size_t>(edge.second),
                edge.weight
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
        auto& worker_graph = graphs[0];

        Nodes nodes;
        std::vector<NodeChunk> chunks;
        if (low_memory) {
            chunks = prepare_merge(worker_graph.nodes_lm, graphs.size(), nodes, pool);
        } else {
            chunks = prepare_merge(worker_graph.nodes, 0, nodes, pool);
        }
        auto edges = finalize_edges(worker_graph.edges, nodes.hashes, pool);

        KmerMaps kmer_maps;
        if (low_memory) {
            kmer_maps = merge_low_memory(worker_graph.nodes_lm, chunks, graphs, nodes, pool);
        } else {
            merge_standard(
                worker_graph.nodes, chunks, graphs, std::vector<std::uint32_t>{0}, nodes, pool
            );
        }

        return {
            Graph{
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
    Nodes nodes;
    std::vector<NodeChunk> chunks;
    if (low_memory) {
        worker_nodes_lm = concat_nodes_lm(graphs, pool);
        chunks = prepare_merge(worker_nodes_lm, graphs.size(), nodes, pool);
    } else {
        worker_nodes = concat_nodes(graphs, pool);
        chunks = prepare_merge(worker_nodes, 0, nodes, pool);
    }
    auto worker_edges = concat_edges(graphs, pool);
    auto edges = finalize_edges(worker_edges, nodes.hashes, pool);

    KmerMaps kmer_maps;
    if (low_memory) {
        kmer_maps = merge_low_memory(worker_nodes_lm, chunks, graphs, nodes, pool);
    } else {
        merge_standard(worker_nodes, chunks, graphs, worker_record_offsets, nodes, pool);
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
            std::move(nodes),
            std::move(edges),
            std::move(record_offsets),
            std::move(record_ids)
        },
        std::move(kmer_maps)
    };
}

} // namespace seqwin::internal

#include "seqwin/build_internals.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {
namespace {

/**
 * @brief Describes a contiguous worker-local k-mer segment and its output position.
 */
struct KmerSegment {
    std::size_t worker_id;
    std::size_t local_start;
    std::size_t out_start;
    std::size_t count;
};

struct MergedNodes {
    NoInitArray<Node> nodes;
    std::vector<KmerSegment> kmer_segments;
    KmerMaps kmer_maps;
};

template <typename T, typename MemberPtr>
NoInitArray<T> concat(std::vector<WorkerGraph>& graphs, MemberPtr member, ThreadPool& pool)
{
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

NoInitArray<WorkerNode> concat_nodes(std::vector<WorkerGraph>& graphs, ThreadPool& pool)
{
    return concat<WorkerNode>(graphs, &WorkerGraph::nodes, pool);
}

NoInitArray<WorkerEdge> concat_edges(std::vector<WorkerGraph>& graphs, ThreadPool& pool)
{
    return concat<WorkerEdge>(graphs, &WorkerGraph::edges, pool);
}

template <typename T, typename KeyPtr>
static void lsd_radix_sort_key(
    T*& src,
    T*& dst,
    std::size_t n,
    KeyPtr key,
    bool ascending,
    std::vector<std::size_t>& counts,
    ThreadPool& pool
) {
    static constexpr std::size_t bucket_count = 65536;
    static constexpr std::uint64_t bucket_mask = bucket_count - 1;

    for (std::size_t shift = 0; shift < 64; shift += 16) {
        std::fill(counts.begin(), counts.end(), 0);

        pool.parallel_for(n, [&](std::size_t start, std::size_t end, std::size_t t) {
            auto* local_counts = counts.data() + t * bucket_count;
            for (std::size_t i = start; i < end; ++i) {
                const auto bucket = ((src[i].*key) >> shift) & bucket_mask;
                ++local_counts[static_cast<std::size_t>(bucket)];
            }
        });

        std::size_t current = 0;
        for (std::size_t bucket_i = 0; bucket_i < bucket_count; ++bucket_i) {
            const auto bucket = ascending ? bucket_i : bucket_count - bucket_i - 1;
            for (std::size_t t = 0; t < pool.size(); ++t) {
                auto& value = counts[t * bucket_count + bucket];
                const auto c = value;
                value = current;
                current += c;
            }
        }

        pool.parallel_for(n, [&](std::size_t start, std::size_t end, std::size_t t) {
            auto* local_offsets = counts.data() + t * bucket_count;
            for (std::size_t i = start; i < end; ++i) {
                const auto bucket = ((src[i].*key) >> shift) & bucket_mask;
                const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                dst[pos] = src[i];
            }
        });

        std::swap(src, dst);
    }
}

/**
 * @brief Stable parallel LSD radix sort over one or more 64-bit member keys.
 *
 * Keys should be supplied in least-significant to most-significant order.
 */
template <typename T, typename... KeyPtrs>
static void lsd_radix_sort(
    NoInitArray<T>& values,
    bool ascending,
    ThreadPool& pool,
    KeyPtrs... keys
) {
    const std::size_t n = values.size();
    if (n == 0) {
        return;
    }

    NoInitArray<T> buf(n);
    auto* src = values.data();
    auto* dst = buf.data();
    std::vector<std::size_t> counts(pool.size() * 65536);

    (lsd_radix_sort_key(src, dst, n, keys, ascending, counts, pool), ...);
}

/**
 * @brief Sort worker-local nodes by hash and collect unique hashes in ascending order.
 *
 * The sorted `WorkerNode` array is subsequently consumed by `merge_nodes()`.
 */
static std::pair<std::vector<std::uint64_t>, std::size_t> sort_nodes(
    NoInitArray<WorkerNode>& nodes,
    ThreadPool& pool
) {
    const std::size_t n_nodes = nodes.size();
    if (n_nodes == 0) {
        return {};
    }

    lsd_radix_sort(nodes, true, pool, &WorkerNode::hash);

    std::vector<std::uint64_t> node_hashes;
    node_hashes.reserve(n_nodes);
    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = nodes[i].hash;
        node_hashes.push_back(hash);
        while (i < n_nodes && nodes[i].hash == hash) {
            ++i;
        }
    }
    node_hashes.shrink_to_fit();

    const auto unique_count = node_hashes.size();
    return {std::move(node_hashes), unique_count};
}

/**
 * @brief Materialize final nodes and k-mer metadata from sorted worker nodes.
 *
 * The input must already be sorted by hash. Nodes with identical hashes are
 * merged while producing either k-mer segments or low-memory k-mer maps.
 */
static MergedNodes merge_nodes(
    const NoInitArray<WorkerNode>& nodes,
    std::size_t unique_count,
    const std::vector<WorkerGraph>& graphs,
    bool low_memory
) {
    MergedNodes merged;
    if (low_memory) {
        merged.kmer_maps = KmerMaps(graphs.size());
    }

    const std::size_t n_nodes = nodes.size();
    if (n_nodes == 0) {
        return merged;
    }

    merged.nodes = NoInitArray<Node>(unique_count);
    if (low_memory) {
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            merged.kmer_maps[i].reserve(graphs[i].n_nodes);
        }
    } else {
        merged.kmer_segments.reserve(n_nodes);
    }

    // Aggregate nodes and track final kmer output positions
    std::size_t n_kmers = 0;
    std::size_t write_i = 0;
    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = nodes[i].hash;
        const auto start = n_kmers;

        while (i < n_nodes && nodes[i].hash == hash) {
            const auto count = nodes[i].count;

            if (low_memory) {
                merged.kmer_maps[nodes[i].worker_id][hash] = n_kmers;
            } else {
                merged.kmer_segments.push_back(KmerSegment{
                    nodes[i].worker_id,
                    nodes[i].start,
                    n_kmers,
                    count
                });
            }
            n_kmers += count;
            ++i;
        }

        merged.nodes[write_i++] = Node{hash, start, n_kmers};
    }
    return merged;
}

static NoInitArray<Kmer> merge_kmers(
    const std::vector<WorkerGraph>& graphs,
    const std::vector<KmerSegment>& kmer_segments,
    const std::vector<std::uint32_t>& worker_record_offsets,
    ThreadPool& pool
) {
    std::size_t total_kmers = 0;
    for (const auto& graph : graphs) {
        total_kmers += graph.n_kmers;
    }
    NoInitArray<Kmer> kmers(total_kmers);

    pool.parallel_for(kmer_segments.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t s = start; s < end; ++s) {
            const auto& segment = kmer_segments[s];
            const auto& local_kmers = graphs[segment.worker_id].kmers;
            const auto offset = worker_record_offsets[segment.worker_id];

            for (std::size_t k = 0; k < segment.count; ++k) {
                auto kmer = local_kmers[segment.local_start + k];
                kmer.record_idx += offset;
                kmers[segment.out_start + k] = kmer;
            }
        }
    });
    return kmers;
}

/**
 * @brief Convert worker-local edge endpoints from hashes to node indices, and merge duplicates.
 *
 * `node_hashes` must contain the unique node hashes in ascending order (the final node order).
 * The underlying memory of `edges` and `node_hashes` is released before returning.
 */
static NoInitArray<Edge> finalize_edges(
    NoInitArray<WorkerEdge>& edges,
    std::vector<std::uint64_t>& node_hashes,
    ThreadPool& pool
) {
    const std::size_t n_edges = edges.size();
    const std::size_t n_nodes = node_hashes.size();
    if (n_edges == 0 || n_nodes == 0) {
        edges.reset();
        std::vector<std::uint64_t>().swap(node_hashes);
        return {};
    }

    // Convert the second endpoints to node indices
    lsd_radix_sort(edges, true, pool, &WorkerEdge::second);
    std::size_t node_i = 0;
    for (auto& edge : edges) {
        while (node_i < n_nodes && node_hashes[node_i] < edge.second) {
            ++node_i;
        }
        if (node_i == n_nodes || node_hashes[node_i] != edge.second) {
            throw std::logic_error("Edge endpoint does not correspond to a node");
        }
        edge.second = node_i;
    }

    // Convert the first endpoints to node indices, and aggregate weights
    lsd_radix_sort(edges, true, pool, &WorkerEdge::first);
    node_i = 0;
    std::size_t write_i = 0;
    for (auto& edge : edges) {
        while (node_i < n_nodes && node_hashes[node_i] < edge.first) {
            ++node_i;
        }
        if (node_i == n_nodes || node_hashes[node_i] != edge.first) {
            throw std::logic_error("Edge endpoint does not correspond to a node");
        }

        const WorkerEdge converted{node_i, edge.second, edge.weight};
        if (
            write_i != 0 &&
            edges[write_i - 1].first == converted.first &&
            edges[write_i - 1].second == converted.second
        ) {
            edges[write_i - 1].weight += converted.weight;
        } else {
            edges[write_i++] = converted;
        }
    }
    std::vector<std::uint64_t>().swap(node_hashes);

    NoInitArray<Edge> out(write_i);
    for (std::size_t i = 0; i < write_i; ++i) {
        out[i] = Edge{
            static_cast<std::size_t>(edges[i].first),
            static_cast<std::size_t>(edges[i].second),
            edges[i].weight
        };
    }
    edges.reset();

    // Sort by descending weight
    lsd_radix_sort(out, false, pool, &Edge::weight);
    return out;
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

        auto [node_hashes, unique_count] = sort_nodes(graph.nodes, pool);
        auto edges = finalize_edges(graph.edges, node_hashes, pool);

        auto merged = merge_nodes(graph.nodes, unique_count, graphs, low_memory);
        graph.nodes.reset();

        NoInitArray<Kmer> kmers;
        if (!low_memory) {
            kmers = merge_kmers(
                graphs,
                merged.kmer_segments,
                std::vector<std::uint32_t>{0},
                pool
            );
            graph.kmers.reset();
            std::vector<KmerSegment>().swap(merged.kmer_segments);
        }

        return {
            Graph{
                std::move(kmers),
                std::move(merged.nodes),
                std::move(edges),
                std::move(graph.record_offsets),
                std::move(graph.record_ids)
            },
            std::move(merged.kmer_maps)
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
    auto [node_hashes, unique_count] = sort_nodes(worker_nodes, pool);
    auto worker_edges = concat_edges(graphs, pool);
    auto edges = finalize_edges(worker_edges, node_hashes, pool);

    auto merged = merge_nodes(worker_nodes, unique_count, graphs, low_memory);
    worker_nodes.reset();

    NoInitArray<Kmer> kmers;
    if (!low_memory) {
        kmers = merge_kmers(graphs, merged.kmer_segments, worker_record_offsets, pool);
        for (auto& graph : graphs) {
            graph.kmers.reset();
        }
        std::vector<KmerSegment>().swap(merged.kmer_segments);
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
            std::move(merged.nodes),
            std::move(edges),
            std::move(record_offsets),
            std::move(record_ids)
        },
        std::move(merged.kmer_maps)
    };
}

} // namespace seqwin::internal

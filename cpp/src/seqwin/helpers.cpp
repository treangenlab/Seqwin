#include "seqwin/helpers.hpp"
#include "seqwin/thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace seqwin {
namespace {

/**
 * @brief Describes a contiguous thread-local k-mer segment and its output position.
 */
struct KmerSegment {
    std::size_t thread_id;
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
NoInitArray<T> concat(std::vector<ThreadGraph>& graphs, MemberPtr member, ThreadPool& pool)
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

NoInitArray<ThreadNode> concat_nodes(std::vector<ThreadGraph>& graphs, ThreadPool& pool)
{
    return concat<ThreadNode>(graphs, &ThreadGraph::nodes, pool);
}

NoInitArray<Edge> concat_edges(std::vector<ThreadGraph>& graphs, ThreadPool& pool)
{
    return concat<Edge>(graphs, &ThreadGraph::edges, pool);
}

template <typename T, typename KeyPtr>
static void lsd_radix_sort_key(
    T*& src,
    T*& dst,
    std::size_t n,
    KeyPtr key,
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
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
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

    (lsd_radix_sort_key(src, dst, n, keys, counts, pool), ...);
}

/**
 * @brief Sort and merge thread-local nodes with identical hashes.
 *
 * 1. Sort nodes by hash.
 * 2. Merge nodes with the same hash into final graph nodes.
 * 3. Track k-mer output positions to build the final `kmers`.
 *
 * @param nodes Thread-local node array to sort and merge.
 * @param graphs Thread-local graphs used to size k-mer maps.
 * @param pool Thread pool used for radix sorting.
 * @param low_memory Whether to emit k-mer maps or k-mer segments.
 * @return Final merged nodes and mode-specific k-mer position metadata.
 */
static MergedNodes merge_nodes(
    NoInitArray<ThreadNode>& nodes,
    const std::vector<ThreadGraph>& graphs,
    ThreadPool& pool,
    bool low_memory
) {
    MergedNodes merged;
    merged.kmer_maps = KmerMaps(graphs.size());

    const std::size_t n_nodes = nodes.size();
    if (n_nodes == 0) {
        return merged;
    }

    lsd_radix_sort(nodes, pool, &ThreadNode::hash);

    // Determine final node count
    std::size_t unique_count = 0;
    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = nodes[i].hash;
        while (i < n_nodes && nodes[i].hash == hash) {
            ++i;
        }
        ++unique_count;
    }
    merged.nodes = NoInitArray<Node>(unique_count);

    // Aggregate nodes and track final kmer output positions
    if (low_memory) {
        for (std::size_t i = 0; i < graphs.size(); ++i) {
            merged.kmer_maps[i].reserve(graphs[i].n_nodes);
        }
    } else {
        merged.kmer_segments.reserve(n_nodes);
    }

    std::size_t n_kmers = 0;
    std::size_t write_i = 0;
    i = 0;
    while (i < n_nodes) {
        const auto hash = nodes[i].hash;
        std::uint32_t n_tar = 0;
        std::uint32_t n_neg = 0;
        const auto start = n_kmers;

        while (i < n_nodes && nodes[i].hash == hash) {
            n_tar += nodes[i].n_tar;
            n_neg += nodes[i].n_neg;
            const auto count = nodes[i].count;

            if (low_memory) {
                merged.kmer_maps[nodes[i].thread_id][hash] = n_kmers;
            } else {
                merged.kmer_segments.push_back(KmerSegment{
                    nodes[i].thread_id,
                    nodes[i].start,
                    n_kmers,
                    count
                });
            }
            n_kmers += count;
            ++i;
        }

        merged.nodes[write_i++] = Node{hash, start, n_kmers, n_tar, n_neg};
    }
    return merged;
}

static NoInitArray<Kmer> merge_kmers(
    const std::vector<ThreadGraph>& graphs,
    const std::vector<KmerSegment>& kmer_segments,
    const std::vector<std::uint32_t>& thread_record_offsets,
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
            const auto& local_kmers = graphs[segment.thread_id].kmers;
            const auto offset = thread_record_offsets[segment.thread_id];

            for (std::size_t k = 0; k < segment.count; ++k) {
                auto kmer = local_kmers[segment.local_start + k];
                kmer.record_idx += offset;
                kmers[segment.out_start + k] = kmer;
            }
        }
    });
    return kmers;
}

static void merge_edges(NoInitArray<Edge>& edges, ThreadPool& pool)
{
    const std::size_t n_edges = edges.size();
    if (n_edges == 0) {
        edges.reset();
        return;
    }

    lsd_radix_sort(edges, pool, &Edge::second, &Edge::first);

    // Determine final edge count
    std::size_t unique_count = 0;
    std::size_t i = 0;
    while (i < n_edges) {
        const auto first = edges[i].first;
        const auto second = edges[i].second;
        while (i < n_edges && edges[i].first == first && edges[i].second == second) {
            ++i;
        }
        ++unique_count;
    }
    NoInitArray<Edge> buf(unique_count);

    std::size_t write_i = 0;
    i = 0;
    while (i < n_edges) {
        const auto first = edges[i].first;
        const auto second = edges[i].second;
        std::size_t weight = 0;

        while (i < n_edges && edges[i].first == first && edges[i].second == second) {
            weight += edges[i].weight;
            ++i;
        }

        buf[write_i++] = Edge{first, second, weight};
    }
    edges = std::move(buf);
}

} // namespace

void log_python(const std::string& message, const std::string& level)
{
    py::gil_scoped_acquire acquire;

    py::object logging = py::module_::import("logging");
    py::object logger = logging.attr("getLogger")();

    if (level == "debug") {
        logger.attr("debug")(message);
    } else if (level == "info") {
        logger.attr("info")(message);
    } else if (level == "warning" || level == "warn") {
        logger.attr("warning")(message);
    } else if (level == "error") {
        logger.attr("error")(message);
    } else if (level == "critical") {
        logger.attr("critical")(message);
    } else {
        logger.attr("info")(message);
    }
}

std::pair<Graph, KmerMaps> merge_thread_graphs(
    std::vector<ThreadGraph>& graphs,
    std::size_t n_assemblies,
    ThreadPool& pool,
    bool low_memory
) {
    if (graphs.size() == 1) {
        auto& graph = graphs[0];
        if (graph.record_offsets.back() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Total number of FASTA records exceeds uint32 range");
        }

        merge_edges(graph.edges, pool); // Sort only

        auto merged = merge_nodes(graph.nodes, graphs, pool, low_memory);
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
                std::move(graph.edges),
                std::move(graph.record_offsets),
                std::move(graph.ids_by_assembly)
            },
            std::move(merged.kmer_maps)
        };
    }

    log_python(" - Merging from " + std::to_string(graphs.size()) + " threads...");

    // Merge record offsets
    std::vector<std::uint32_t> thread_record_offsets(graphs.size());
    std::vector<std::size_t> record_offsets;
    record_offsets.reserve(n_assemblies + 1);
    record_offsets.push_back(0);

    std::size_t total_records = 0;
    for (std::size_t t = 0; t < graphs.size(); ++t) {
        auto& local_offsets = graphs[t].record_offsets;

        const auto base = total_records;
        thread_record_offsets[t] = static_cast<std::uint32_t>(base);
        total_records += local_offsets.back();

        for (std::size_t i = 1; i < local_offsets.size(); ++i) {
            record_offsets.push_back(base + local_offsets[i]);
        }
        std::vector<std::size_t>().swap(local_offsets);
    }
    if (total_records > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Total number of FASTA records exceeds uint32 range");
    }

    // Merge edges and nodes first to reduce peak memory
    auto edges = concat_edges(graphs, pool);
    merge_edges(edges, pool);

    auto thread_nodes = concat_nodes(graphs, pool);
    auto merged = merge_nodes(thread_nodes, graphs, pool, low_memory);
    thread_nodes.reset();

    NoInitArray<Kmer> kmers;
    if (!low_memory) {
        kmers = merge_kmers(graphs, merged.kmer_segments, thread_record_offsets, pool);
        for (auto& graph : graphs) {
            graph.kmers.reset();
        }
        std::vector<KmerSegment>().swap(merged.kmer_segments);
    }

    std::vector<std::vector<std::string>> ids_by_assembly;
    ids_by_assembly.reserve(n_assemblies);
    for (auto& graph : graphs) {
        for (auto& ids : graph.ids_by_assembly) {
            ids_by_assembly.push_back(std::move(ids));
        }
    }

    return {
        Graph{
            std::move(kmers),
            std::move(merged.nodes),
            std::move(edges),
            std::move(record_offsets),
            std::move(ids_by_assembly)
        },
        std::move(merged.kmer_maps)
    };
}

Graph filter_kmers(
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
) {
    std::sort(used_hashes.begin(), used_hashes.end());

    Graph graph;

    std::vector<std::size_t> used_node_indices;
    used_node_indices.reserve(used_hashes.size());

    std::size_t n_kmers = 0;
    std::size_t node_i = 0;
    std::size_t used_i = 0;
    while (node_i < n_nodes && used_i < used_hashes.size()) {
        const auto node_hash = nodes[node_i].hash;
        const auto used_hash = used_hashes[used_i];

        if (node_hash < used_hash) {
            ++node_i;
            continue;
        }
        if (used_hash < node_hash) {
            ++used_i;
            continue;
        }

        used_node_indices.push_back(node_i);
        n_kmers += nodes[node_i].stop - nodes[node_i].start;
        ++node_i;
        ++used_i;
    }

    graph.nodes = NoInitArray<Node>(used_node_indices.size());
    graph.kmers = NoInitArray<Kmer>(n_kmers);

    std::size_t new_start = 0;
    for (std::size_t out_node_i = 0; out_node_i < used_node_indices.size(); ++out_node_i) {
        const auto in_node_i = used_node_indices[out_node_i];
        const Node& old_node = nodes[in_node_i];

        const auto old_start = old_node.start;
        const auto old_stop = old_node.stop;
        const auto size = old_stop - old_start;

        Node new_node = old_node;
        new_node.start = new_start;
        new_node.stop = new_start + size;
        graph.nodes[out_node_i] = new_node;

        for (std::size_t k = 0; k < size; ++k) {
            const auto out_i = new_start + k;
            const auto in_i = old_start + k;
            graph.kmers[out_i] = kmers[in_i];
        }

        new_start += size;
    }

    return graph;
}

} // namespace seqwin

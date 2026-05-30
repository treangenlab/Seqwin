#include "seqwin/helpers.hpp"
#include "seqwin/thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace seqwin {
namespace {

/**
 * @brief Describes a contiguous local `idx` segment and its output position.
 */
struct IdxSegment {
    std::size_t thread_id;
    std::size_t local_start;
    std::size_t out_start;
    std::size_t length;
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

NoInitArray<Node> concat_nodes(std::vector<ThreadGraph>& graphs, ThreadPool& pool)
{
    return concat<Node>(graphs, &ThreadGraph::nodes, pool);
}

NoInitArray<Edge> concat_edges(std::vector<ThreadGraph>& graphs, ThreadPool& pool)
{
    return concat<Edge>(graphs, &ThreadGraph::edges, pool);
}

/**
 * @brief Sort and merge nodes with identical hashes.
 *
 * 1. Parallel LSD radix sort on node hash.
 * 2. Merge nodes with the same hash.
 * 3. Return local `idx` segment metadata to materialize `kmers`.
 *
 * @param nodes Node array to sort and merge.
 * @param pool Thread pool used for radix sorting.
 * @return Segment metadata for rebuilding `kmers` and `idx`.
 */
static std::vector<IdxSegment> merge_nodes(
    NoInitArray<Node>& nodes,
    ThreadPool& pool
) {
    const std::size_t n_nodes = nodes.size();
    if (n_nodes == 0) {
        return {};
    }

    NoInitArray<Node> buf(nodes.size());
    auto* src = nodes.data();
    auto* dst = buf.data();
    std::vector<std::uint64_t> counts(pool.size() * 65536);

    for (std::size_t shift = 0; shift < 64; shift += 16) {
        std::fill(counts.begin(), counts.end(), 0);

        pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t t) {
            auto* local_counts = counts.data() + t * 65536;
            for (std::size_t i = start; i < end; ++i) {
                const auto bucket = (src[i].hash >> shift) & 0xFFFFULL;
                ++local_counts[static_cast<std::size_t>(bucket)];
            }
        });

        std::uint64_t current = 0;
        for (std::size_t bucket = 0; bucket < 65536; ++bucket) {
            for (std::size_t t = 0; t < pool.size(); ++t) {
                auto& value = counts[t * 65536 + bucket];
                const auto c = value;
                value = current;
                current += c;
            }
        }

        pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t t) {
            auto* local_offsets = counts.data() + t * 65536;
            for (std::size_t i = start; i < end; ++i) {
                const auto bucket = (src[i].hash >> shift) & 0xFFFFULL;
                const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                dst[pos] = src[i];
            }
        });

        std::swap(src, dst);
    }

    // Determine final node count
    std::size_t unique_count = 0;
    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = src[i].hash;
        while (i < n_nodes && src[i].hash == hash) {
            ++i;
        }
        ++unique_count;
    }
    buf.reset();
    buf = NoInitArray<Node>(unique_count);

    // Aggregate nodes and keep idx ranges
    std::vector<IdxSegment> idx_segments;
    idx_segments.reserve(n_nodes);

    std::uint64_t n_kmers = 0;
    std::size_t write_i = 0;
    i = 0;
    while (i < n_nodes) {
        const auto hash = src[i].hash;
        std::uint32_t n_tar = 0;
        std::uint32_t n_neg = 0;
        const auto start = n_kmers;

        while (i < n_nodes && src[i].hash == hash) {
            n_tar += src[i].n_tar;
            n_neg += src[i].n_neg;

            const auto local_start = src[i].start;
            const auto local_stop = src[i].stop;
            const auto length = local_stop - local_start;

            if (length != 0) {
                idx_segments.push_back(IdxSegment{
                    static_cast<std::size_t>(src[i].penalty), // thread_id
                    static_cast<std::size_t>(local_start),
                    static_cast<std::size_t>(n_kmers),
                    static_cast<std::size_t>(length)
                });
                n_kmers += length;
            }
            ++i;
        }

        const auto stop = n_kmers;
        buf[write_i++] = Node{hash, start, stop, n_tar, n_neg, 0.0};
    }
    nodes = std::move(buf);
    return idx_segments;
}

static NoInitArray<Kmer> merge_kmers(
    const std::vector<IdxSegment>& idx_segments,
    std::vector<ThreadGraph>& graphs,
    std::size_t total_kmers,
    const std::vector<std::uint64_t>& thread_record_offsets,
    ThreadPool& pool
) {
    NoInitArray<Kmer> kmers(total_kmers);

    pool.parallel_for(idx_segments.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t s = start; s < end; ++s) {
            const auto& segment = idx_segments[s];
            const auto& local_idx = graphs[segment.thread_id].idx;
            const auto& local_kmers = graphs[segment.thread_id].kmers;
            const auto offset = thread_record_offsets[segment.thread_id];

            for (std::size_t k = 0; k < segment.length; ++k) {
                const auto local_kmer_i = local_idx[segment.local_start + k];
                auto kmer = local_kmers[static_cast<std::size_t>(local_kmer_i)];
                kmer.record_idx += static_cast<std::uint32_t>(offset);
                kmers[segment.out_start + k] = kmer;
            }
        }
    });
    return kmers;
}

static NoInitArray<std::uint64_t> merge_idx(
    const std::vector<IdxSegment>& idx_segments,
    const std::vector<std::uint64_t>& kmer_offsets,
    const std::vector<ThreadGraph>& graphs,
    std::size_t total_kmers,
    ThreadPool& pool
) {
    NoInitArray<std::uint64_t> idx(total_kmers);

    pool.parallel_for(idx_segments.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t s = start; s < end; ++s) {
            const auto& segment = idx_segments[s];
            const auto offset = kmer_offsets[segment.thread_id];
            const auto& local_idx = graphs[segment.thread_id].idx;

            for (std::size_t k = 0; k < segment.length; ++k) {
                idx[segment.out_start + k] = local_idx[segment.local_start + k] + offset;
            }
        }
    });
    return idx;
}

static void merge_edges(NoInitArray<Edge>& edges, ThreadPool& pool)
{
    const std::size_t n_edges = edges.size();
    if (n_edges == 0) {
        edges.reset();
        return;
    }

    NoInitArray<Edge> buf(edges.size());
    auto* src = edges.data();
    auto* dst = buf.data();
    std::vector<std::uint64_t> counts(pool.size() * 65536);

    for (auto key : {&Edge::second, &Edge::first}) {
        for (std::size_t shift = 0; shift < 64; shift += 16) {
            std::fill(counts.begin(), counts.end(), 0);

            pool.parallel_for(n_edges, [&](std::size_t start, std::size_t end, std::size_t t) {
                auto* local_counts = counts.data() + t * 65536;
                for (std::size_t i = start; i < end; ++i) {
                    const auto bucket = ((src[i].*key) >> shift) & 0xFFFFULL;
                    ++local_counts[static_cast<std::size_t>(bucket)];
                }
            });

            std::uint64_t current = 0;
            for (std::size_t bucket = 0; bucket < 65536; ++bucket) {
                for (std::size_t t = 0; t < pool.size(); ++t) {
                    auto& value = counts[t * 65536 + bucket];
                    const auto c = value;
                    value = current;
                    current += c;
                }
            }

            pool.parallel_for(n_edges, [&](std::size_t start, std::size_t end, std::size_t t) {
                auto* local_offsets = counts.data() + t * 65536;
                for (std::size_t i = start; i < end; ++i) {
                    const auto bucket = ((src[i].*key) >> shift) & 0xFFFFULL;
                    const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                    dst[pos] = src[i];
                }
            });

            std::swap(src, dst);
        }
    }

    // Determine final edge count
    std::size_t unique_count = 0;
    std::size_t i = 0;
    while (i < n_edges) {
        const auto first = src[i].first;
        const auto second = src[i].second;
        while (i < n_edges && src[i].first == first && src[i].second == second) {
            ++i;
        }
        ++unique_count;
    }
    buf.reset();
    buf = NoInitArray<Edge>(unique_count);

    std::size_t write_i = 0;
    i = 0;
    while (i < n_edges) {
        const auto first = src[i].first;
        const auto second = src[i].second;
        std::uint64_t weight = 0;

        while (i < n_edges && src[i].first == first && src[i].second == second) {
            weight += src[i].weight;
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

Graph merge_thread_graphs(
    std::vector<ThreadGraph>& graphs,
    std::size_t n_assemblies,
    ThreadPool& pool
) {
    if (graphs.size() == 1) {
        auto& graph = graphs[0];
        if (graph.record_offsets.back() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Total number of FASTA records exceeds uint32 range");
        }

        merge_edges(graph.edges, pool);
        auto idx_segments = merge_nodes(graph.nodes, pool);

        std::vector<std::uint64_t> thread_record_offsets{0};
        auto kmers = merge_kmers(
            idx_segments,
            graphs,
            graph.n_kmers,
            thread_record_offsets,
            pool
        );
        std::vector<Kmer>().swap(graph.kmers);

        std::vector<std::uint64_t> kmer_offsets{0};
        auto idx = merge_idx(
            idx_segments,
            kmer_offsets,
            graphs,
            graph.n_kmers,
            pool
        );
        graph.idx.reset();

        return {
            std::move(kmers),
            std::move(idx),
            std::move(graph.nodes),
            std::move(graph.edges),
            std::move(graph.record_offsets),
            std::move(graph.ids_by_assembly)
        };
    }

    log_python(" - Merging from " + std::to_string(graphs.size()) + " threads...");

    // Merge record offsets
    std::vector<std::uint64_t> thread_record_offsets(graphs.size());
    std::vector<std::uint64_t> record_offsets;
    record_offsets.reserve(n_assemblies + 1);
    record_offsets.push_back(0);

    std::uint64_t total_records = 0;
    for (std::size_t t = 0; t < graphs.size(); ++t) {
        auto& local_offsets = graphs[t].record_offsets;

        const auto base = total_records;
        thread_record_offsets[t] = base;
        total_records += local_offsets.back();

        for (std::size_t i = 1; i < local_offsets.size(); ++i) {
            record_offsets.push_back(base + local_offsets[i]);
        }
        std::vector<std::uint64_t>().swap(local_offsets);
    }
    if (total_records > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Total number of FASTA records exceeds uint32 range");
    }

    // Merge edges and nodes first to reduce peak memory
    auto edges = concat_edges(graphs, pool);
    merge_edges(edges, pool);

    auto nodes = concat_nodes(graphs, pool);
    auto idx_segments = merge_nodes(nodes, pool);

    std::vector<std::uint64_t> kmer_offsets(graphs.size());
    std::uint64_t total_kmers = 0;
    for (std::size_t t = 0; t < graphs.size(); ++t) {
        kmer_offsets[t] = total_kmers;
        total_kmers += graphs[t].n_kmers;
    }

    auto kmers = merge_kmers(idx_segments, graphs, total_kmers, thread_record_offsets, pool);
    for (auto& graph : graphs) {
        std::vector<Kmer>().swap(graph.kmers);
    }

    auto idx = merge_idx(idx_segments, kmer_offsets, graphs, total_kmers, pool);
    for (auto& graph : graphs) {
        graph.idx.reset();
    }

    std::vector<std::vector<std::string>> ids_by_assembly(n_assemblies);
    for (auto& graph : graphs) {
        for (std::size_t i = 0; i < graph.ids_by_assembly.size(); ++i) {
            ids_by_assembly[graph.start_assembly + i] = std::move(graph.ids_by_assembly[i]);
        }
    }

    std::vector<ThreadGraph>().swap(graphs);

    return {
        std::move(kmers),
        std::move(idx),
        std::move(nodes),
        std::move(edges),
        std::move(record_offsets),
        std::move(ids_by_assembly)
    };
}

Graph filter_kmers(
    const Kmer* kmers,
    const std::uint64_t* idx,
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
        n_kmers += static_cast<std::size_t>(nodes[node_i].stop - nodes[node_i].start);
        ++node_i;
        ++used_i;
    }

    graph.nodes = NoInitArray<Node>(used_node_indices.size());
    graph.kmers = NoInitArray<Kmer>(n_kmers);
    graph.idx = NoInitArray<std::uint64_t>(n_kmers);

    std::uint64_t new_start = 0;
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

        for (std::uint64_t k = 0; k < size; ++k) {
            const auto out_i = static_cast<std::size_t>(new_start + k);
            const auto in_i = static_cast<std::size_t>(old_start + k);
            graph.kmers[out_i] = kmers[in_i];
            graph.idx[out_i] = idx[in_i];
        }

        new_start += size;
    }

    return graph;
}

} // namespace seqwin

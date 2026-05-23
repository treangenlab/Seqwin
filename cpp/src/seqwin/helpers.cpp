#include "seqwin/helpers.hpp"
#include "seqwin/thread_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace seqwin {
namespace {

struct IdxSegment {
    std::size_t thread_id;
    std::size_t local_start;
    std::size_t out_start;
    std::size_t length;
};

template <typename T, typename MemberPtr>
std::vector<T> concat(std::vector<ThreadResult>& results, MemberPtr member, ThreadPool& pool)
{
    std::vector<T> out;
    if (results.empty()) {
        return out;
    }

    std::vector<std::size_t> offsets(results.size(), 0);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        offsets[i] = cursor;
        cursor += (results[i].*member).size();
    }
    out.resize(cursor);

    pool.parallel_for(results.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t i = start; i < end; ++i) {
            auto& local = results[i].*member;
            const auto local_size = local.size();
            if (local_size != 0) {
                std::copy(
                    local.begin(),
                    local.end(),
                    out.begin() + static_cast<std::ptrdiff_t>(offsets[i])
                );
            }
            std::vector<T>().swap(local);
        }
    });

    return out;
}

std::vector<Node> concat_nodes(std::vector<ThreadResult>& results, ThreadPool& pool)
{
    return concat<Node>(results, &ThreadResult::nodes, pool);
}

std::vector<std::uint64_t> concat_edges(std::vector<ThreadResult>& results, ThreadPool& pool)
{
    return concat<std::uint64_t>(results, &ThreadResult::edges, pool);
}

// 1. Parallel LSD radix sort based on hash (stable)
// 2. Merge nodes with the same hash
// 3. Build segment metadata for final materialization
static std::pair<std::vector<Node>, std::vector<IdxSegment>> merge_nodes(
    std::vector<Node>& nodes,
    ThreadPool& pool
) {
    const std::size_t n_nodes = nodes.size();
    if (n_nodes == 0) {
        return {{}, {}};
    }

    std::vector<Node> buf(nodes.size());
    auto* src = &nodes;
    auto* dst = &buf;
    std::vector<std::uint64_t> counts(pool.size() * 65536);

    for (std::size_t shift = 0; shift < 64; shift += 16) {
        std::fill(counts.begin(), counts.end(), 0);

        pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t t) {
            auto* local_counts = counts.data() + t * 65536;
            for (std::size_t i = start; i < end; ++i) {
                const auto bucket = ((*src)[i].hash >> shift) & 0xFFFFULL;
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
                const auto bucket = ((*src)[i].hash >> shift) & 0xFFFFULL;
                const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                (*dst)[pos] = (*src)[i];
            }
        });

        std::swap(src, dst);
    }

    // Aggregate nodes and keep idx ranges
    std::vector<IdxSegment> idx_segments;
    idx_segments.reserve(n_nodes);

    std::uint64_t n_kmers = 0;
    std::size_t write_i = 0;
    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = (*src)[i].hash;
        std::uint32_t n_tar = 0;
        std::uint32_t n_neg = 0;
        const auto start = n_kmers;

        while (i < n_nodes && (*src)[i].hash == hash) {
            n_tar += (*src)[i].n_tar;
            n_neg += (*src)[i].n_neg;

            const auto local_start = (*src)[i].start;
            const auto local_stop = (*src)[i].stop;
            const auto length = local_stop - local_start;

            if (length != 0) {
                idx_segments.push_back(IdxSegment{
                    static_cast<std::size_t>((*src)[i].penalty), // thread_id
                    static_cast<std::size_t>(local_start),
                    static_cast<std::size_t>(n_kmers),
                    static_cast<std::size_t>(length)
                });
                n_kmers += length;
            }
            ++i;
        }

        const auto stop = n_kmers;
        nodes[write_i++] = Node{hash, start, stop, n_tar, n_neg, 0.0};
    }
    nodes.resize(write_i);
    return {std::move(nodes), std::move(idx_segments)};
}

static std::vector<Kmer> merge_kmers(
    const std::vector<IdxSegment>& idx_segments,
    std::vector<ThreadResult>& results,
    std::size_t total_kmers,
    ThreadPool& pool
) {
    std::vector<Kmer> kmers(total_kmers);

    pool.parallel_for(idx_segments.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t s = start; s < end; ++s) {
            const auto& segment = idx_segments[s];
            const auto& local_idx = results[segment.thread_id].idx;
            const auto& local_kmers = results[segment.thread_id].kmers;

            for (std::size_t k = 0; k < segment.length; ++k) {
                const auto local_kmer_i = local_idx[segment.local_start + k];
                kmers[segment.out_start + k] = local_kmers[static_cast<std::size_t>(local_kmer_i)];
            }
        }
    });
    return kmers;
}

static std::vector<std::uint64_t> merge_idx(
    const std::vector<IdxSegment>& idx_segments,
    const std::vector<std::uint64_t>& kmer_offsets,
    const std::vector<ThreadResult>& results,
    std::size_t total_kmers,
    ThreadPool& pool
) {
    std::vector<std::uint64_t> idx(total_kmers);

    pool.parallel_for(idx_segments.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t s = start; s < end; ++s) {
            const auto& segment = idx_segments[s];
            const auto offset = kmer_offsets[segment.thread_id];
            const auto& local_idx = results[segment.thread_id].idx;

            for (std::size_t k = 0; k < segment.length; ++k) {
                idx[segment.out_start + k] = local_idx[segment.local_start + k] + offset;
            }
        }
    });
    return idx;
}

static void merge_weighted_edges(std::vector<std::uint64_t>& edges, ThreadPool& pool)
{
    const std::size_t n_edges = edges.size() / 3;
    if (n_edges == 0) {
        edges.clear();
        return;
    }

    std::vector<std::uint64_t> buf(edges.size());
    auto* src = &edges;
    auto* dst = &buf;
    std::vector<std::uint64_t> counts(pool.size() * 65536);

    for (std::size_t column : {std::size_t{1}, std::size_t{0}}) {
        for (std::size_t shift = 0; shift < 64; shift += 16) {
            std::fill(counts.begin(), counts.end(), 0);

            pool.parallel_for(n_edges, [&](std::size_t start, std::size_t end, std::size_t t) {
                auto* local_counts = counts.data() + t * 65536;
                for (std::size_t i = start; i < end; ++i) {
                    const auto bucket = ((*src)[3 * i + column] >> shift) & 0xFFFFULL;
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
                    const auto bucket = ((*src)[3 * i + column] >> shift) & 0xFFFFULL;
                    const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                    (*dst)[3 * pos + 0] = (*src)[3 * i + 0];
                    (*dst)[3 * pos + 1] = (*src)[3 * i + 1];
                    (*dst)[3 * pos + 2] = (*src)[3 * i + 2];
                }
            });

            std::swap(src, dst);
        }
    }

    std::size_t write_i = 0;
    std::uint64_t u = (*src)[0];
    std::uint64_t v = (*src)[1];
    std::uint64_t w = (*src)[2];

    for (std::size_t i = 1; i < n_edges; ++i) {
        const auto next_u = (*src)[3 * i + 0];
        const auto next_v = (*src)[3 * i + 1];
        const auto next_w = (*src)[3 * i + 2];
        if (next_u == u && next_v == v) {
            w += next_w;
            continue;
        }

        edges[3 * write_i + 0] = u;
        edges[3 * write_i + 1] = v;
        edges[3 * write_i + 2] = w;
        ++write_i;

        u = next_u;
        v = next_v;
        w = next_w;
    }

    edges[3 * write_i + 0] = u;
    edges[3 * write_i + 1] = v;
    edges[3 * write_i + 2] = w;
    ++write_i;
    edges.resize(write_i * 3);
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

BuildResult merge_thread_results(
    std::vector<ThreadResult>& results,
    std::size_t n_assemblies,
    ThreadPool& pool
) {
    if (results.size() == 1) {
        auto& result = results[0];

        auto [nodes, idx_segments] = merge_nodes(result.nodes, pool);
        std::vector<Node>().swap(result.nodes);

        auto kmers = merge_kmers(
            idx_segments,
            results,
            result.n_kmers,
            pool
        );
        std::vector<Kmer>().swap(result.kmers);

        std::vector<std::uint64_t> kmer_offsets{0};
        auto idx = merge_idx(
            idx_segments,
            kmer_offsets,
            results,
            result.n_kmers,
            pool
        );
        std::vector<std::uint64_t>().swap(result.idx);

        return {
            std::move(kmers),
            std::move(idx),
            std::move(nodes),
            std::move(result.edges),
            std::move(result.ids_by_assembly)
        };
    }

    log_python(" - Merging from " + std::to_string(results.size()) + " threads...");

    // Merge edges and nodes first to reduce peak memory
    auto edges = concat_edges(results, pool);
    merge_weighted_edges(edges, pool);

    auto nodes_raw = concat_nodes(results, pool);
    auto [nodes, idx_segments] = merge_nodes(nodes_raw, pool);
    std::vector<Node>().swap(nodes_raw);

    std::vector<std::uint64_t> kmer_offsets(results.size());
    std::uint64_t total_kmers = 0;
    for (std::size_t r = 0; r < results.size(); ++r) {
        kmer_offsets[r] = total_kmers;
        total_kmers += results[r].n_kmers;
    }

    auto kmers = merge_kmers(idx_segments, results, total_kmers, pool);
    for (auto& result : results) {
        std::vector<Kmer>().swap(result.kmers);
    }

    auto idx = merge_idx(idx_segments, kmer_offsets, results, total_kmers, pool);
    for (auto& result : results) {
        std::vector<std::uint64_t>().swap(result.idx);
    }

    std::vector<std::vector<std::string>> ids_by_assembly(n_assemblies);
    for (auto& result : results) {
        for (std::size_t i = 0; i < result.ids_by_assembly.size(); ++i) {
            ids_by_assembly[result.start_assembly + i] = std::move(result.ids_by_assembly[i]);
        }
    }

    std::vector<ThreadResult>().swap(results);

    return {
        std::move(kmers),
        std::move(idx),
        std::move(nodes),
        std::move(edges),
        std::move(ids_by_assembly)
    };
}

FilterResult filter_kmers(
    const Kmer* kmers,
    const std::uint64_t* idx,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
) {
    std::sort(used_hashes.begin(), used_hashes.end());

    FilterResult result;
    result.nodes.reserve(used_hashes.size());

    std::size_t node_i = 0;
    std::size_t used_i = 0;
    std::uint64_t new_start = 0;

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

        const Node& old_node = nodes[node_i];
        const auto old_start = old_node.start;
        const auto old_stop = old_node.stop;
        const auto size = old_stop - old_start;

        Node new_node = old_node;
        new_node.start = new_start;
        new_node.stop = new_start + size;
        result.nodes.push_back(new_node);

        result.kmers.insert(result.kmers.end(), kmers + old_start, kmers + old_stop);
        result.idx.insert(result.idx.end(), idx + old_start, idx + old_stop);

        new_start += size;
        ++node_i;
        ++used_i;
    }

    return result;
}

} // namespace seqwin

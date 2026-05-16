#include "seqwin/helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace seqwin {
namespace {

template <typename T, typename MemberPtr>
std::vector<T> concat(std::vector<ThreadResult>& results, MemberPtr member)
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

    std::vector<std::thread> threads;
    threads.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&, i]() {
            auto& local = results[i].*member;
            const auto local_size = local.size();
            if (local_size != 0) {
                std::memcpy(
                    out.data() + offsets[i],
                    local.data(),
                    local_size * sizeof(T));
            }
            std::vector<T>().swap(local);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    return out;
}

std::vector<std::uint8_t> concat_kmers(std::vector<ThreadResult>& results)
{
    return concat<std::uint8_t>(results, &ThreadResult::kmers);
}

std::vector<std::uint64_t> concat_nodes(std::vector<ThreadResult>& results)
{
    return concat<std::uint64_t>(results, &ThreadResult::nodes);
}

std::vector<std::uint64_t> concat_edges(std::vector<ThreadResult>& results)
{
    return concat<std::uint64_t>(results, &ThreadResult::edges);
}

constexpr std::size_t serialized_node_size = 36;
constexpr std::size_t serialized_kmer_size = 17;

static void reorder_kmers_by_idx(
    std::vector<std::uint8_t>& kmers,
    const std::vector<std::uint64_t>& idx,
    std::size_t n_workers
) {
    const std::size_t n = idx.size();
    if (n == 0) {
        return;
    }

    std::vector<std::uint64_t> buf(n);
    const std::size_t workers = std::max<std::size_t>(1, n_workers);
    const std::size_t chunk_size = (n + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(workers);

    // hash: uint64 at offset 0
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                std::uint64_t value = 0;
                std::memcpy(&value, kmers.data() + serialized_kmer_size * idx[i] + 0, sizeof(value));
                buf[i] = value;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    threads.clear();
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                const auto value = buf[i];
                std::memcpy(kmers.data() + serialized_kmer_size * i + 0, &value, sizeof(value));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // pos + record_idx + assembly_idx packed into uint64 from offsets 8..15
    threads.clear();
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                std::uint64_t packed = 0;
                std::memcpy(&packed, kmers.data() + serialized_kmer_size * idx[i] + 8, sizeof(packed));
                buf[i] = packed;
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    threads.clear();
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                const auto packed = buf[i];
                std::memcpy(kmers.data() + serialized_kmer_size * i + 8, &packed, sizeof(packed));
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // is_target: uint8 at offset 16, reuse first n bytes of buf storage
    auto* byte_buf = reinterpret_cast<std::uint8_t*>(buf.data());
    threads.clear();
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                byte_buf[i] = kmers[serialized_kmer_size * idx[i] + 16];
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    threads.clear();
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t start = t * chunk_size;
        if (start >= n) {
            break;
        }
        const std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
            for (std::size_t i = start; i < end; ++i) {
                kmers[serialized_kmer_size * i + 16] = byte_buf[i];
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

static void append_full_node(
    std::vector<std::uint8_t>& out,
    std::uint64_t hash,
    std::uint64_t n_tar,
    std::uint64_t n_neg,
    std::uint64_t start,
    std::uint64_t stop
) {
    const auto old_size = out.size();
    out.resize(old_size + serialized_node_size);
    auto* dst = out.data() + old_size;

    constexpr double penalty = 0.0;

    const auto n_tar16 = static_cast<std::uint16_t>(n_tar);
    const auto n_neg16 = static_cast<std::uint16_t>(n_neg);

    std::memcpy(dst + 0, &hash, sizeof(hash));
    std::memcpy(dst + 8, &n_tar16, sizeof(n_tar16));
    std::memcpy(dst + 10, &n_neg16, sizeof(n_neg16));
    std::memcpy(dst + 12, &penalty, sizeof(penalty));
    std::memcpy(dst + 20, &start, sizeof(start));
    std::memcpy(dst + 28, &stop, sizeof(stop));
}

// 1. Parallel LSD radix sort based on hash (stable)
// 2. Merge nodes with the same hash
// 3. Build the idx array
// 4. Return serialized nodes
static std::vector<std::uint8_t> merge_nodes(
    std::vector<std::uint64_t>& nodes,
    std::vector<ThreadResult>& results,
    const std::vector<std::uint64_t>& kmer_offsets,
    std::vector<std::uint64_t>& idx,
    std::size_t n_workers
) {
    constexpr std::size_t node_width = 6;
    const std::size_t n_nodes = nodes.size() / node_width;
    if (n_nodes == 0) {
        return {};
    }

    std::vector<std::uint64_t> buf(nodes.size());
    auto* src = &nodes;
    auto* dst = &buf;
    std::vector<std::uint64_t> counts(n_workers * 65536);
    const std::size_t chunk_size = (n_nodes + n_workers - 1) / n_workers;

    for (std::size_t shift = 0; shift < 64; shift += 16) {
        std::fill(counts.begin(), counts.end(), 0);

        std::vector<std::thread> threads;
        threads.reserve(n_workers);

        for (std::size_t t = 0; t < n_workers; ++t) {
            const std::size_t start = t * chunk_size;
            const std::size_t end = std::min(start + chunk_size, n_nodes);
            threads.emplace_back([&, t, start, end, shift]() {
                auto* local_counts = counts.data() + t * 65536;
                for (std::size_t i = start; i < end; ++i) {
                    const auto bucket = ((*src)[node_width * i + 0] >> shift) & 0xFFFFULL;
                    ++local_counts[static_cast<std::size_t>(bucket)];
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        std::uint64_t current = 0;
        for (std::size_t bucket = 0; bucket < 65536; ++bucket) {
            for (std::size_t t = 0; t < n_workers; ++t) {
                auto& value = counts[t * 65536 + bucket];
                const auto c = value;
                value = current;
                current += c;
            }
        }

        threads.clear();
        for (std::size_t t = 0; t < n_workers; ++t) {
            const std::size_t start = t * chunk_size;
            const std::size_t end = std::min(start + chunk_size, n_nodes);
            threads.emplace_back([&, t, start, end, shift]() {
                auto* local_offsets = counts.data() + t * 65536;
                for (std::size_t i = start; i < end; ++i) {
                    const auto bucket = ((*src)[node_width * i + 0] >> shift) & 0xFFFFULL;
                    const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                    for (std::size_t col = 0; col < node_width; ++col) {
                        (*dst)[node_width * pos + col] = (*src)[node_width * i + col];
                    }
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        std::swap(src, dst);
    }

    std::vector<std::uint8_t> out;
    out.reserve(n_nodes * serialized_node_size);

    std::size_t i = 0;
    while (i < n_nodes) {
        const auto hash = (*src)[node_width * i + 0];
        std::uint64_t n_tar = 0;
        std::uint64_t n_neg = 0;
        const auto start = static_cast<std::uint64_t>(idx.size());

        while (i < n_nodes && (*src)[node_width * i + 0] == hash) {
            n_tar += (*src)[node_width * i + 1];
            n_neg += (*src)[node_width * i + 2];

            const auto thread_id = static_cast<std::size_t>((*src)[node_width * i + 3]);
            const auto local_start = static_cast<std::size_t>((*src)[node_width * i + 4]);
            const auto local_stop = static_cast<std::size_t>((*src)[node_width * i + 5]);
            const auto offset = kmer_offsets[thread_id];
            const auto& local_idx = results[thread_id].idx;

            for (std::size_t j = local_start; j < local_stop; ++j) {
                idx.push_back(local_idx[j] + offset);
            }
            ++i;
        }

        const auto stop = static_cast<std::uint64_t>(idx.size());
        append_full_node(out, hash, n_tar, n_neg, start, stop);
    }

    for (auto& result : results) {
        std::vector<std::uint64_t>().swap(result.idx);
    }

    return out;
}

static void merge_weighted_edges(std::vector<std::uint64_t>& edges, std::size_t n_workers)
{
    const std::size_t n_edges = edges.size() / 3;
    if (n_edges == 0) {
        edges.clear();
        return;
    }

    const std::size_t chunk_size = (n_edges + n_workers - 1) / n_workers;

    std::vector<std::uint64_t> buf(edges.size());
    auto* src = &edges;
    auto* dst = &buf;
    std::vector<std::uint64_t> counts(n_workers * 65536);

    for (std::size_t column : {std::size_t{1}, std::size_t{0}}) {
        for (std::size_t shift = 0; shift < 64; shift += 16) {
            std::fill(counts.begin(), counts.end(), 0);

            std::vector<std::thread> threads;
            threads.reserve(n_workers);

            for (std::size_t t = 0; t < n_workers; ++t) {
                const std::size_t start = t * chunk_size;
                const std::size_t end = std::min(start + chunk_size, n_edges);
                threads.emplace_back([&, t, start, end, shift, column]() {
                    auto* local_counts = counts.data() + t * 65536;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto bucket = ((*src)[3 * i + column] >> shift) & 0xFFFFULL;
                        ++local_counts[static_cast<std::size_t>(bucket)];
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }

            std::uint64_t current = 0;
            for (std::size_t bucket = 0; bucket < 65536; ++bucket) {
                for (std::size_t t = 0; t < n_workers; ++t) {
                    auto& value = counts[t * 65536 + bucket];
                    const auto c = value;
                    value = current;
                    current += c;
                }
            }

            threads.clear();
            for (std::size_t t = 0; t < n_workers; ++t) {
                const std::size_t start = t * chunk_size;
                const std::size_t end = std::min(start + chunk_size, n_edges);
                threads.emplace_back([&, t, start, end, shift, column]() {
                    auto* local_offsets = counts.data() + t * 65536;
                    for (std::size_t i = start; i < end; ++i) {
                        const auto bucket = ((*src)[3 * i + column] >> shift) & 0xFFFFULL;
                        const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                        (*dst)[3 * pos + 0] = (*src)[3 * i + 0];
                        (*dst)[3 * pos + 1] = (*src)[3 * i + 1];
                        (*dst)[3 * pos + 2] = (*src)[3 * i + 2];
                    }
                });
            }
            for (auto& thread : threads) {
                thread.join();
            }

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
    std::size_t n_workers
) {
    if (results.size() == 1) {
        auto& result = results[0];

        std::vector<std::uint64_t> idx;
        idx.reserve(static_cast<std::size_t>(result.n_kmers));

        std::vector<std::uint64_t> kmer_offsets{0};
        auto nodes = merge_nodes(result.nodes, results, kmer_offsets, idx, n_workers);
        reorder_kmers_by_idx(result.kmers, idx, n_workers);
        return {
            std::move(result.kmers),
            std::move(idx),
            std::move(nodes),
            std::move(result.edges),
            std::move(result.ids_by_assembly)
        };
    }

    log_python(" - Merging from " + std::to_string(n_workers) + " threads...");

    std::vector<std::uint64_t> kmer_offsets(results.size());
    std::uint64_t total_kmers = 0;
    for (std::size_t r = 0; r < results.size(); ++r) {
        kmer_offsets[r] = total_kmers;
        total_kmers += results[r].n_kmers;
    }

    auto kmers = concat_kmers(results);
    std::vector<std::uint64_t> idx;
    idx.reserve(static_cast<std::size_t>(total_kmers));

    auto nodes_raw = concat_nodes(results);
    auto nodes = merge_nodes(nodes_raw, results, kmer_offsets, idx, n_workers);
    reorder_kmers_by_idx(kmers, idx, n_workers);

    auto edges = concat_edges(results);
    merge_weighted_edges(edges, n_workers);

    std::vector<std::vector<std::string>> ids_by_assembly(n_assemblies);
    for (auto& result : results) {
        for (std::size_t i = 0; i < result.ids_by_assembly.size(); ++i) {
            ids_by_assembly[result.start_assembly + i] = std::move(result.ids_by_assembly[i]);
        }
    }

    return {
        std::move(kmers),
        std::move(idx),
        std::move(nodes),
        std::move(edges),
        std::move(ids_by_assembly)
    };
}

} // namespace seqwin

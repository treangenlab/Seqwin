#include "seqwin/helpers.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

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
static std::vector<std::uint8_t> merge_nodes(
    std::vector<std::uint64_t>& compact_nodes,
    std::vector<std::vector<std::uint64_t>>& all_idx_global,
    std::vector<std::uint64_t>& final_idx,
    std::size_t n_workers
) {
    const std::size_t n_nodes = compact_nodes.size() / 4;
    if (n_nodes == 0) {
        return {};
    }

    std::vector<std::uint64_t> buf(compact_nodes.size());
    auto* src = &compact_nodes;
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
                    const auto bucket = ((*src)[4 * i + 0] >> shift) & 0xFFFFULL;
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
                    const auto bucket = ((*src)[4 * i + 0] >> shift) & 0xFFFFULL;
                    const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
                    (*dst)[4 * pos + 0] = (*src)[4 * i + 0];
                    (*dst)[4 * pos + 1] = (*src)[4 * i + 1];
                    (*dst)[4 * pos + 2] = (*src)[4 * i + 2];
                    (*dst)[4 * pos + 3] = (*src)[4 * i + 3];
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
        const auto hash = (*src)[4 * i + 0];
        std::uint64_t n_tar = 0;
        std::uint64_t n_neg = 0;
        const auto start = static_cast<std::uint64_t>(final_idx.size());

        while (i < n_nodes && (*src)[4 * i + 0] == hash) {
            n_tar += (*src)[4 * i + 1];
            n_neg += (*src)[4 * i + 2];

            const auto idx_ptr = static_cast<std::size_t>((*src)[4 * i + 3]);
            auto& src_idx = all_idx_global[idx_ptr];
            final_idx.insert(final_idx.end(), src_idx.begin(), src_idx.end());
            std::vector<std::uint64_t>().swap(src_idx);
            ++i;
        }

        const auto stop = static_cast<std::uint64_t>(final_idx.size());
        append_full_node(out, hash, n_tar, n_neg, start, stop);
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

    std::size_t write_idx = 0;
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

        edges[3 * write_idx + 0] = u;
        edges[3 * write_idx + 1] = v;
        edges[3 * write_idx + 2] = w;
        ++write_idx;

        u = next_u;
        v = next_v;
        w = next_w;
    }

    edges[3 * write_idx + 0] = u;
    edges[3 * write_idx + 1] = v;
    edges[3 * write_idx + 2] = w;
    ++write_idx;
    edges.resize(write_idx * 3);
}

} // namespace

BuildResult merge_thread_results(
    std::vector<ThreadResult>& results,
    std::size_t n_assemblies,
    std::size_t n_workers
) {
    if (results.size() == 1) {
        auto& result = results[0];

        std::vector<std::uint64_t> idx;
        idx.reserve(static_cast<std::size_t>(result.n_kmers));
        auto nodes = merge_nodes(result.nodes, result.all_idx, idx, n_workers);
        return {
            std::move(result.kmers),
            std::move(idx),
            std::move(nodes),
            std::move(result.edges),
            std::move(result.ids_by_assembly)
        };
    }

    // Update local k-mer indices in each thread to global indices
    // Concat all_idx from all threads, and update idx pointers (4th column of result.nodes)
    std::uint64_t kmer_offset = results[0].n_kmers;
    std::uint64_t all_idx_offset = static_cast<std::uint64_t>(results[0].all_idx.size());
    for (std::size_t r = 1; r < results.size(); ++r) {
        auto& result = results[r];

        for (auto& local_idx : result.all_idx) {
            for (auto& idx : local_idx) {
                idx += kmer_offset;
            }
        }
        const std::size_t n_rows = result.nodes.size() / 4;
        for (std::size_t row = 0; row < n_rows; ++row) {
            result.nodes[4 * row + 3] += all_idx_offset;
        }
        kmer_offset += result.n_kmers;
        all_idx_offset += static_cast<std::uint64_t>(result.all_idx.size());
    }

    std::vector<std::vector<std::uint64_t>> all_idx_global;
    all_idx_global.reserve(static_cast<std::size_t>(all_idx_offset));
    for (auto& result : results) {
        for (auto& idx_vec : result.all_idx) {
            all_idx_global.emplace_back(std::move(idx_vec));
        }
        std::vector<std::vector<std::uint64_t>>().swap(result.all_idx);
    }

    auto kmers = concat_kmers(results);
    std::vector<std::uint64_t> idx;
    idx.reserve(static_cast<std::size_t>(kmer_offset));

    auto compact_nodes = concat_nodes(results);
    auto nodes = merge_nodes(compact_nodes, all_idx_global, idx, n_workers);

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

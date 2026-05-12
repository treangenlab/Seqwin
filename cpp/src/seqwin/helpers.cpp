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

std::vector<std::uint64_t> concat_edges(std::vector<ThreadResult>& results)
{
    return concat<std::uint64_t>(results, &ThreadResult::edges);
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
    auto kmers = concat_kmers(results);
    auto edges = concat_edges(results);
    merge_weighted_edges(edges, n_workers);

    std::vector<std::vector<std::string>> ids_by_assembly(n_assemblies);
    for (auto& result : results) {
        for (std::size_t i = 0; i < result.ids_by_assembly.size(); ++i) {
            ids_by_assembly[result.start_assembly + i] = std::move(result.ids_by_assembly[i]);
        }
    }

    return {std::move(kmers), std::move(edges), std::move(ids_by_assembly)};
}

} // namespace seqwin

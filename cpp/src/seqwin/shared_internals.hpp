#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "seqwin/graph.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {

/**
 * @brief Return the half-open k-mer range for a node in `Graph`.
 *
 * Nodes and k-mers share hash order, so the end is the next node's `start`,
 * or `n_kmers` for the final node.
 */
inline std::pair<std::size_t, std::size_t> kmer_range(
    const Node* nodes,
    std::size_t n_nodes,
    std::size_t n_kmers,
    std::size_t node_idx
) {
    if (node_idx >= n_nodes) {
        throw std::invalid_argument("Node index is out of bounds");
    }
    const auto start = nodes[node_idx].start;
    const auto stop = node_idx + 1 < n_nodes
        ? nodes[node_idx + 1].start
        : n_kmers;
    if (start >= stop || stop > n_kmers) {
        throw std::invalid_argument("Node k-mer range is invalid");
    }
    return {start, stop};
}

/**
 * @brief Stable parallel LSD radix sort over an unsigned 64-bit key.
 *
 * Sorts `values` in place according to the key returned by `key`.
 *
 * Requirements:
 * @li `Container` provides mutable `data()` and `size()`.
 * @li Elements are default-constructible and copy-assignable.
 * @li Keys are unsigned 64-bit integral values.
 *
 * @tparam Container Mutable contiguous container type.
 * @tparam KeyFn Key-extractor type accepted by `std::invoke`.
 * @param values Container to sort in place.
 * @param key Extractor returning the unsigned 64-bit key for an element.
 * @param ascending If `true`, sort in ascending key order; otherwise sort in descending key order.
 * @param pool Thread pool.
 */
template <typename Container, typename KeyFn>
void lsd_radix_sort(
    Container& values,
    KeyFn key,
    bool ascending,
    ThreadPool& pool
) {
    using T = std::remove_pointer_t<decltype(values.data())>;
    using KeyType = std::remove_cv_t<std::remove_reference_t<
        std::invoke_result_t<KeyFn&, const T&>
    >>;
    static_assert(std::is_integral_v<KeyType>, "Radix-sort keys must be integral");
    static_assert(std::is_unsigned_v<KeyType>, "Radix-sort keys must be unsigned");
    static_assert(sizeof(KeyType) == 8, "Radix-sort keys must be exactly 64 bits");
    static_assert(std::numeric_limits<KeyType>::digits == 64, "Radix-sort keys must be exactly 64 bits");

    const std::size_t n = values.size();
    if (n == 0) {
        return;
    }

    static constexpr std::size_t bucket_count = 65536;
    static constexpr std::uint64_t bucket_mask = bucket_count - 1;

    NoInitArray<T> buf(n);
    auto* src = values.data();
    auto* dst = buf.data();
    // Zeroed before every radix pass
    NoInitArray<std::size_t> counts(pool.size() * bucket_count);

    for (std::size_t shift = 0; shift < 64; shift += 16) {
        std::fill(counts.begin(), counts.end(), 0);

        pool.parallel_for(n, [&](std::size_t i, std::size_t chunk_id) {
            auto* local_counts = counts.data() + chunk_id * bucket_count;
            const auto bucket = (std::invoke(key, static_cast<const T&>(src[i])) >> shift) & bucket_mask;
            ++local_counts[static_cast<std::size_t>(bucket)];
        });

        std::size_t current = 0;
        for (std::size_t bucket_i = 0; bucket_i < bucket_count; ++bucket_i) {
            const auto bucket = ascending ? bucket_i : bucket_count - bucket_i - 1;
            for (std::size_t chunk_id = 0; chunk_id < pool.size(); ++chunk_id) {
                auto& value = counts[chunk_id * bucket_count + bucket];
                const auto count = value;
                value = current;
                current += count;
            }
        }

        pool.parallel_for(n, [&](std::size_t i, std::size_t chunk_id) {
            auto* local_offsets = counts.data() + chunk_id * bucket_count;
            const auto bucket = (std::invoke(key, static_cast<const T&>(src[i])) >> shift) & bucket_mask;
            const auto pos = local_offsets[static_cast<std::size_t>(bucket)]++;
            dst[pos] = src[i];
        });

        std::swap(src, dst);
    }
}

/**
 * @brief Stable parallel LSD radix sort over an unsigned 64-bit key in ascending order.
 *
 * This is a convenience overload of `lsd_radix_sort()`.
 */
template <typename Container, typename KeyFn>
void lsd_radix_sort(
    Container& values,
    KeyFn key,
    ThreadPool& pool
) {
    lsd_radix_sort(values, key, true, pool);
}

/**
 * @brief Stable parallel LSD radix sort of unsigned 64-bit scalar values in ascending order.
 *
 * This is a convenience overload of `lsd_radix_sort()`.
 */
template <typename Container>
void lsd_radix_sort(
    Container& values,
    ThreadPool& pool
) {
    const auto identity = [](const auto& value) -> const auto& {
        return value;
    };
    lsd_radix_sort(values, identity, true, pool);
}

} // namespace seqwin::internal

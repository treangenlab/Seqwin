#include "seqwin/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "utils/thread_pool.hpp"

namespace seqwin {

void get_penalty(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const std::size_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t n_cpu
) {
    if (n_record_offsets != n_assemblies + 1) {
        throw std::invalid_argument("len(record_offsets) must equal len(is_targets) + 1");
    }
    if (n_record_offsets == 0 || record_offsets[0] != 0) {
        throw std::invalid_argument("record_offsets must start with 0");
    }
    if (n_assemblies > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Number of assemblies exceeds uint32 range");
    }

    std::size_t total_targets = 0;
    std::size_t total_non_targets = 0;
    for (std::size_t i = 0; i < n_assemblies; ++i) {
        if (record_offsets[i + 1] < record_offsets[i]) {
            throw std::invalid_argument("record_offsets must be nondecreasing");
        }
        if (is_targets[i]) {
            ++total_targets;
        } else {
            ++total_non_targets;
        }
    }
    if (total_targets == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (total_non_targets == 0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }

    std::size_t n_workers = std::max<std::size_t>(1, n_cpu);
    if (n_nodes > 0) {
        n_workers = std::min(n_workers, n_nodes);
    }
    internal::ThreadPool pool(n_workers);

    pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t node_i = start; node_i < end; ++node_i) {
            auto& node = nodes[node_i];
            if (node.start == node.stop) {
                node.n_tar = 0;
                node.n_neg = 0;
                node.penalty = 1.0;
                continue;
            }

            std::uint32_t n_tar = 0;
            std::uint32_t n_neg = 0;

            // Monotonic scan of record_idx and record_offsets
            // Each node range has nondecreasing record_idx values, so one upper_bound()
            // maps the first record to its assembly and the scan only advances forward
            std::uint32_t previous_record_idx = kmers[node.start].record_idx;
            if (static_cast<std::size_t>(previous_record_idx) >= record_offsets[n_assemblies]) {
                throw std::invalid_argument("record_idx is outside record_offsets range");
            }
            const auto* offset_it = std::upper_bound(
                record_offsets,
                record_offsets + n_record_offsets,
                static_cast<std::size_t>(previous_record_idx)
            );
            std::size_t assembly_idx = static_cast<std::size_t>(offset_it - record_offsets - 1);
            // Count each assembly once per node
            std::size_t last_counted_assembly = std::numeric_limits<std::size_t>::max();

            for (std::size_t kmer_i = node.start; kmer_i < node.stop; ++kmer_i) {
                const std::uint32_t record_idx_u32 = kmers[kmer_i].record_idx;
                if (record_idx_u32 < previous_record_idx) {
                    throw std::invalid_argument("record_idx must be nondecreasing within each node range");
                }
                previous_record_idx = record_idx_u32;

                const std::size_t record_idx = static_cast<std::size_t>(record_idx_u32);
                if (record_idx >= record_offsets[n_assemblies]) {
                    throw std::invalid_argument("record_idx is outside record_offsets range");
                }
                // Duplicate record offsets are allowed for zero-record assemblies
                while (
                    assembly_idx + 1 < n_assemblies &&
                    record_idx >= record_offsets[assembly_idx + 1]
                ) {
                    ++assembly_idx;
                }
                if (assembly_idx != last_counted_assembly) {
                    if (is_targets[assembly_idx]) {
                        ++n_tar;
                    } else {
                        ++n_neg;
                    }
                    last_counted_assembly = assembly_idx;
                }
            }

            node.n_tar = n_tar;
            node.n_neg = n_neg;
            const double frac_tar = static_cast<double>(n_tar) / static_cast<double>(total_targets);
            const double frac_neg = static_cast<double>(n_neg) / static_cast<double>(total_non_targets);
            node.penalty = std::hypot(1.0 - frac_tar, frac_neg);
        }
    });
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

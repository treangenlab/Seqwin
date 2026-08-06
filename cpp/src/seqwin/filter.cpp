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
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t n_cpu
) {
    /** Metadata shared by all FASTA records in one assembly. */
    struct RecordInfo {
        /** Inclusive global index of the assembly's final FASTA record. */
        std::uint32_t last_record_idx;
        /** Whether the assembly belongs to the target set, as 0 or 1. */
        std::uint32_t is_target;
    };

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

    const std::uint32_t n_records = record_offsets[n_assemblies];
    NoInitArray<RecordInfo> record_info(n_records);
    pool.parallel_for(n_assemblies, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t assembly_idx = start; assembly_idx < end; ++assembly_idx) {
            const std::uint32_t record_start = record_offsets[assembly_idx];
            const std::uint32_t record_stop = record_offsets[assembly_idx + 1];
            if (record_start == record_stop) {
                continue;
            }
            const RecordInfo info{
                record_stop - 1,
                is_targets[assembly_idx] ? 1U : 0U
            };
            std::fill(
                record_info.begin() + record_start,
                record_info.begin() + record_stop,
                info
            );
        }
    });

    const double inv_total_targets = 1.0 / static_cast<double>(total_targets);
    const double inv_total_non_targets = 1.0 / static_cast<double>(total_non_targets);

    pool.parallel_for(n_nodes, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t node_i = start; node_i < end; ++node_i) {
            auto& node = nodes[node_i];
            if (node.start == node.stop) {
                node.n_tar = 0;
                node.n_neg = 0;
                node.penalty = 1.0;
                continue;
            }

            auto previous_record_idx = kmers[node.start].record_idx;
            if (previous_record_idx >= n_records) {
                throw std::invalid_argument("record_idx is outside record_offsets range");
            }
            auto info = record_info[previous_record_idx];
            auto last_record_idx = info.last_record_idx;
            std::uint32_t n_tar = info.is_target;
            std::uint32_t n_neg = 1U - info.is_target;

            for (std::size_t kmer_i = node.start + 1; kmer_i < node.stop; ++kmer_i) {
                const std::uint32_t record_idx = kmers[kmer_i].record_idx;
                if (record_idx < previous_record_idx) {
                    throw std::invalid_argument("record_idx must be nondecreasing within each node range");
                }
                previous_record_idx = record_idx;

                if (record_idx <= last_record_idx) {
                    continue;
                }
                if (record_idx >= n_records) {
                    throw std::invalid_argument("record_idx is outside record_offsets range");
                }
                info = record_info[record_idx];
                last_record_idx = info.last_record_idx;
                n_tar += info.is_target;
                n_neg += 1U - info.is_target;
            }

            node.n_tar = n_tar;
            node.n_neg = n_neg;
            const double frac_tar = static_cast<double>(n_tar) * inv_total_targets;
            const double frac_neg = static_cast<double>(n_neg) * inv_total_non_targets;
            node.penalty = std::sqrt((1.0 - frac_tar) * (1.0 - frac_tar) + frac_neg * frac_neg);
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

#include "seqwin/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "btllib/minimizer.hpp"
#include "seqwin/fasta_reader.hpp"
#include "seqwin/helpers.hpp"
#include "seqwin/thread_pool.hpp"

namespace seqwin {
namespace {

struct NodeState {
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    std::size_t last_seen_assembly;
    std::uint64_t start = 0;
    std::uint64_t count = 0;
    std::uint64_t cursor = 0;
};

using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const noexcept
    {
        return std::hash<std::uint64_t>{}(key.first) ^
               (std::hash<std::uint64_t>{}(key.second) << 1);
    }
};

struct EdgeState {
    std::uint64_t weight;
    std::size_t last_seen_assembly;
};

ThreadResult build_worker(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t start_assembly,
    std::size_t end_assembly,
    std::size_t thread_id
) {
    const auto n_kmers_est = seqwin::est_kmer_number(
        std::vector<std::string>(
            assembly_paths.begin() + static_cast<std::ptrdiff_t>(start_assembly),
            assembly_paths.begin() + static_cast<std::ptrdiff_t>(end_assembly)
        ),
        windowsize
    );

    ThreadResult result;
    result.kmers.reserve(n_kmers_est);
    result.ids_by_assembly.resize(end_assembly - start_assembly);
    result.start_assembly = start_assembly;

    std::vector<std::uint64_t> hashes;
    hashes.reserve(n_kmers_est);

    // Reserving for unordered_map will actually allocate physical memory
    std::unordered_map<std::uint64_t, NodeState> node_map;
    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash> edge_map;

    for (std::size_t assembly_i = start_assembly; assembly_i < end_assembly; ++assembly_i) {
        const auto assembly_idx = assembly_indices[assembly_i];
        if (assembly_idx > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("assembly_idx must fit in uint16");
        }
        const auto assembly_idx16 = static_cast<std::uint16_t>(assembly_idx);
        const bool is_target = is_targets[assembly_i];

        auto records = seqwin::read_fasta(assembly_paths[assembly_i]);
        auto& record_IDs = result.ids_by_assembly[assembly_i - start_assembly];
        record_IDs.resize(records.size());

        for (std::size_t record_idx = 0; record_idx < records.size(); ++record_idx) {
            if (record_idx > std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("record_idx must fit in uint16");
            }
            const auto record_idx16 = static_cast<std::uint16_t>(record_idx);

            auto& record = records[record_idx];
            record_IDs[record_idx] = std::move(record.id);

            const auto mins = btllib::minimize_sequence(record.sequence, kmerlen, windowsize);

            for (const auto& m : mins) {
                if (m.pos > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("minimizer position exceeds uint32 range");
                }
                result.kmers.push_back(Kmer{
                    static_cast<std::uint32_t>(m.pos),
                    record_idx16,
                    assembly_idx16
                });
                hashes.push_back(m.out_hash);

                auto [node_it, node_inserted] = node_map.try_emplace(
                    m.out_hash,
                    NodeState{
                        is_target ? std::uint32_t{1} : std::uint32_t{0},
                        is_target ? std::uint32_t{0} : std::uint32_t{1},
                        assembly_i, 0, 0, 0
                    }
                );
                if (!node_inserted && node_it->second.last_seen_assembly != assembly_i) {
                    if (is_target) {
                        ++(node_it->second.n_tar);
                    } else {
                        ++(node_it->second.n_neg);
                    }
                    node_it->second.last_seen_assembly = assembly_i;
                }
                ++(node_it->second.count);
                ++result.n_kmers;
            }

            if (mins.size() < 2) {
                continue;
            }

            for (std::size_t i = 0; i + 1 < mins.size(); ++i) {
                auto u = mins[i].out_hash;
                auto v = mins[i + 1].out_hash;
                if (v < u) {
                    std::swap(u, v);
                }
                const EdgeKey key{u, v};
                auto [edge_it, edge_inserted] = edge_map.try_emplace(
                    key,
                    EdgeState{1, assembly_i}
                );
                if (!edge_inserted && edge_it->second.last_seen_assembly != assembly_i) {
                    ++(edge_it->second.weight);
                    edge_it->second.last_seen_assembly = assembly_i;
                }
            }
        }
    }

    result.idx.resize(result.n_kmers);
    std::uint64_t cursor = 0;
    for (auto& [hash, state] : node_map) {
        (void)hash;
        state.start = cursor;
        state.cursor = cursor;
        cursor += state.count;
    }
    for (std::size_t i = 0; i < result.n_kmers; ++i) {
        const auto hash = hashes[i];
        auto node_it = node_map.find(hash);
        result.idx[node_it->second.cursor++] = static_cast<std::uint64_t>(i);
    }

    result.nodes.resize(node_map.size());
    std::size_t node_i = 0;
    for (const auto& [hash, state] : node_map) {
        const auto stop = state.start + state.count;
        result.nodes[node_i++] = ThreadNode{
            hash,
            state.n_tar,
            state.n_neg,
            static_cast<std::uint64_t>(thread_id),
            state.start,
            stop
        };
    }

    result.edges.resize(edge_map.size() * 3);
    std::size_t edge_i = 0;
    for (const auto& [key, state] : edge_map) {
        result.edges[edge_i++] = key.first;
        result.edges[edge_i++] = key.second;
        result.edges[edge_i++] = state.weight;
    }

    return result;
}

} // namespace

BuildResult build_impl(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu
) {
    if (assembly_paths.size() != assembly_indices.size() ||
        assembly_paths.size() != is_targets.size()) {
        throw std::runtime_error(
            "assembly_paths, assembly_idx, and is_target must have the same length");
    }
    for (std::size_t i = 0; i < assembly_indices.size(); ++i) {
        if (assembly_indices[i] != i) {
            throw std::runtime_error("assembly_indices must be strictly incremental from 0 to N-1");
        }
    }

    const auto n_assemblies = assembly_paths.size();
    std::size_t n_workers = std::max<std::size_t>(1, n_cpu);
    if (n_assemblies > 0) {
        n_workers = std::min(n_workers, n_assemblies);
    }

    ThreadPool pool(n_workers);
    std::vector<ThreadResult> results(n_workers);

    const std::size_t base = n_assemblies / n_workers;
    const std::size_t rem = n_assemblies % n_workers;

    pool.parallel_for(n_workers, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t i = start; i < end; ++i) {
            std::size_t chunk_start = i * base + std::min(i, rem);
            std::size_t chunk_end = chunk_start + base + (i < rem ? 1 : 0);
            results[i] = build_worker(
                assembly_paths,
                kmerlen,
                windowsize,
                assembly_indices,
                is_targets,
                chunk_start,
                chunk_end,
                i
            );
        }
    });

    return merge_thread_results(results, n_assemblies, pool);
}

} // namespace seqwin

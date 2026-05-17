#include "seqwin/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "btllib/minimizer.hpp"
#include "seqwin/fasta_reader.hpp"
#include "seqwin/helpers.hpp"

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
    ThreadResult result;
    result.start_assembly = start_assembly;
    result.kmers.reserve(
        seqwin::est_kmer_number(
            std::vector<std::string>(
                assembly_paths.begin() + static_cast<std::ptrdiff_t>(start_assembly),
                assembly_paths.begin() + static_cast<std::ptrdiff_t>(end_assembly)
            ),
            windowsize
        )
    );
    result.ids_by_assembly.reserve(end_assembly - start_assembly);
    std::vector<std::uint64_t> hashes;
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
        auto& idx_to_id = result.ids_by_assembly.emplace_back();
        idx_to_id.reserve(records.size());

        for (std::size_t record_idx = 0; record_idx < records.size(); ++record_idx) {
            if (record_idx > std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("record_idx must fit in uint16");
            }
            const auto record_idx16 = static_cast<std::uint16_t>(record_idx);

            auto& record = records[record_idx];
            idx_to_id.push_back(std::move(record.id));

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


    result.idx.resize(static_cast<std::size_t>(result.n_kmers));
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

    result.nodes.reserve(node_map.size());
    for (const auto& [hash, state] : node_map) {
        const auto stop = state.start + state.count;
        result.nodes.push_back(ThreadNode{
            hash,
            state.n_tar,
            state.n_neg,
            static_cast<std::uint64_t>(thread_id),
            state.start,
            stop
        });
    }

    result.edges.reserve(edge_map.size() * 3);
    for (const auto& [key, state] : edge_map) {
        result.edges.push_back(key.first);
        result.edges.push_back(key.second);
        result.edges.push_back(state.weight);
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

    std::vector<ThreadResult> results(n_workers);
    std::vector<std::thread> threads;
    threads.reserve(n_workers);
    std::exception_ptr thread_error = nullptr;
    std::mutex error_mutex;

    const std::size_t base = n_assemblies / n_workers;
    const std::size_t rem = n_assemblies % n_workers;
    std::size_t start = 0;
    for (std::size_t i = 0; i < n_workers; ++i) {
        const auto chunk_size = base + (i < rem ? 1 : 0);
        const auto end = start + chunk_size;
        threads.emplace_back([&, i, start, end]() {
            try {
                results[i] = build_worker(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    assembly_indices,
                    is_targets,
                    start,
                    end,
                    i
                );
            } catch (...) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!thread_error) {
                    thread_error = std::current_exception();
                }
            }
        });
        start = end;
    }

    for (auto& t : threads) {
        t.join();
    }
    if (thread_error) {
        std::rethrow_exception(thread_error);
    }

    return merge_thread_results(results, n_assemblies, n_workers);
}

} // namespace seqwin

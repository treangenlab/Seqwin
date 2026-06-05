#include "seqwin/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "btllib/minimizer.hpp"
#include "seqwin/fasta_reader.hpp"
#include "seqwin/helpers.hpp"
#include "seqwin/thread_pool.hpp"

namespace seqwin {
namespace {

constexpr std::size_t map_reserve_divisor = 100;

struct RawKmer {
    std::uint64_t hash;
    Kmer kmer;
};

struct NodeState {
    std::size_t count = 0;
    std::size_t start = 0;
    std::size_t cursor = 0;
    std::uint32_t n_tar = 0;
    std::uint32_t n_neg = 0;
    std::size_t last_seen_assembly = std::numeric_limits<std::size_t>::max();
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
    std::size_t weight = 0;
    std::size_t last_seen_assembly = std::numeric_limits<std::size_t>::max();
};

ThreadGraph build_worker(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<bool>& is_targets,
    std::size_t start_assembly,
    std::size_t end_assembly,
    std::size_t thread_id,
    bool low_memory
) {
    // Estimate total minimizer count in all assemblies
    const auto n_kmers_est = seqwin::est_kmer_number(
        std::vector<std::string>(
            assembly_paths.begin() + static_cast<std::ptrdiff_t>(start_assembly),
            assembly_paths.begin() + static_cast<std::ptrdiff_t>(end_assembly)
        ),
        windowsize
    );
    std::vector<RawKmer> raw_kmers;
    if (!low_memory) {
        raw_kmers.reserve(n_kmers_est);
    }

    ThreadGraph graph;
    graph.record_offsets.reserve(end_assembly - start_assembly + 1);
    graph.record_offsets.push_back(0);
    graph.ids_by_assembly.reserve(end_assembly - start_assembly);
    graph.start_assembly = start_assembly;
    graph.end_assembly = end_assembly;

    // Reserving for unordered_map will actually allocate physical memory
    std::unordered_map<std::uint64_t, NodeState> node_map;
    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash> edge_map;
    const auto n_map_entries_est = n_kmers_est / map_reserve_divisor;
    node_map.reserve(n_map_entries_est);
    edge_map.reserve(n_map_entries_est);

    for (std::size_t assembly_i = start_assembly; assembly_i < end_assembly; ++assembly_i) {
        const bool is_target = is_targets[assembly_i];

        auto records = seqwin::read_fasta(assembly_paths[assembly_i]);
        std::vector<std::string> record_ids;
        record_ids.reserve(records.size());

        for (std::size_t record_i = 0; record_i < records.size(); ++record_i) {
            const auto record_idx = graph.record_offsets.back() + record_i;
            auto& record = records[record_i];
            if (record.sequence.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "Sequence length exceeds uint32 range for record " +
                    record.id + " in assembly " + assembly_paths[assembly_i]);
            }
            record_ids.push_back(std::move(record.id));

            // Generate minimizers for the current record
            const auto mins = btllib::minimize_sequence(record.sequence, kmerlen, windowsize);

            for (const auto& m : mins) {
                if (!low_memory) {
                    raw_kmers.push_back(RawKmer{
                        m.out_hash,
                        Kmer{
                            static_cast<std::uint32_t>(m.pos),
                            static_cast<std::uint32_t>(record_idx)
                        }
                    });
                }

                // Add this minimizer to an existing node, or create a new node
                auto [node_it, node_inserted] = node_map.try_emplace(m.out_hash);
                ++(node_it->second.count);
                if (node_inserted || node_it->second.last_seen_assembly != assembly_i) {
                    if (is_target) {
                        ++(node_it->second.n_tar);
                    } else {
                        ++(node_it->second.n_neg);
                    }
                    node_it->second.last_seen_assembly = assembly_i;
                }
                ++graph.n_kmers;
            }

            // Current record is too short for an edge
            if (mins.size() < 2) {
                continue;
            }

            // Add undirected edges
            for (std::size_t i = 0; i + 1 < mins.size(); ++i) {
                auto u = mins[i].out_hash;
                auto v = mins[i + 1].out_hash;
                if (v < u) {
                    std::swap(u, v);
                }
                const EdgeKey key{u, v};
                auto edge_it = edge_map.try_emplace(key).first;
                if (edge_it->second.last_seen_assembly != assembly_i) {
                    ++(edge_it->second.weight);
                    edge_it->second.last_seen_assembly = assembly_i;
                }
            }
        }
        graph.record_offsets.push_back(graph.record_offsets.back() + records.size());
        graph.ids_by_assembly.push_back(std::move(record_ids));
    }

    // Materialize edges first to reduce peak memory
    graph.edges = NoInitArray<Edge>(edge_map.size());
    std::size_t edge_i = 0;
    for (const auto& [key, state] : edge_map) {
        graph.edges[edge_i++] = Edge{key.first, key.second, state.weight};
    }
    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash>().swap(edge_map);

    if (!low_memory) {
        // Build ThreadGraph.kmers (grouped by hash)
        graph.kmers = NoInitArray<Kmer>(graph.n_kmers);
        std::size_t cursor = 0;
        for (auto& [hash, state] : node_map) {
            (void)hash;
            state.start = cursor;
            state.cursor = cursor;
            cursor += state.count;
        }
        for (const auto& rk : raw_kmers) {
            auto node_it = node_map.find(rk.hash);
            graph.kmers[node_it->second.cursor++] = rk.kmer;
        }
        std::vector<RawKmer>().swap(raw_kmers);
    }

    graph.n_nodes = node_map.size();
    graph.nodes = NoInitArray<ThreadNode>(node_map.size());
    std::size_t node_i = 0;
    for (const auto& [hash, state] : node_map) {
        graph.nodes[node_i++] = ThreadNode{
            hash,
            state.start,
            state.count,
            state.n_tar,
            state.n_neg,
            thread_id
        };
    }

    return graph;
}

NoInitArray<Kmer> recompute_kmers(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<ThreadGraph>& graphs,
    const std::vector<std::size_t>& record_offsets,
    NodeMaps& node_maps,
    ThreadPool& pool
) {
    std::size_t total_kmers = 0;
    for (const auto& graph : graphs) {
        total_kmers += graph.n_kmers;
    }
    NoInitArray<Kmer> kmers(total_kmers);

    pool.parallel_for(graphs.size(), [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t thread_id = start; thread_id < end; ++thread_id) {
            const auto& graph = graphs[thread_id];
            auto& hash_to_cursor = node_maps[thread_id];

            for (std::size_t assembly_i = graph.start_assembly;
                 assembly_i < graph.end_assembly;
                 ++assembly_i) {
                auto records = seqwin::read_fasta(assembly_paths[assembly_i]);

                for (std::size_t record_i = 0; record_i < records.size(); ++record_i) {
                    const auto record_idx = record_offsets[assembly_i] + record_i;
                    auto& record = records[record_i];
                    if (record.sequence.size() > std::numeric_limits<std::uint32_t>::max()) {
                        throw std::runtime_error(
                            "Sequence length exceeds uint32 range for record " +
                            record.id + " in assembly " + assembly_paths[assembly_i]);
                    }

                    const auto mins = btllib::minimize_sequence(
                        record.sequence,
                        kmerlen,
                        windowsize
                    );

                    for (const auto& m : mins) {
                        auto cursor_it = hash_to_cursor.find(m.out_hash);
                        if (cursor_it == hash_to_cursor.end()) {
                            throw std::runtime_error(
                                "Low-memory minimizer recomputation found an unknown minimizer hash"
                            );
                        }
                        auto& cursor = cursor_it->second;
                        kmers[cursor++] = Kmer{
                            static_cast<std::uint32_t>(m.pos),
                            static_cast<std::uint32_t>(record_idx)
                        };
                    }
                }
            }
        }
    });

    return kmers;
}

} // namespace

Graph build(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu,
    bool low_memory
) {
    if (assembly_paths.size() != is_targets.size()) {
        throw std::runtime_error("assembly_paths and is_targets must have the same length");
    }
    if (assembly_paths.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Number of input assemblies exceeds uint32 range");
    }

    const auto n_assemblies = assembly_paths.size();
    std::size_t n_workers = std::max<std::size_t>(1, n_cpu);
    if (n_assemblies > 0) {
        n_workers = std::min(n_workers, n_assemblies);
    }

    ThreadPool pool(n_workers); // Avoid spawning threads every time
    std::vector<ThreadGraph> graphs(n_workers);

    const std::size_t base = n_assemblies / n_workers;
    const std::size_t rem = n_assemblies % n_workers;

    pool.parallel_for(n_workers, [&](std::size_t start, std::size_t end, std::size_t) {
        for (std::size_t thread_id = start; thread_id < end; ++thread_id) {
            std::size_t start_assembly = thread_id * base + std::min(thread_id, rem);
            std::size_t end_assembly = start_assembly + base + (thread_id < rem ? 1 : 0);
            graphs[thread_id] = build_worker(
                assembly_paths,
                kmerlen,
                windowsize,
                is_targets,
                start_assembly,
                end_assembly,
                thread_id,
                low_memory
            );
        }
    });

    NodeMaps node_maps;
    auto graph = merge_thread_graphs(
        graphs,
        n_assemblies,
        pool,
        low_memory,
        low_memory ? &node_maps : nullptr
    );
    if (low_memory) {
        log_python(" - Recomputing minimizers for low-memory mode...");
        graph.kmers = recompute_kmers(
            assembly_paths,
            kmerlen,
            windowsize,
            graphs,
            graph.record_offsets,
            node_maps,
            pool
        );
    }
    return graph;
}

} // namespace seqwin

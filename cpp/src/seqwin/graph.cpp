#include "seqwin/graph.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "seqwin/fasta_reader.hpp"
#include "btllib/minimizer.hpp"

namespace seqwin {
namespace {

constexpr std::size_t serialized_record_size = 17;

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

struct ThreadResult {
    std::vector<std::uint8_t> kmers;
    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash> edge_map;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::size_t start_assembly;
};

void append_record(
    std::vector<std::uint8_t>& out,
    std::uint64_t out_hash,
    std::uint32_t pos,
    std::uint16_t record_idx,
    std::uint16_t assembly_idx,
    std::uint8_t is_target
) {
    const auto old_size = out.size();
    out.resize(old_size + serialized_record_size);
    auto* record = out.data() + old_size;

    std::memcpy(record + 0, &out_hash, sizeof(out_hash));
    std::memcpy(record + 8, &pos, sizeof(pos));
    std::memcpy(record + 12, &record_idx, sizeof(record_idx));
    std::memcpy(record + 14, &assembly_idx, sizeof(assembly_idx));
    std::memcpy(record + 16, &is_target, sizeof(is_target));
}

ThreadResult build_worker(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t start_assembly,
    std::size_t end_assembly
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
        ) * serialized_record_size
    );
    result.ids_by_assembly.reserve(end_assembly - start_assembly);

    for (std::size_t assembly_i = start_assembly; assembly_i < end_assembly; ++assembly_i) {
        const auto assembly_idx = assembly_indices[assembly_i];
        if (assembly_idx > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("assembly_idx must fit in uint16");
        }
        const auto assembly_idx16 = static_cast<std::uint16_t>(assembly_idx);
        const std::uint8_t is_target_u8 =
            is_targets[assembly_i] ? std::uint8_t{1} : std::uint8_t{0};

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
                append_record(
                    result.kmers,
                    m.out_hash,
                    static_cast<std::uint32_t>(m.pos),
                    record_idx16,
                    assembly_idx16,
                    is_target_u8);
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
                auto [it, inserted] = result.edge_map.try_emplace(
                    key,
                    EdgeState{1, assembly_i}
                );
                if (!inserted && it->second.last_seen_assembly != assembly_i) {
                    ++(it->second.weight);
                    it->second.last_seen_assembly = assembly_i;
                }
            }
        }
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

    // Divide assemblies into n_workers chunks for multithreading
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
                end
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

    std::sort(results.begin(), results.end(), [](const ThreadResult& a, const ThreadResult& b) {
        return a.start_assembly < b.start_assembly;
    });

    // Merge results from all threads
    std::size_t total_kmer_bytes = 0;
    std::size_t total_edge_entries = 0;
    for (std::size_t i = 0; i < results.size(); ++i) {
        total_kmer_bytes += results[i].kmers.size();
        total_edge_entries += results[i].edge_map.size();
    }

    // Merge kmers
    std::vector<std::uint8_t> kmers;
    if (!results.empty()) {
        kmers = std::move(results[0].kmers);

        if (results.size() > 1) {
            std::vector<std::size_t> offsets(results.size(), 0);
            std::size_t cursor = kmers.size();
            for (std::size_t i = 1; i < results.size(); ++i) {
                offsets[i] = cursor;
                cursor += results[i].kmers.size();
            }

            kmers.resize(total_kmer_bytes);
            std::vector<std::thread> threads;
            threads.reserve(results.size() - 1);
            for (std::size_t i = 1; i < results.size(); ++i) {
                threads.emplace_back([&, i]() {
                    auto& local_kmers = results[i].kmers;
                    const auto size = local_kmers.size();
                    if (size != 0) {
                        std::memcpy(kmers.data() + offsets[i], local_kmers.data(), size);
                    }
                    std::vector<std::uint8_t>().swap(local_kmers);
                });
            }
            for (auto& t : threads) {
                t.join();
            }
        }
    }

    // Merge edges
    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash> edge_map;
    if (!results.empty()) {
        edge_map = std::move(results[0].edge_map);
        edge_map.reserve(total_edge_entries);
    }

    for (std::size_t i = 1; i < results.size(); ++i) {
        auto& src = results[i].edge_map;
        for (auto it = src.begin(); it != src.end();) {
            auto dst = edge_map.find(it->first);
            if (dst != edge_map.end()) {
                dst->second.weight += it->second.weight;
                it = src.erase(it);
            } else {
                auto nh = src.extract(it++);
                edge_map.insert(std::move(nh));
            }
        }
        std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash>().swap(src);
    }

    std::vector<std::uint64_t> edges;
    edges.reserve(edge_map.size() * 3);
    for (const auto& [key, state] : edge_map) {
        edges.push_back(key.first);
        edges.push_back(key.second);
        edges.push_back(state.weight);
    }

    // Merge record IDs
    std::vector<std::vector<std::string>> all_idx_to_id;
    if (!results.empty()) {
        all_idx_to_id = std::move(results[0].ids_by_assembly);

        if (results.size() > 1) {
            all_idx_to_id.resize(n_assemblies);
            for (std::size_t i = 1; i < results.size(); ++i) {
                for (std::size_t j = 0; j < results[i].ids_by_assembly.size(); ++j) {
                    all_idx_to_id[results[i].start_assembly + j] = std::move(results[i].ids_by_assembly[j]);
                }
            }
        }
    }

    return {std::move(kmers), std::move(edges), std::move(all_idx_to_id)};
}

} // namespace seqwin

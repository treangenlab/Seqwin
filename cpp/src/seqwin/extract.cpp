#include "seqwin/extract.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <ankerl/unordered_dense.h>

#include "utils/fasta_reader.hpp"

namespace seqwin::internal {
namespace {

struct FullKmer {
    std::uint64_t hash;
    std::uint32_t record_idx;
    std::uint32_t pos;
};

struct VectorHash {
    using is_avalanching = void;

    std::uint64_t operator()(const std::vector<std::uint64_t>& values) const noexcept
    {
        std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
        for (const auto value : values) {
            hash ^= ankerl::unordered_dense::hash<std::uint64_t>{}(value);
            hash *= UINT64_C(0x100000001b3);
        }
        return hash;
    }
};

struct CanonicalCount {
    std::vector<std::uint64_t> order;
    std::size_t count = 0;
    std::size_t forward_count = 0;
    std::size_t reverse_count = 0;
};

} // namespace

std::optional<Signature> extract_worker(
    std::size_t subgraph_idx,
    const std::vector<std::size_t>& subgraph,
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    std::size_t total_tar,
    double consec_kmer_mul
) {
    std::vector<FullKmer> sg_kmers; // K-mers of the current subgraph
    std::size_t n_sg_kmers = 0;
    for (const auto node_idx : subgraph) {
        if (node_idx >= n_nodes) {
            throw std::invalid_argument("subgraph node index is out of bounds");
        }
        n_sg_kmers += nodes[node_idx].stop - nodes[node_idx].start;
    }
    sg_kmers.reserve(n_sg_kmers);
    for (const auto node_idx : subgraph) {
        const auto& node = nodes[node_idx];
        for (std::size_t i = node.start; i < node.stop; ++i) {
            sg_kmers.push_back({node.hash, kmers[i].record_idx, kmers[i].pos});
        }
    }

    std::stable_sort(sg_kmers.begin(), sg_kmers.end(), [](const auto& a, const auto& b) {
        if (a.record_idx != b.record_idx) {
            return a.record_idx < b.record_idx;
        } else {
            return a.pos < b.pos;
        }
    });

    std::vector<ConsecutiveKmers> all_runs;
    all_runs.reserve(std::min(n_assemblies, sg_kmers.size()));
    const double max_gap = consec_kmer_mul * windowsize;
    std::size_t assembly_idx = 0;
    for (std::size_t begin = 0; begin < sg_kmers.size();) {
        while (
            assembly_idx + 1 < n_record_offsets &&
            sg_kmers[begin].record_idx >= record_offsets[assembly_idx + 1]
        ) {
            ++assembly_idx;
        }
        if (
            assembly_idx >= n_assemblies ||
            sg_kmers[begin].record_idx < record_offsets[assembly_idx]
        ) {
            throw std::invalid_argument("k-mer record_idx is outside record_offsets");
        }

        const std::size_t assembly_end_record = record_offsets[assembly_idx + 1];
        ConsecutiveKmers best_run{};
        std::size_t n_repeats = 0;
        while (
            begin < sg_kmers.size() &&
            sg_kmers[begin].record_idx < assembly_end_record
        ) {
            std::size_t end = begin + 1;
            while (
                end < sg_kmers.size() &&
                sg_kmers[end].record_idx == sg_kmers[begin].record_idx &&
                static_cast<double>(sg_kmers[end].pos - sg_kmers[end - 1].pos) <= max_gap
            ) {
                ++end;
            }
            ++n_repeats;

            const std::size_t run_size = end - begin;
            if (run_size > best_run.location.n_kmers) {
                best_run = ConsecutiveKmers{
                    {
                        assembly_idx,
                        sg_kmers[begin].record_idx - record_offsets[assembly_idx],
                        sg_kmers[begin].pos,
                        static_cast<std::uint32_t>(sg_kmers[end - 1].pos + kmerlen),
                        run_size,
                        0 // n_repeats
                    },
                    is_targets[assembly_idx],
                    {} // order
                };
                best_run.order.reserve(run_size);
                for (std::size_t i = begin; i < end; ++i) {
                    best_run.order.push_back(sg_kmers[i].hash);
                }
            }
            begin = end;
        }
        best_run.location.n_repeats = n_repeats;
        all_runs.push_back(std::move(best_run));
    }

    std::vector<CanonicalCount> canonical_counts;
    canonical_counts.reserve(all_runs.size());
    ankerl::unordered_dense::map<
        std::vector<std::uint64_t>,
        std::size_t,
        VectorHash
    > indices; // Map a canonical order to its index in canonical_counts
    indices.reserve(all_runs.size());
    for (const auto& run : all_runs) {
        if (!run.is_target) {
            continue;
        }
        std::vector<std::uint64_t> reverse(run.order.rbegin(), run.order.rend());
        const bool is_forward = run.order <= reverse;
        const auto& key = is_forward ? run.order : reverse;

        auto [counts_it, inserted] = indices.try_emplace(key, canonical_counts.size());
        if (inserted) {
            canonical_counts.push_back({key});
        }
        auto& counts = canonical_counts[counts_it->second];
        ++counts.count;
        ++(is_forward ? counts.forward_count : counts.reverse_count);
    }
    if (canonical_counts.empty()) {
        throw std::invalid_argument("subgraph has no k-mers in target assemblies");
    }

    auto rep_canonical = canonical_counts.begin();
    for (auto it = canonical_counts.begin() + 1; it != canonical_counts.end(); ++it) {
        if (it->order.size() * it->count > rep_canonical->order.size() * rep_canonical->count) {
            rep_canonical = it;
        }
    }
    auto rep_order = rep_canonical->order;
    if (rep_canonical->reverse_count > rep_canonical->forward_count) {
        std::reverse(rep_order.begin(), rep_order.end());
    }

    if (rep_order.size() == 1) {
        return std::nullopt;
    }
    ankerl::unordered_dense::set<std::uint64_t> unique_hashes;
    unique_hashes.reserve(rep_order.size());
    for (const auto hash : rep_order) {
        if (!unique_hashes.insert(hash).second) {
            return std::nullopt;
        }
    }

    const auto rep_run = std::find_if(all_runs.begin(), all_runs.end(), [&](const auto& run) {
        return run.order == rep_order;
    });
    if (rep_run == all_runs.end()) {
        throw std::logic_error("representative signature location not found");
    }
    const std::size_t length = rep_run->location.stop - rep_run->location.start;
    if (length < min_len) {
        return std::nullopt;
    }

    return Signature{
        subgraph_idx,
        rep_run->location,
        {}, // sequence
        length,
        rep_canonical->count,
        static_cast<double>(rep_canonical->count) / static_cast<double>(total_tar)
    };
}

void fetch_signature_sequences(
    std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    ThreadPool& pool
) {
    struct RequestGroup {
        std::size_t assembly_idx;
        std::vector<std::size_t> signature_indices;
    };

    std::vector<RequestGroup> groups;
    groups.reserve(std::min(assembly_paths.size(), signatures.size()));
    std::vector<std::size_t> group_indices(assembly_paths.size(), signatures.size());
    for (std::size_t i = 0; i < signatures.size(); ++i) {
        const auto assembly_idx = signatures[i].location.assembly_idx;
        if (assembly_idx >= assembly_paths.size()) {
            throw std::runtime_error("signature assembly index is outside assembly paths");
        }
        if (group_indices[assembly_idx] == signatures.size()) {
            group_indices[assembly_idx] = groups.size();
            groups.push_back({assembly_idx, {}});
        }
        groups[group_indices[assembly_idx]].signature_indices.push_back(i);
    }

    pool.parallel_for(groups.size(), [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t group_idx = begin; group_idx < end; ++group_idx) {
            const auto& group = groups[group_idx];
            const auto records = read_fasta(assembly_paths[group.assembly_idx]);

            for (const auto signature_idx : group.signature_indices) {
                auto& signature = signatures[signature_idx];
                if (signature.location.record_idx >= records.size()) {
                    throw std::runtime_error("signature record index is outside assembly FASTA");
                }
                const auto& sequence = records[signature.location.record_idx].sequence;
                const auto start = std::min<std::size_t>(signature.location.start, sequence.size());
                const auto stop = std::min<std::size_t>(signature.location.stop, sequence.size());
                signature.sequence = sequence.substr(start, stop > start ? stop - start : 0);
            }
        }
    });
}

} // namespace seqwin::internal

namespace seqwin {

std::vector<Signature> extract(
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const Subgraphs& subgraphs,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    const std::vector<std::string>& assembly_paths,
    const ExtractConfig& config
) {
    if (n_record_offsets != n_assemblies + 1 || assembly_paths.size() != n_assemblies) {
        throw std::invalid_argument("assembly metadata dimensions do not match");
    }
    if (config.total_tar == 0) {
        throw std::invalid_argument("at least one target assembly is required");
    }
    if (subgraphs.empty()) {
        return {};
    }

    std::vector<std::optional<Signature>> extracted(subgraphs.size());
    const std::size_t n_workers = std::min(
        std::max<std::size_t>(1, config.n_cpu), subgraphs.size()
    );
    internal::ThreadPool pool(n_workers);
    pool.parallel_for(subgraphs.size(), [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t i = begin; i < end; ++i) {
            extracted[i] = internal::extract_worker(
                i,
                subgraphs[i],
                kmers,
                nodes,
                n_nodes,
                record_offsets,
                n_record_offsets,
                is_targets,
                n_assemblies,
                config.kmerlen,
                config.windowsize,
                config.min_len,
                config.total_tar,
                config.consec_kmer_mul
            );
        }
    });

    std::vector<Signature> signatures;
    signatures.reserve(subgraphs.size());
    for (auto& signature : extracted) {
        if (signature) {
            signatures.push_back(std::move(*signature));
        }
    }

    internal::fetch_signature_sequences(signatures, assembly_paths, pool);
    return signatures;
}

} // namespace seqwin

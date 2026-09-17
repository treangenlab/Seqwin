#include "seqwin/extract.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <tuple>
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

/** Hash a vector of k-mer hashes using an FNV-1a-style combining scheme. */
struct VectorHash {
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

/**
 * Count the number of target assemblies containing a canonical k-mer order.
 * The first encountered order wins ties.
 */
struct CanonicalCount {
    std::size_t count = 0;
    std::size_t forward_count = 0;
    std::size_t reverse_count = 0;
    std::size_t first_seen = 0;
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
    // Collect and position-sort all k-mers included in the subgraph
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

    // Select the longest consecutive k-mer run from each assembly
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
        std::size_t best_begin = begin;
        std::size_t best_end = begin;
        std::size_t best_run_size = 0;
        std::size_t n_repeats = 0;
        while (
            begin < sg_kmers.size() &&
            sg_kmers[begin].record_idx < assembly_end_record
        ) {
            std::size_t end = begin + 1;
            while (
                end < sg_kmers.size() &&
                sg_kmers[end].record_idx == sg_kmers[begin].record_idx &&
                sg_kmers[end].pos - sg_kmers[end - 1].pos <= max_gap
            ) {
                ++end;
            }
            ++n_repeats;

            const std::size_t run_size = end - begin;
            if (run_size > best_run_size) {
                best_begin = begin;
                best_end = end;
                best_run_size = run_size;
            }
            begin = end;
        }

        ConsecutiveKmers best_run{
            {
                assembly_idx,
                sg_kmers[best_begin].record_idx - record_offsets[assembly_idx],
                sg_kmers[best_begin].pos,
                sg_kmers[best_end - 1].pos + kmerlen,
                best_run_size,
                n_repeats
            },
            is_targets[assembly_idx],
            {} // kmers
        };
        best_run.kmers.reserve(best_run_size);
        for (std::size_t i = best_begin; i < best_end; ++i) {
            best_run.kmers.push_back(sg_kmers[i].hash);
        }
        all_runs.push_back(std::move(best_run));
    }

    // K-mer runs in different assemblies may have the same k-mer order (regardless of orientation)
    // Count the number of target assemblies for each unique canonical k-mer order
    ankerl::unordered_dense::map<
        std::vector<std::uint64_t>,
        CanonicalCount,
        VectorHash
    > canonical_counts;
    canonical_counts.reserve(all_runs.size());
    std::size_t first_seen = 0;
    for (const auto& run : all_runs) {
        if (!run.is_target) {
            continue;
        }

        const bool is_forward = std::lexicographical_compare(
            run.kmers.begin(), run.kmers.end(),
            run.kmers.rbegin(), run.kmers.rend()
        );
        decltype(canonical_counts)::iterator it;
        bool inserted;
        if (is_forward) {
            std::tie(it, inserted) = canonical_counts.try_emplace(
                run.kmers,
                CanonicalCount{0, 0, 0, first_seen}
            );
        } else {
            std::vector<std::uint64_t> reverse(run.kmers.rbegin(), run.kmers.rend());
            std::tie(it, inserted) = canonical_counts.try_emplace(
                std::move(reverse),
                CanonicalCount{0, 0, 0, first_seen}
            );
        }
        if (inserted) {
            ++first_seen;
        }
        auto& cnt = it->second;
        ++cnt.count;
        ++(is_forward ? cnt.forward_count : cnt.reverse_count);
    }
    if (canonical_counts.empty()) {
        throw std::invalid_argument("subgraph has no k-mers in target assemblies");
    }

    // Choose the highest-scoring canonical order
    // Choose the first encountered order (smaller assembly index) when scores are the same
    auto best_order = canonical_counts.begin();
    for (auto it = canonical_counts.begin(); it != canonical_counts.end(); ++it) {
        const auto score = it->first.size() * it->second.count;
        const auto best_score = best_order->first.size() * best_order->second.count;
        if (
            score > best_score ||
            (score == best_score && it->second.first_seen < best_order->second.first_seen)
        ) {
            best_order = it;
        }
    }
    // Choose the most common orientation as the representative
    auto rep = best_order->first;
    const auto& rep_count = best_order->second;
    if (rep_count.reverse_count > rep_count.forward_count) {
        std::reverse(rep.begin(), rep.end());
    }

    // Reject representatives with only one k-mer
    if (rep.size() == 1) {
        return std::nullopt;
    }
    // Reject representatives with duplicate k-mers
    ankerl::unordered_dense::set<std::uint64_t> unique_hashes;
    unique_hashes.reserve(rep.size());
    for (const auto hash : rep) {
        if (!unique_hashes.insert(hash).second) {
            return std::nullopt;
        }
    }

    // Find the first target assembly containing the representative
    const auto rep_run = std::find_if(all_runs.begin(), all_runs.end(), [&](const auto& run) {
        return run.is_target && run.kmers == rep;
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
        rep_count.count,
        rep_count.count / static_cast<double>(total_tar)
    };
}

void fetch_signature_sequences(
    std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    ThreadPool& pool
) {
    if (signatures.empty()) {
        throw std::invalid_argument("no valid signatures found in subgraphs");
    }

    struct RequestGroup {
        std::size_t assembly_idx;
        std::vector<std::size_t> signature_indices;
    };

    // Group requests by assembly so that each FASTA file is read only once
    std::vector<RequestGroup> groups;
    groups.reserve(std::min(assembly_paths.size(), signatures.size()));
    ankerl::unordered_dense::map<std::size_t, std::size_t> group_indices;
    group_indices.reserve(std::min(assembly_paths.size(), signatures.size()));

    for (std::size_t i = 0; i < signatures.size(); ++i) {
        const auto assembly_idx = signatures[i].location.assembly_idx;
        if (assembly_idx >= assembly_paths.size()) {
            throw std::runtime_error("signature assembly index is outside assembly paths");
        }
        auto [group_it, inserted] = group_indices.try_emplace(assembly_idx, groups.size());
        if (inserted) {
            groups.push_back({assembly_idx, {}});
        }
        groups[group_it->second].signature_indices.push_back(i);
    }

    // Read different assemblies in parallel and slice each requested sequence interval
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
                std::transform(
                    signature.sequence.begin(), signature.sequence.end(),
                    signature.sequence.begin(),
                    [](unsigned char base) { return static_cast<char>(std::toupper(base)); }
                );
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
    if (!std::isfinite(config.consec_kmer_mul) || config.consec_kmer_mul <= 0.0) {
        throw std::invalid_argument("consec_kmer_mul must be finite and greater than zero");
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

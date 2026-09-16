#include "seqwin/extract.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "utils/fasta_reader.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {
namespace {

constexpr double CONSEC_KMER_MUL = 1.5;

struct FullKmer {
    std::uint64_t hash;
    std::uint32_t assembly_idx;
    std::uint32_t record_idx;
    std::uint32_t pos;
};

std::pair<std::uint32_t, std::uint32_t> locate_record(
    std::uint32_t record_idx,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets
) {
    const auto* idx = std::upper_bound(
        record_offsets,
        record_offsets + n_record_offsets,
        record_idx
    );
    if (idx == record_offsets || idx == record_offsets + n_record_offsets) {
        throw std::invalid_argument("k-mer record_idx is outside record_offsets");
    }
    const std::uint32_t assembly_idx = idx - record_offsets - 1;
    return {assembly_idx, record_idx - record_offsets[assembly_idx]};
}

std::vector<std::uint64_t> canonical(
    const std::vector<std::uint64_t>& order
) {
    std::vector<std::uint64_t> reverse(order.rbegin(), order.rend());
    return std::min(order, reverse);
}

std::size_t count_order(
    const std::vector<ConsecutiveKmers>& runs,
    const std::vector<std::uint64_t>& order
) {
    return static_cast<std::size_t>(std::count_if(
        runs.begin(), runs.end(), [&](const ConsecutiveKmers& run) {
            return run.is_target && run.order == order;
        }
    ));
}

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
    std::size_t total_tar
) {
    std::vector<FullKmer> sg_kmers; // K-mers of the current subgraph
    for (const auto node_idx : subgraph) {
        if (node_idx >= n_nodes) {
            throw std::invalid_argument("subgraph node index is out of bounds");
        }

        const auto& node = nodes[node_idx];
        for (std::size_t i = node.start; i < node.stop; ++i) {
            const auto [assembly_idx, local_record_idx] = locate_record(
                kmers[i].record_idx, record_offsets, n_record_offsets
            );
            if (assembly_idx >= n_assemblies) {
                throw std::invalid_argument("record_offsets refers to an unknown assembly");
            }
            sg_kmers.push_back({
                node.hash,
                assembly_idx,
                local_record_idx,
                kmers[i].pos
            });
        }
    }

    std::stable_sort(sg_kmers.begin(), sg_kmers.end(), [](const auto& a, const auto& b) {
        if (a.assembly_idx != b.assembly_idx) return a.assembly_idx < b.assembly_idx;
        if (a.record_idx != b.record_idx) return a.record_idx < b.record_idx;
        return a.pos < b.pos;
    });

    std::vector<ConsecutiveKmers> all_runs;
    const double max_gap = CONSEC_KMER_MUL * windowsize;
    for (std::size_t begin = 0; begin < sg_kmers.size();) {
        const auto assembly_idx = sg_kmers[begin].assembly_idx;
        std::vector<ConsecutiveKmers> assembly_runs;

        while (begin < sg_kmers.size() && sg_kmers[begin].assembly_idx == assembly_idx) {
            std::size_t end = begin + 1;
            const double gap = sg_kmers[end].pos - sg_kmers[end - 1].pos;
            while (
                end < sg_kmers.size()
                && sg_kmers[end].assembly_idx == assembly_idx
                && sg_kmers[end].record_idx == sg_kmers[begin].record_idx
                && gap <= max_gap
            ) {
                ++end;
            }
            ConsecutiveKmers run{
                {
                    assembly_idx,
                    sg_kmers[begin].record_idx,
                    sg_kmers[begin].pos,
                    static_cast<std::uint32_t>(sg_kmers[end - 1].pos + kmerlen),
                    end - begin,
                    0
                },
                is_targets[assembly_idx],
                {}
            };
            run.order.reserve(end - begin);
            for (std::size_t i = begin; i < end; ++i) {
                run.order.push_back(sg_kmers[i].hash);
            }
            assembly_runs.push_back(std::move(run));
            begin = end;
        }

        if (!assembly_runs.empty()) {
            auto best = assembly_runs.begin();
            for (auto it = assembly_runs.begin() + 1; it != assembly_runs.end(); ++it) {
                if (it->location.n_kmers > best->location.n_kmers) {
                    best = it;
                }
            }
            best->location.n_repeats = assembly_runs.size();
            all_runs.push_back(std::move(*best));
        }
    }

    std::vector<std::pair<std::vector<std::uint64_t>, std::size_t>> canonical_counts;
    for (const auto& run : all_runs) {
        if (!run.is_target) {
            continue;
        }
        auto key = canonical(run.order);
        auto found = std::find_if(canonical_counts.begin(), canonical_counts.end(), [&](const auto& item) {
            return item.first == key;
        });
        if (found == canonical_counts.end()) {
            canonical_counts.push_back({std::move(key), 1});
        } else {
            ++found->second;
        }
    }
    if (canonical_counts.empty()) {
        throw std::invalid_argument("subgraph has no k-mers in target assemblies");
    }
    auto rep_canonical = canonical_counts.begin();
    for (auto it = canonical_counts.begin() + 1; it != canonical_counts.end(); ++it) {
        if (it->first.size() * it->second > rep_canonical->first.size() * rep_canonical->second) {
            rep_canonical = it;
        }
    }
    auto rep_order = rep_canonical->first;
    std::vector<std::uint64_t> reverse(rep_order.rbegin(), rep_order.rend());
    if (count_order(all_runs, reverse) > count_order(all_runs, rep_order)) {
        rep_order = std::move(reverse);
    }

    if (rep_order.size() == 1) {
        return std::nullopt;
    }
    auto sorted_hashes = rep_order;
    std::sort(sorted_hashes.begin(), sorted_hashes.end());
    if (std::adjacent_find(sorted_hashes.begin(), sorted_hashes.end()) != sorted_hashes.end()) {
        return std::nullopt;
    }

    const auto location = std::find_if(all_runs.begin(), all_runs.end(), [&](const auto& run) {
        return run.order == rep_order;
    });
    if (location == all_runs.end()) {
        throw std::logic_error("representative signature location not found");
    }
    const std::size_t length = location->location.stop - location->location.start;
    if (length < min_len) {
        return std::nullopt;
    }

    return Signature{
        subgraph_idx,
        location->location,
        {},
        length,
        rep_canonical->second,
        static_cast<double>(rep_canonical->second) / static_cast<double>(total_tar)
    };
}

void fetch_signature_sequences(
    std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    std::size_t n_cpu
) {
    std::vector<std::vector<std::size_t>> requests(assembly_paths.size());
    for (std::size_t i = 0; i < signatures.size(); ++i) {
        requests[signatures[i].location.assembly_idx].push_back(i);
    }
    ThreadPool pool(n_cpu);
    pool.parallel_for(assembly_paths.size(), [&](std::size_t begin, std::size_t end, std::size_t) {
        for (std::size_t assembly_idx = begin; assembly_idx < end; ++assembly_idx) {
            if (requests[assembly_idx].empty()) continue;
            const auto records = read_fasta(assembly_paths[assembly_idx]);
            for (const auto signature_idx : requests[assembly_idx]) {
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
    const std::size_t total_tar = std::count(is_targets, is_targets + n_assemblies, true);
    if (total_tar == 0) {
        throw std::invalid_argument("at least one target assembly is required");
    }

    std::vector<std::optional<Signature>> extracted(subgraphs.size());
    internal::ThreadPool pool(config.n_cpu);
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
                total_tar
            );
        }
    });

    std::vector<Signature> signatures;
    signatures.reserve(subgraphs.size());
    for (auto& signature : extracted) {
        if (signature) signatures.push_back(std::move(*signature));
    }

    internal::fetch_signature_sequences(signatures, assembly_paths, config.n_cpu);
    return signatures;
}

} // namespace seqwin

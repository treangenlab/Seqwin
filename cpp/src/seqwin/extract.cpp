#include "seqwin/filter_internals.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include <ankerl/unordered_dense.h>

#include "seqwin/shared_internals.hpp"
#include "utils/fasta_reader.hpp"
#include "utils/logging.hpp"

namespace seqwin::internal {
namespace {

constexpr std::size_t fasta_width = 80;

struct SubgraphKmer {
    std::size_t node_idx;
    std::uint32_t record_idx;
    std::uint32_t pos;
};

/**
 * @brief A run of consecutive k-mers found in an assembly.
 *
 * K-mers are consecutive when their positions in the same FASTA
 * record differ by at most the configured maximum gap.
 */
struct ConsecutiveKmers {
    /** Location spanned by the k-mer run. */
    SubgraphLoc location;
    /** Whether the containing assembly belongs to the target set. */
    bool is_target;
    /** Node indices sorted by k-mer position. */
    std::vector<std::size_t> node_indices;
};

struct VectorHash {
    std::uint64_t operator()(const std::vector<std::size_t>& values) const noexcept
    {
        std::uint64_t hash = UINT64_C(0xcbf29ce484222325);
        for (const auto value : values) {
            hash ^= ankerl::unordered_dense::hash<std::size_t>{}(value);
            hash *= UINT64_C(0x100000001b3);
        }
        return hash;
    }
};

/**
 * @brief Count the number of target assemblies containing a canonical k-mer order.
 * The first encountered order wins ties.
 */
struct CanonicalCount {
    std::size_t count = 0;
    std::size_t forward_count = 0;
    std::size_t reverse_count = 0;
    std::size_t first_seen = 0;
};

void write_csv_field(std::ostream& out, std::string_view value) {
    if (value.find_first_of(",\"\r\n") == std::string_view::npos) {
        out << value;
        return;
    }
    out << '"';
    for (const char c : value) {
        if (c == '"') {
            out << '"';
        }
        out << c;
    }
    out << '"';
}

/**
 * @brief Extract a signature from one low-penalty subgraph.
 *
 * Return `std::nullopt` if the signature is invalid.
 */
std::optional<Signature> extract_worker(
    std::size_t subgraph_idx,
    const Subgraph& subgraph,
    const Kmer* kmers,
    std::size_t n_kmers,
    const std::size_t* offsets,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    double consec_kmer_mul,
    std::size_t total_tar
) {
    // Collect and position-sort all k-mers included in the subgraph
    std::vector<SubgraphKmer> sg_kmers; // K-mers of the current subgraph
    // Each node in a low-penalty subgraph should contain roughly total_tar k-mers
    sg_kmers.reserve(2 * total_tar * subgraph.size());
    for (const auto node_idx : subgraph) {
        if (node_idx >= n_nodes) {
            throw std::invalid_argument("subgraph node index is out of bounds");
        }
        const auto start = offsets[node_idx];
        const auto stop = offsets[node_idx + 1];
        if (start >= stop || stop > n_kmers) {
            throw std::invalid_argument("Node k-mer range is invalid");
        }
        for (std::size_t i = start; i < stop; ++i) {
            sg_kmers.push_back({node_idx, kmers[i].record_idx, kmers[i].pos});
        }
    }
    std::sort(sg_kmers.begin(), sg_kmers.end(), [](const auto& a, const auto& b) {
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
            {}
        };
        best_run.node_indices.reserve(best_run_size);
        for (std::size_t i = best_begin; i < best_end; ++i) {
            best_run.node_indices.push_back(sg_kmers[i].node_idx);
        }
        all_runs.push_back(std::move(best_run));
    }

    // K-mer runs in different assemblies may have the same k-mer order (regardless of orientation)
    // Count the number of target assemblies for each unique canonical k-mer order
    ankerl::unordered_dense::map<
        std::vector<std::size_t>,
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
            run.node_indices.begin(), run.node_indices.end(),
            run.node_indices.rbegin(), run.node_indices.rend()
        );
        decltype(canonical_counts)::iterator it;
        bool inserted;
        if (is_forward) {
            std::tie(it, inserted) = canonical_counts.try_emplace(
                run.node_indices,
                CanonicalCount{0, 0, 0, first_seen}
            );
        } else {
            std::vector<std::size_t> reverse(run.node_indices.rbegin(), run.node_indices.rend());
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
    ankerl::unordered_dense::set<std::size_t> unique_indices;
    unique_indices.reserve(rep.size());
    for (const auto node_idx : rep) {
        if (!unique_indices.insert(node_idx).second) {
            return std::nullopt;
        }
    }

    // Find the first target assembly containing the representative
    const auto rep_run = std::find_if(all_runs.begin(), all_runs.end(), [&](const auto& run) {
        return run.is_target && run.node_indices == rep;
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
        rep_count.count
    };
}

/**
 * @brief Fetch signature sequences and write FASTA and CSV.
 */
void fetch_signature_sequences(
    const std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    const std::string& fasta_path,
    const std::string& csv_path,
    std::size_t total_tar,
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
    log_python(
        " - Fetching signature sequences (" + std::to_string(groups.size()) + " assemblies to be loaded)..."
    );
    std::vector<FastaRecord> fasta_records(signatures.size());
    pool.parallel_for(groups.size(), [&](std::size_t group_idx) {
        const auto& group = groups[group_idx];
        const auto records = read_fasta(assembly_paths[group.assembly_idx]);

        for (const auto signature_idx : group.signature_indices) {
            const auto& signature = signatures[signature_idx];
            const auto& loc = signature.location;
            if (loc.record_idx >= records.size()) {
                throw std::runtime_error("signature record index is outside assembly FASTA");
            }
            const auto& record = records[loc.record_idx];
            if (loc.start > loc.stop || loc.stop > record.sequence.size()) {
                throw std::runtime_error("signature sequence bounds are outside assembly FASTA");
            }
            auto& out = fasta_records[signature_idx];
            out.id = std::to_string(loc.assembly_idx) + "-" + record.id + "-" +
                     std::to_string(loc.start) + ":" + std::to_string(loc.stop);
            out.sequence = record.sequence.substr(loc.start, loc.stop - loc.start);
            std::transform(
                out.sequence.begin(),
                out.sequence.end(),
                out.sequence.begin(),
                [](unsigned char base) {
                    return static_cast<char>(std::toupper(base));
                }
            );
        }
    });

    std::ofstream fasta;
    std::ofstream csv;
    const double total_tar_inv = 1.0 / total_tar;
    fasta.exceptions(std::ios::failbit | std::ios::badbit);
    csv.exceptions(std::ios::failbit | std::ios::badbit);
    fasta.open(fasta_path, std::ios::binary);
    csv.open(csv_path, std::ios::binary);
    csv.imbue(std::locale::classic());
    csv << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t i = 0; i < METADATA_NAMES.size(); ++i) {
        if (i > 0) {
            csv << ',';
        }
        write_csv_field(csv, METADATA_NAMES[i]);
    }
    csv << '\n';
    for (std::size_t i = 0; i < signatures.size(); ++i) {
        const auto& sig = signatures[i];
        const auto& rec = fasta_records[i];
        fasta << '>' << rec.id << '\n';
        for (std::size_t pos = 0; pos < rec.sequence.size(); pos += fasta_width) {
            fasta << rec.sequence.substr(pos, fasta_width) << '\n';
        }
        write_csv_field(csv, rec.id);
        csv << ','
            << sig.location.stop - sig.location.start << ','
            << sig.n_tar * total_tar_inv << ','
            << sig.location.n_kmers << ','
            << sig.location.n_repeats << '\n';
    }
    fasta.close();
    csv.close();
    log_python(" - Signature sequences saved as " + fasta_path);
    log_python(" - Signature metadata saved as " + csv_path);
}

} // namespace

std::vector<Signature> extract_signatures(
    const std::vector<Subgraph>& subgraphs,
    const Kmer* kmers,
    std::size_t n_kmers,
    const std::size_t* offsets,
    std::size_t n_nodes,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const std::vector<std::string>& assembly_paths,
    const bool* is_targets,
    std::size_t n_assemblies,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t min_len,
    double consec_kmer_mul,
    const std::optional<std::string>& fasta_path,
    const std::optional<std::string>& csv_path,
    std::size_t total_tar,
    ThreadPool& pool
) {
    if (n_record_offsets != n_assemblies + 1 || assembly_paths.size() != n_assemblies) {
        throw std::invalid_argument("assembly metadata dimensions do not match");
    }
    if (total_tar == 0) {
        throw std::invalid_argument("at least one target assembly is required");
    }
    if (!std::isfinite(consec_kmer_mul) || consec_kmer_mul <= 0.0) {
        throw std::invalid_argument("consec_kmer_mul must be finite and greater than zero");
    }
    if (subgraphs.empty()) {
        return {};
    }

    std::vector<std::optional<Signature>> extracted(subgraphs.size());
    pool.parallel_for(subgraphs.size(), [&](std::size_t subgraph_idx) {
        extracted[subgraph_idx] = extract_worker(
            subgraph_idx,
            subgraphs[subgraph_idx],
            kmers,
            n_kmers,
            offsets,
            n_nodes,
            record_offsets,
            n_record_offsets,
            is_targets,
            n_assemblies,
            kmerlen,
            windowsize,
            min_len,
            consec_kmer_mul,
            total_tar
        );
    });

    std::vector<Signature> signatures;
    signatures.reserve(subgraphs.size());
    for (auto& signature : extracted) {
        if (signature) {
            signatures.push_back(std::move(*signature));
        }
    }
    if (signatures.empty()) {
        throw std::invalid_argument("no valid signatures found in subgraphs");
    }
    log_python(" - Found " + std::to_string(signatures.size()) + " valid signatures");

    if (fasta_path && csv_path) {
        fetch_signature_sequences(signatures, assembly_paths, *fasta_path, *csv_path, total_tar, pool);
    }
    return signatures;
}

} // namespace seqwin::internal

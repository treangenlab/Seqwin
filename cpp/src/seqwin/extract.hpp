#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "seqwin/filter.hpp"
#include "utils/thread_pool.hpp"

namespace seqwin::internal {

/**
 * @brief A run of consecutive k-mers found in an assembly.
 */
struct ConsecutiveKmers {
    /** Location spanned by the k-mer run. */
    SubgraphLoc location;
    /** Whether the containing assembly belongs to the target set. */
    bool is_target;
    /** K-mer hashes in positional order. */
    std::vector<std::uint64_t> kmers;
};

/**
 * @brief Extract a signature from one low-penalty subgraph.
 *
 * Return `std::nullopt` if the signature is invalid.
 */
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
);

/**
 * @brief Fetch signature nucleotide sequences from their assembly FASTA files.
 *
 * Sequences are added to `signatures` in place.
 */
void fetch_signature_sequences(
    std::vector<Signature>& signatures,
    const std::vector<std::string>& assembly_paths,
    ThreadPool& pool
);

} // namespace seqwin::internal

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace seqwin {

/**
 * @brief Columns in the signature metadata CSV.
 *
 * @li fasta_header: `<SubgraphLoc::assembly_idx>-<record.id>-<SubgraphLoc::start>:<SubgraphLoc::stop>`
 * @li length: `SubgraphLoc::stop - SubgraphLoc::start`
 * @li f_tar: `Signature::n_tar / total_tar`
 * @li n_kmers: `SubgraphLoc::n_kmers`
 * @li n_repeats: `SubgraphLoc::n_repeats`
 */
inline constexpr std::array<std::string_view, 5> METADATA_NAMES = {
    "fasta_header", "length", "f_tar", "n_kmers", "n_repeats"
};

/**
 * @brief Location and minimizer metadata for a subgraph found in an assembly.
 *
 * This is determined by the longest consecutive minimizer run in the assembly,
 * when considering only the minimizers included in the subgraph.
 *
 * Note that a subgraph may appear more than once in an assembly.
 */
struct SubgraphLoc {
    /** Index of the assembly containing the subgraph. */
    std::size_t assembly_idx;
    /** Index of the FASTA record within the assembly. */
    std::size_t record_idx;
    /** 0-based start position in the FASTA record. */
    std::size_t start;
    /** Exclusive stop position in the FASTA record. */
    std::size_t stop;
    /** Size of the longest consecutive minimizer run in the assembly. */
    std::size_t n_kmers;
    /** Number of consecutive minimizer runs found in the assembly. */
    std::size_t n_repeats;
};

/**
 * @brief A signature is extracted from a low-penalty subgraph, represented by a
 * consecutive minimizer run (a.k.a. the representative) found in target assemblies.
 *
 * Forward and reversed k-mer orders are treated as the same canonical ordering.
 *
 * The nucleotide sequence of the signature is fetched from the first target assembly
 * containing the representative.
 */
struct Signature {
    /** Index of the low-penalty subgraph that produced the signature. */
    std::size_t subgraph_idx;
    /** Location and metadata of the representative. */
    SubgraphLoc location;
    /** Number of target assemblies containing the representative. */
    std::size_t n_tar;
};

} // namespace seqwin

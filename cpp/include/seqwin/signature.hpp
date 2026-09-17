#pragma once

#include <cstddef>
#include <string>

namespace seqwin {

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
 * The representative is insensitive of orientation (strand).
 *
 * The nucleotide sequence of the signature is fetched from the first target assembly
 * containing the representative.
 */
struct Signature {
    /** Index of the low-penalty subgraph that produced the signature. */
    std::size_t subgraph_idx;
    /** Location and metadata of the representative. */
    SubgraphLoc location;
    /** Nucleotide sequence of the signature. */
    std::string sequence;
    /** Length of the nucleotide sequence. */
    std::size_t length;
    /** Number of target assemblies containing the representative. */
    std::size_t n_rep;
    /** Fraction of target assemblies containing the representative. */
    double rep_ratio;
};

} // namespace seqwin

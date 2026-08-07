#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/no_init_array.hpp"

namespace seqwin {

/**
 * @brief Location metadata for a minimizer.
 */
struct Kmer {
    /** 0-based position of the minimizer within its FASTA record. */
    std::uint32_t pos;
    /** 0-based global index of the FASTA record. */
    std::uint32_t record_idx;
};

/**
 * @brief Minimizer graph node for one unique minimizer hash.
 *
 * The `[start, stop)` range is a half-open interval into `Graph.kmers`
 * for minimizers with this hash.
 */
struct Node {
    /** Hash value of the minimizers represented by this node. */
    std::uint64_t hash;
    /** Start of the half-open range for this node's minimizer entries. */
    std::size_t start;
    /** End of the half-open range for this node's minimizer entries. */
    std::size_t stop;
    /** Node scoring placeholder. */
    std::uint32_t n_tar = 0;
    /** Node scoring placeholder. */
    std::uint32_t n_neg = 0;
    /** Node scoring placeholder. */
    double penalty = 0.0;
};

/**
 * @brief Undirected weighted edge between two minimizers.
 */
struct Edge {
    /** Smaller endpoint hash of the undirected edge. */
    std::uint64_t first;
    /** Larger endpoint hash of the undirected edge. */
    std::uint64_t second;
    /** Number of assemblies where the endpoints are adjacent. */
    std::size_t weight;
};

/**
 * @brief Container for the minimizer graph returned by `build()`.
 */
struct Graph {
    /**
     * Minimizer occurrences in all assemblies, grouped and sorted by hash.
     * `record_idx` is nondecreasing within each node range.
     */
    NoInitArray<Kmer> kmers;
    /** Sorted by hash. */
    NoInitArray<Node> nodes;
    /** Sorted by hash. */
    NoInitArray<Edge> edges;
    /** Cumulative global FASTA record offsets by assembly. */
    std::vector<std::uint32_t> record_offsets;
    /** FASTA record IDs of each assembly. */
    std::vector<std::vector<std::string>> ids_by_assembly;
};

} // namespace seqwin

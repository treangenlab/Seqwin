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
 * The `[start, stop)` range identifies minimizers with this hash.
 * Before merging, it selects entries in `ThreadGraph.idx`, whose values are indices into `ThreadGraph.kmers`.
 * After merging, it indexes directly into the parallel `Graph.kmers` and `Graph.idx` arrays.
 */
struct Node {
    /** Hash value of the minimizer represented by this node. */
    std::uint64_t hash;
    /** Start of the half-open range for this node's minimizer entries. */
    std::uint64_t start;
    /** End of the half-open range for this node's minimizer entries. */
    std::uint64_t stop;
    /** Number of target assemblies containing this minimizer hash. */
    std::uint32_t n_tar;
    /** Number of non-target assemblies containing this minimizer hash. */
    std::uint32_t n_neg;
    /**
     * Node penalty score used for downstream graph filtering (set to 0.0).
     * Temporarily stores thread ID before thread-local graphs are merged.
     */
    double penalty;
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
    std::uint64_t weight;
};

/**
 * @brief Container for the minimizer graph returned by `build()`.
 */
struct Graph {
    /** Minimizer occurrences in all assemblies, grouped and sorted by hash. */
    NoInitArray<Kmer> kmers;
    /** Parallel to `kmers` and stores each minimizer's original generation index, ordered by genomic position. */
    NoInitArray<std::uint64_t> idx;
    /** Sorted by hash. */
    NoInitArray<Node> nodes;
    /** Sorted by hash. */
    NoInitArray<Edge> edges;
    /** Cumulative global FASTA record offsets by assembly. */
    std::vector<std::uint64_t> record_offsets;
    /** FASTA record IDs of each assembly. */
    std::vector<std::vector<std::string>> ids_by_assembly;
};

/**
 * @brief Build a minimizer graph from assembly FASTA files.
 *
 * `assembly_paths` and `is_targets` are parallel lists.
 *
 * @param assembly_paths Paths to input assemblies in FASTA format (plain or gzipped).
 * @param kmerlen K-mer length for minimizer sketch.
 * @param windowsize Window size for minimizer sketch.
 * @param is_targets Whether each assembly is a target assembly.
 * @param n_cpu Number of worker threads to use.
 * @return Minimizer graph.
 * @throws `std::runtime_error` If input sizes are inconsistent or counts exceed supported ranges.
 */
Graph build(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu
);

} // namespace seqwin

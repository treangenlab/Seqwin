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
    std::uint32_t pos;
    std::uint16_t record_idx;
    std::uint16_t assembly_idx;
};

/**
 * @brief Minimizer graph node for one unique minimizer hash.
 *
 * The `[start, stop)` range identifies minimizers with this hash.
 * Before merging, it selects entries in `ThreadGraph.idx`, whose values are indices into `ThreadGraph.kmers`.
 * After merging, it indexes directly into the parallel `Graph.kmers` and `Graph.idx` arrays.
 */
struct Node {
    std::uint64_t hash;
    std::uint64_t start;
    std::uint64_t stop;
    std::uint32_t n_tar;
    std::uint32_t n_neg;
    double penalty; // Used as thread_id before merging from different threads
};

/**
 * @brief Undirected weighted edge between two minimizers.
 */
struct Edge {
    std::uint64_t first;
    std::uint64_t second;
    std::uint64_t weight;
};

/**
 * @brief Container for the minimizer graph returned by `build()`.
 *
 * `kmers` stores minimizer occurrences in all assemblies, grouped and sorted by hash.
 * `idx` is parallel to `kmers` and stores each minimizer's original generation index, ordered by genomic position.
 * `nodes` and `edges` are sorted by hash.
 */
struct Graph {
    NoInitArray<Kmer> kmers;
    NoInitArray<std::uint64_t> idx;
    NoInitArray<Node> nodes;
    NoInitArray<Edge> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
};

/**
 * @brief Build a minimizer graph from assembly FASTA files.
 *
 * `assembly_paths`, `assembly_indices`, and `is_targets` are parallel lists.
 *
 * @param assembly_paths Paths to input assemblies in FASTA format (plain or gzipped).
 * @param kmerlen K-mer length for minimizer sketch.
 * @param windowsize Window size for minimizer sketch.
 * @param assembly_indices Assembly indices, expected to be `0..N-1`.
 * @param is_targets Whether each assembly is a target assembly.
 * @param n_cpu Number of worker threads to use.
 * @return Minimizer graph.
 * @throws `std::runtime_error` If input sizes are inconsistent or indices exceed supported ranges.
 */
Graph build(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets,
    std::size_t n_cpu
);

} // namespace seqwin

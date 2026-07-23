#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

/**
 * @brief Build a minimizer graph from assembly FASTA files.
 *
 * @param assembly_paths Paths to input assemblies in FASTA format (plain or gzipped).
 * @param kmerlen K-mer length for minimizer sketch.
 * @param windowsize Window size for minimizer sketch.
 * @param n_cpu Number of worker threads to use.
 * @param low_memory Recompute minimizers in a second pass to reduce peak memory.
 * @return Minimizer graph.
 * @throws `std::runtime_error` If input sizes are inconsistent or counts exceed supported ranges.
 */
Graph build(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    std::size_t n_cpu = 1,
    bool low_memory = false
);

} // namespace seqwin

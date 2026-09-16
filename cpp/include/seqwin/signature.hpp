#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace seqwin {

/** Location of a genomic sequence. */
struct SeqLocation {
    std::uint32_t assembly_idx;
    std::uint32_t record_idx;
    std::uint32_t start;
    std::uint32_t stop;
    std::size_t n_kmers;
    std::size_t n_repeats;
};

/** Pre-BLAST representation of an extracted signature. */
struct Signature {
    std::size_t subgraph_idx;
    SeqLocation location;
    std::string sequence;
    std::size_t length;
    std::size_t n_rep;
    double rep_ratio;
};

} // namespace seqwin

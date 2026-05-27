#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace seqwin {

struct FastaRecord {
    std::string id;
    std::string sequence;
};

/**
 * @brief Read records from a plain-text or gzip-compressed FASTA file.
 *
 * @param assembly_path Path to a FASTA or `.gz` FASTA file.
 * @return Vector of parsed FASTA records.
 * @throws `std::runtime_error` If the file cannot be opened or the FASTA format is invalid.
 */
std::vector<FastaRecord> read_fasta(const std::string& assembly_path);

/**
 * @brief Estimate the number of minimizers based on the size of assembly files.
 *
 * @param assembly_paths Paths to assembly FASTA files (plain or gzipped).
 * @param windowsize Minimizer window size used for the estimate.
 * @return Estimated total number of minimizers for all assembly files.
 */
std::size_t est_kmer_number(
    const std::vector<std::string>& assembly_paths,
    std::size_t windowsize
);

} // namespace seqwin

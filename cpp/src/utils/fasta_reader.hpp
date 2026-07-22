#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace seqwin::internal {

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

} // namespace seqwin::internal

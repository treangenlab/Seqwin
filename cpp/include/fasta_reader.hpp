#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace btllib {

struct FastaRecord
{
  std::string id;
  std::string sequence;
};

std::vector<FastaRecord>
read_fasta(const std::string& assembly_path);

std::size_t
est_kmer_number(const std::vector<std::string>& assembly_paths, std::size_t windowsize);

} // namespace btllib

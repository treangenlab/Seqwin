#include "fasta_reader.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <zlib.h>

namespace btllib {
namespace {

constexpr std::size_t plain_fasta_seq_len_per_byte = 1;
constexpr std::size_t gz_fasta_seq_len_per_byte = 4;

bool
ends_with(std::string_view text, std::string_view suffix)
{
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

std::string
extract_id(std::string_view header)
{
  const std::size_t id_end = header.find_first_of(" \t\n\r\f\v");
  if (id_end == std::string_view::npos) {
    return std::string(header);
  }
  return std::string(header.substr(0, id_end));
}

bool
is_ascii_whitespace_only(std::string_view text)
{
  return text.find_first_not_of(" \t\n\r\f\v") == std::string_view::npos;
}

template<typename NextLine>
std::vector<FastaRecord>
read_fasta_core(NextLine&& next_line)
{
  std::vector<FastaRecord> records;

  std::string line;
  FastaRecord current;
  bool have_current = false;

  while (next_line(line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (line.empty() || is_ascii_whitespace_only(line)) {
      continue;
    }
    if (line[0] == '>') {
      if (have_current) {
        records.push_back(std::move(current));
        current = FastaRecord{};
      }
      current.id = extract_id(line.substr(1));
      current.sequence.clear();
      have_current = true;
      continue;
    }

    if (!have_current) {
      throw std::runtime_error("Invalid FASTA: sequence encountered before header");
    }

    const auto first_ws = line.find_first_of(" \t\n\r\f\v");
    if (first_ws == std::string::npos) {
      current.sequence.append(line);
      continue;
    }

    const std::size_t needed = current.sequence.size() + line.size();
    if (needed > current.sequence.capacity()) {
      const std::size_t grown = std::max(current.sequence.capacity() * 2, needed);
      current.sequence.reserve(grown);
    }
    for (char c : line) {
      if (!std::isspace(static_cast<unsigned char>(c))) {
        current.sequence.push_back(c);
      }
    }
  }

  if (have_current) {
    records.push_back(std::move(current));
  }

  return records;
}

std::vector<FastaRecord>
read_plain_fasta(const std::string& assembly_path)
{
  std::ifstream in(assembly_path);
  if (!in) {
    throw std::runtime_error("Unable to open FASTA: " + assembly_path);
  }

  return read_fasta_core([&in](std::string& line) -> bool {
    return static_cast<bool>(std::getline(in, line));
  });
}

std::vector<FastaRecord>
read_gz_fasta(const std::string& assembly_path)
{
  gzFile raw_gz = gzopen(assembly_path.c_str(), "rb");
  if (raw_gz == nullptr) {
    throw std::runtime_error("Unable to open gzip FASTA: " + assembly_path);
  }

  struct GzCloser
  {
    void operator()(gzFile_s* f) const
    {
      if (f != nullptr) {
        gzclose(reinterpret_cast<gzFile>(f));
      }
    }
  };

  std::unique_ptr<gzFile_s, GzCloser> gz(reinterpret_cast<gzFile_s*>(raw_gz));

  constexpr int kBufSize = 1 << 16;
  std::string carry;
  std::size_t line_start = 0;
  std::size_t search_start = 0;
  bool at_eof = false;

  auto fill_buffer = [&]() {
    if (at_eof) {
      return;
    }

    char buf[kBufSize];
    int bytes = gzread(reinterpret_cast<gzFile>(gz.get()), buf, kBufSize);
    if (bytes < 0) {
      int errnum = 0;
      const char* err = gzerror(reinterpret_cast<gzFile>(gz.get()), &errnum);
      throw std::runtime_error(std::string("gzip read error: ") + (err ? err : "unknown"));
    }
    if (bytes == 0) {
      at_eof = true;
      return;
    }
    carry.append(buf, static_cast<std::size_t>(bytes));
  };

  auto records = read_fasta_core([&](std::string& line) -> bool {
    for (;;) {
      const std::size_t nl = carry.find('\n', search_start);
      if (nl != std::string::npos) {
        std::size_t end = nl;
        if (end > line_start && carry[end - 1] == '\r') {
          --end;
        }
        line.assign(carry.data() + line_start, end - line_start);
        line_start = nl + 1;
        search_start = line_start;

        if (line_start > kBufSize && line_start >= carry.size() / 2) {
          carry.erase(0, line_start);
          search_start -= line_start;
          line_start = 0;
        }
        return true;
      }

      search_start = carry.size();

      if (at_eof) {
        if (line_start < carry.size()) {
          std::size_t end = carry.size();
          if (end > line_start && carry[end - 1] == '\r') {
            --end;
          }
          line.assign(carry.data() + line_start, end - line_start);
          line_start = carry.size();
          search_start = line_start;
          return true;
        }
        return false;
      }

      fill_buffer();
      if (line_start > kBufSize && line_start >= carry.size() / 2) {
        carry.erase(0, line_start);
        search_start -= line_start;
        line_start = 0;
      }
    }
  });

  return records;
}

} // namespace

std::vector<FastaRecord>
read_fasta(const std::string& assembly_path)
{
  if (ends_with(assembly_path, ".gz")) {
    return read_gz_fasta(assembly_path);
  }
  return read_plain_fasta(assembly_path);
}

std::size_t
est_kmer_number(const std::vector<std::string>& assembly_paths, std::size_t windowsize)
{
  // Reserve-only heuristic, not correctness-critical.
  std::size_t est_total_seq_len = 0;
  for (const auto& assembly_path : assembly_paths) {
    const std::size_t seq_len_per_byte =
      ends_with(assembly_path, ".gz") ? gz_fasta_seq_len_per_byte : plain_fasta_seq_len_per_byte;
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(assembly_path, ec);
    if (ec) {
      continue;
    }
    est_total_seq_len += static_cast<std::size_t>(file_bytes * seq_len_per_byte);
  }
  return (2 * est_total_seq_len) / (windowsize + 1);
}

} // namespace btllib

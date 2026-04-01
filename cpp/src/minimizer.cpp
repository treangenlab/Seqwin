#include "minimizer.hpp"

#include "nthash.hpp"

#include <limits>
#include <string>
#include <vector>

namespace btllib {

namespace {

inline void
calc_minimizer(const std::vector<Minimizer>& hashed_kmers_buffer,
               const Minimizer*& min_current,
               const std::size_t idx,
               ssize_t& min_idx_left,
               ssize_t& min_idx_right,
               ssize_t& min_pos_prev,
               const std::size_t w,
               std::vector<Minimizer>& minimizers)
{
  min_idx_left = ssize_t(idx + 1 - w);
  min_idx_right = ssize_t(idx + 1);

  const auto& min_left =
    hashed_kmers_buffer[std::size_t(min_idx_left) % hashed_kmers_buffer.size()];
  const auto& min_right = hashed_kmers_buffer[(std::size_t(min_idx_right) - 1) %
                                               hashed_kmers_buffer.size()];

  if (min_current == nullptr || min_current->pos < min_left.pos) {
    min_current = &min_left;
    for (ssize_t i = min_idx_left; i < min_idx_right; i++) {
      const auto& min_i = hashed_kmers_buffer[std::size_t(i) % hashed_kmers_buffer.size()];
      if (min_i.min_hash <= min_current->min_hash) {
        min_current = &min_i;
      }
    }
  } else if (min_right.min_hash <= min_current->min_hash) {
    min_current = &min_right;
  }

  if (ssize_t(min_current->pos) > min_pos_prev &&
      min_current->min_hash != std::numeric_limits<uint64_t>::max()) {
    min_pos_prev = ssize_t(min_current->pos);
    minimizers.push_back(*min_current);
  }
}

} // namespace

std::vector<Minimizer>
minimize_sequence(const std::string& seq, std::size_t k, std::size_t w)
{
  if ((k > seq.size()) || (w > seq.size() - k + 1)) {
    return {};
  }

  std::vector<Minimizer> minimizers;
  minimizers.reserve(2 * (seq.size() - k + 1) / w);

  std::vector<Minimizer> hashed_kmers_buffer(w + 1);
  ssize_t min_idx_left = -1;
  ssize_t min_idx_right = -1;
  ssize_t min_pos_prev = -1;
  const Minimizer* min_current = nullptr;

  std::size_t idx = 0;
  for (btllib::NtHash nh(seq, 2, k); nh.roll(); ++idx) {
    auto& hk = hashed_kmers_buffer[idx % hashed_kmers_buffer.size()];
    hk = Minimizer{ nh.hashes()[0],
                    nh.hashes()[1],
                    nh.get_pos(),
                    nh.get_forward_hash() <= nh.get_reverse_hash() };

    if (idx + 1 >= w) {
      calc_minimizer(hashed_kmers_buffer,
                     min_current,
                     idx,
                     min_idx_left,
                     min_idx_right,
                     min_pos_prev,
                     w,
                     minimizers);
    }
  }

  return minimizers;
}

} // namespace btllib

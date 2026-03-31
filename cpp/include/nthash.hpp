#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "hashing_internals.hpp"
#include "nthash_kmer.hpp"
#include "nthash_seed.hpp"
#include "status.hpp"

namespace btllib {

/**
 * String representing the hash function's name. Only change if hash outputs
 * are different from the previous version. Useful for tracking differences in
 * saved hashes, e.g., in Bloom filters.
 */
static const char* const NTHASH_FN_NAME = "ntHash_v2";

} // namespace btllib
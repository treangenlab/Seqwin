#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "seqwin/graph.hpp"

namespace seqwin {

struct ThreadResult {
    std::vector<std::uint8_t> kmers;
    std::vector<std::uint64_t> edges;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::size_t start_assembly = 0;
};

BuildResult merge_thread_results(
    std::vector<ThreadResult>& results,
    std::size_t n_assemblies,
    std::size_t n_workers
);

} // namespace seqwin

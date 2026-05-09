#include "graph.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fasta_reader.hpp"
#include "minimizer.hpp"

namespace {

constexpr std::size_t serialized_record_size = 17;

using EdgeKey = std::pair<std::uint64_t, std::uint64_t>;

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const noexcept
    {
        return std::hash<std::uint64_t>{}(key.first) ^
               (std::hash<std::uint64_t>{}(key.second) << 1);
    }
};

struct EdgeState {
    std::uint64_t weight;
    std::size_t last_seen_assembly;
};

void append_record(
    std::vector<std::uint8_t>& out,
    std::uint64_t out_hash,
    std::uint32_t pos,
    std::uint16_t record_idx,
    std::uint16_t assembly_idx,
    std::uint8_t is_target
) {
    const auto old_size = out.size();
    out.resize(old_size + serialized_record_size);
    auto* record = out.data() + old_size;

    std::memcpy(record + 0, &out_hash, sizeof(out_hash));
    std::memcpy(record + 8, &pos, sizeof(pos));
    std::memcpy(record + 12, &record_idx, sizeof(record_idx));
    std::memcpy(record + 14, &assembly_idx, sizeof(assembly_idx));
    std::memcpy(record + 16, &is_target, sizeof(is_target));
}

} // namespace

IndexlrResult indexlr_impl(
    const std::vector<std::string>& assembly_paths,
    std::size_t kmerlen,
    std::size_t windowsize,
    const std::vector<std::size_t>& assembly_indices,
    const std::vector<bool>& is_targets
) {
    if (assembly_paths.size() != assembly_indices.size() ||
        assembly_paths.size() != is_targets.size()) {
        throw std::runtime_error(
            "assembly_paths, assembly_idx, and is_target must have the same length");
    }

    std::vector<std::uint8_t> kmers;
    kmers.reserve(btllib::est_kmer_number(assembly_paths, windowsize) * serialized_record_size);
    std::vector<std::vector<std::string>> all_idx_to_id;
    all_idx_to_id.reserve(assembly_paths.size());

    std::unordered_map<EdgeKey, EdgeState, EdgeKeyHash> edge_map;

    for (std::size_t assembly_i = 0; assembly_i < assembly_paths.size(); ++assembly_i) {
        const auto assembly_idx = assembly_indices[assembly_i];
        if (assembly_idx > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("assembly_idx must fit in uint16");
        }
        const auto assembly_idx16 = static_cast<std::uint16_t>(assembly_idx);
        const std::uint8_t is_target_u8 =
            is_targets[assembly_i] ? std::uint8_t{1} : std::uint8_t{0};

        auto records = btllib::read_fasta(assembly_paths[assembly_i]);
        auto& idx_to_id = all_idx_to_id.emplace_back();
        idx_to_id.reserve(records.size());

        for (std::size_t record_idx = 0; record_idx < records.size(); ++record_idx) {
            if (record_idx > std::numeric_limits<std::uint16_t>::max()) {
                throw std::runtime_error("record_idx must fit in uint16");
            }
            const auto record_idx16 = static_cast<std::uint16_t>(record_idx);

            auto& record = records[record_idx];
            idx_to_id.push_back(std::move(record.id));

            const auto mins = btllib::minimize_sequence(record.sequence, kmerlen, windowsize);

            for (const auto& m : mins) {
                if (m.pos > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("minimizer position exceeds uint32 range");
                }
                append_record(
                    kmers,
                    m.out_hash,
                    static_cast<std::uint32_t>(m.pos),
                    record_idx16,
                    assembly_idx16,
                    is_target_u8);
            }

            if (mins.size() < 2) {
                continue;
            }

            for (std::size_t i = 0; i + 1 < mins.size(); ++i) {
                auto u = mins[i].out_hash;
                auto v = mins[i + 1].out_hash;
                if (v < u) {
                    std::swap(u, v);
                }
                const EdgeKey key{u, v};
                auto [it, inserted] = edge_map.try_emplace(
                    key,
                    EdgeState{1, assembly_i}
                );
                if (!inserted && it->second.last_seen_assembly != assembly_i) {
                    ++(it->second.weight);
                    it->second.last_seen_assembly = assembly_i;
                }
            }
        }
    }

    std::vector<std::uint64_t> edges;
    edges.reserve(edge_map.size() * 3);
    for (const auto& [key, state] : edge_map) {
        edges.push_back(key.first);
        edges.push_back(key.second);
        edges.push_back(state.weight);
    }

    return {std::move(kmers), std::move(edges), std::move(all_idx_to_id)};
}

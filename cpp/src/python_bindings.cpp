#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fasta_reader.hpp"
#include "minimizer.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t serialized_record_size = 17;

struct IndexlrResult {
    std::vector<std::uint8_t> kmers;
    std::vector<std::vector<std::string>> ids_by_assembly;
    std::vector<std::uint64_t> record_offsets;
    std::vector<std::uint64_t> assembly_offsets;
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
    // Estimate total number of k-mers by file sizes
    kmers.reserve(btllib::est_kmer_number(assembly_paths, windowsize) *
                  serialized_record_size);
    std::vector<std::vector<std::string>> all_idx_to_id;
    all_idx_to_id.reserve(assembly_paths.size());
    std::vector<std::uint64_t> record_offsets;
    std::vector<std::uint64_t> assembly_offsets;
    std::uint64_t global_minimizer_idx = 0;

    for (std::size_t assembly_i = 0; assembly_i < assembly_paths.size(); ++assembly_i) {
        const auto assembly_idx = assembly_indices[assembly_i];
        if (assembly_idx > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("assembly_idx must fit in uint16");
        }
        const auto assembly_idx16 = static_cast<std::uint16_t>(assembly_idx);
        const std::uint8_t is_target_u8 =
            is_targets[assembly_i] ? std::uint8_t{1} : std::uint8_t{0};

        // Read fasta records
        auto records = btllib::read_fasta(assembly_paths[assembly_i]);
        auto& idx_to_id = all_idx_to_id.emplace_back();
        idx_to_id.reserve(records.size());
        bool assembly_has_minimizers = false;

        // Generate minimizers for each record
        for (std::size_t record_idx = 0; record_idx < records.size(); ++record_idx) {
            if (record_idx > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("record_idx must fit in uint16");
            }
            const auto record_idx16 = static_cast<std::uint16_t>(record_idx);

            auto& record = records[record_idx];
            idx_to_id.push_back(std::move(record.id));

            const auto mins =
                btllib::minimize_sequence(record.sequence, kmerlen, windowsize);
            if (!mins.empty()) {
                if (!assembly_has_minimizers) {
                    assembly_offsets.push_back(global_minimizer_idx);
                    assembly_has_minimizers = true;
                }
                record_offsets.push_back(global_minimizer_idx);
            }
            for (const auto& m : mins) {
                if (m.pos > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("minimizer position exceeds uint32 range");
                }

                append_record(
                    kmers,
                    m.out_hash,
                    static_cast<std::uint32_t>(m.pos),
                    record_idx16,
                    assembly_idx16,
                    is_target_u8
                );
                ++global_minimizer_idx;
            }
        }
    }

    return {
        std::move(kmers),
        std::move(all_idx_to_id),
        std::move(record_offsets),
        std::move(assembly_offsets)
    };
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Minimal btllib indexlr bindings";

    m.def("_indexlr_native",
        [](const std::vector<std::string>& assembly_paths,
           std::size_t kmerlen,
           std::size_t windowsize,
           const std::vector<std::size_t>& assembly_indices,
           const std::vector<bool>& is_targets
        ) {
            IndexlrResult result;
            {
                py::gil_scoped_release release;
                result = indexlr_impl(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    assembly_indices,
                    is_targets
                );
            }
            auto owner = std::make_shared<IndexlrResult>(std::move(result));
            auto capsule = py::capsule(
                new std::shared_ptr<IndexlrResult>(owner),
                [](void* ptr) {
                    delete static_cast<std::shared_ptr<IndexlrResult>*>(ptr);
                }
            );

            auto kmers = py::array_t<std::uint8_t>(
                {static_cast<py::ssize_t>(owner->kmers.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint8_t))},
                owner->kmers.data(),
                capsule
            );
            auto record_offsets = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->record_offsets.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->record_offsets.data(),
                capsule
            );
            auto assembly_offsets = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->assembly_offsets.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->assembly_offsets.data(),
                capsule
            );
            py::list all_idx_to_id;
            for (const auto& ids : owner->ids_by_assembly) {
                py::tuple idx_to_id(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    idx_to_id[i] = ids[i];
                }
                all_idx_to_id.append(idx_to_id);
            }

            return py::make_tuple(
                kmers,
                all_idx_to_id,
                record_offsets,
                assembly_offsets
            );
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("assembly_idx"),
        py::arg("is_target")
    );
}

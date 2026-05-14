#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Seqwin minimizer graph bindings";

    m.def("_build_native",
        [](const std::vector<std::string>& assembly_paths,
           std::size_t kmerlen,
           std::size_t windowsize,
           const std::vector<std::size_t>& assembly_indices,
           const std::vector<bool>& is_targets,
           std::size_t n_cpu
        ) {
            seqwin::BuildResult result;
            {
                py::gil_scoped_release release;
                result = seqwin::build_impl(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    assembly_indices,
                    is_targets,
                    n_cpu
                );
            }
            auto owner = std::make_shared<seqwin::BuildResult>(std::move(result));
            auto capsule = py::capsule(
                new std::shared_ptr<seqwin::BuildResult>(owner),
                [](void* ptr) {
                    delete static_cast<std::shared_ptr<seqwin::BuildResult>*>(ptr);
                }
            );

            auto kmers = py::array_t<std::uint8_t>(
                {static_cast<py::ssize_t>(owner->kmers.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint8_t))},
                owner->kmers.data(),
                capsule
            );
            auto idx = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->idx.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->idx.data(),
                capsule
            );
            auto nodes = py::array_t<std::uint8_t>(
                {static_cast<py::ssize_t>(owner->nodes.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint8_t))},
                owner->nodes.data(),
                capsule
            );
            const auto n_edges = static_cast<py::ssize_t>(owner->edges.size() / 3);
            auto edges = py::array_t<std::uint64_t>(
                {n_edges, static_cast<py::ssize_t>(3)},
                {static_cast<py::ssize_t>(3 * sizeof(std::uint64_t)),
                 static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->edges.data(),
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

            return py::make_tuple(kmers, idx, nodes, edges, all_idx_to_id);
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("assembly_idx"),
        py::arg("is_target"),
        py::arg("n_cpu") = 1
    );
}

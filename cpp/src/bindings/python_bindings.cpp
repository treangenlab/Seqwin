#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/graph.hpp"
#include "seqwin/helpers.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    PYBIND11_NUMPY_DTYPE(seqwin::Kmer, pos, record_idx);
    PYBIND11_NUMPY_DTYPE(seqwin::Node, hash, start, stop, n_tar, n_neg, penalty);
    PYBIND11_NUMPY_DTYPE(seqwin::Edge, first, second, weight);

    m.doc() = "Seqwin minimizer graph bindings";

    m.def("_build_native",
        [](const std::vector<std::string>& assembly_paths,
           std::size_t kmerlen,
           std::size_t windowsize,
           const std::vector<bool>& is_targets,
           std::size_t n_cpu
        ) {
            seqwin::Graph graph;
            {
                py::gil_scoped_release release;
                graph = seqwin::build(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    is_targets,
                    n_cpu
                );
            }
            auto owner = std::make_shared<seqwin::Graph>(std::move(graph));
            auto capsule = py::capsule(
                new std::shared_ptr<seqwin::Graph>(owner),
                [](void* ptr) {
                    delete static_cast<std::shared_ptr<seqwin::Graph>*>(ptr);
                }
            );

            auto kmers = py::array_t<seqwin::Kmer>(
                {static_cast<py::ssize_t>(owner->kmers.size())},
                {static_cast<py::ssize_t>(sizeof(seqwin::Kmer))},
                owner->kmers.data(),
                capsule
            );
            auto idx = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->idx.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->idx.data(),
                capsule
            );
            auto nodes = py::array_t<seqwin::Node>(
                {static_cast<py::ssize_t>(owner->nodes.size())},
                {static_cast<py::ssize_t>(sizeof(seqwin::Node))},
                owner->nodes.data(),
                capsule
            );
            auto edges = py::array_t<seqwin::Edge>(
                {static_cast<py::ssize_t>(owner->edges.size())},
                {static_cast<py::ssize_t>(sizeof(seqwin::Edge))},
                owner->edges.data(),
                capsule
            );
            auto record_offsets = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->record_offsets.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->record_offsets.data(),
                capsule
            );
            py::list ids_by_assembly;
            for (const auto& ids : owner->ids_by_assembly) {
                py::tuple ids_tuple(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    ids_tuple[i] = ids[i];
                }
                ids_by_assembly.append(ids_tuple);
            }

            return py::make_tuple(kmers, idx, nodes, edges, record_offsets, ids_by_assembly);
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("is_targets"),
        py::arg("n_cpu") = 1
    );

    m.def("_filter_kmers_native",
        [](py::array_t<seqwin::Kmer, py::array::c_style> kmers,
           py::array_t<std::uint64_t, py::array::c_style> idx,
           py::array_t<seqwin::Node, py::array::c_style> nodes,
           const std::vector<std::uint64_t>& used_hashes
        ) {
            auto kmers_buf = kmers.request();
            auto idx_buf = idx.request();
            auto nodes_buf = nodes.request();

            const auto* kmers_ptr = static_cast<const seqwin::Kmer*>(kmers_buf.ptr);
            const auto* idx_ptr = static_cast<const std::uint64_t*>(idx_buf.ptr);
            const auto* nodes_ptr = static_cast<const seqwin::Node*>(nodes_buf.ptr);
            const auto n_nodes = static_cast<std::size_t>(nodes_buf.shape[0]);

            seqwin::Graph graph;
            {
                py::gil_scoped_release release;
                graph = seqwin::filter_kmers(
                    kmers_ptr,
                    idx_ptr,
                    nodes_ptr,
                    n_nodes,
                    used_hashes
                );
            }
            auto owner = std::make_shared<seqwin::Graph>(std::move(graph));
            auto capsule = py::capsule(
                new std::shared_ptr<seqwin::Graph>(owner),
                [](void* ptr) {
                    delete static_cast<std::shared_ptr<seqwin::Graph>*>(ptr);
                }
            );

            auto kmers_new = py::array_t<seqwin::Kmer>(
                {static_cast<py::ssize_t>(owner->kmers.size())},
                {static_cast<py::ssize_t>(sizeof(seqwin::Kmer))},
                owner->kmers.data(),
                capsule
            );
            auto idx_new = py::array_t<std::uint64_t>(
                {static_cast<py::ssize_t>(owner->idx.size())},
                {static_cast<py::ssize_t>(sizeof(std::uint64_t))},
                owner->idx.data(),
                capsule
            );
            auto nodes_new = py::array_t<seqwin::Node>(
                {static_cast<py::ssize_t>(owner->nodes.size())},
                {static_cast<py::ssize_t>(sizeof(seqwin::Node))},
                owner->nodes.data(),
                capsule
            );

            return py::make_tuple(kmers_new, idx_new, nodes_new);
        },
        py::arg("kmers").noconvert(),
        py::arg("idx").noconvert(),
        py::arg("nodes").noconvert(),
        py::arg("used_hashes")
    );
}

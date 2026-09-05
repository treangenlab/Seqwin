#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/build.hpp"
#include "seqwin/filter.hpp"
#include "seqwin/graph.hpp"

namespace py = pybind11;

namespace {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style>;

template <typename Array>
auto array_to_numpy(Array&& values) {
    using Owner = std::decay_t<Array>;
    using T = std::remove_cv_t<
        std::remove_pointer_t<decltype(std::declval<Owner&>().data())>
    >;

    auto* owner = new Owner(std::forward<Array>(values));
    auto capsule = py::capsule(owner, [](void* ptr) {
        delete static_cast<Owner*>(ptr);
    });

    return py::array_t<T>(
        {static_cast<py::ssize_t>(owner->size())},
        {static_cast<py::ssize_t>(sizeof(T))},
        owner->data(),
        capsule
    );
}

template <typename Array>
std::size_t require_1d_size(const Array& array, const char* name) {
    if (array.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    return static_cast<std::size_t>(array.shape(0));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    PYBIND11_NUMPY_DTYPE(seqwin::Kmer, pos, record_idx);
    PYBIND11_NUMPY_DTYPE(seqwin::Node, hash, start, stop, n_tar, n_neg, penalty);
    PYBIND11_NUMPY_DTYPE(seqwin::Edge, first, second, weight);

    m.doc() = "Seqwin minimizer graph bindings";

    m.def("_build_native",
        [](const std::vector<std::string>& assembly_paths,
           std::size_t kmerlen,
           std::size_t windowsize,
           std::size_t n_cpu,
           bool low_memory
        ) {
            seqwin::Graph graph;
            {
                py::gil_scoped_release release;
                graph = seqwin::build(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    n_cpu,
                    low_memory
                );
            }

            return py::make_tuple(
                array_to_numpy(std::move(graph.kmers)),
                array_to_numpy(std::move(graph.nodes)),
                array_to_numpy(std::move(graph.edges)),
                array_to_numpy(std::move(graph.record_offsets)),
                std::move(graph.record_ids)
            );
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("n_cpu") = 1,
        py::arg("low_memory") = false
    );

    m.def("_filter_native",
        [](NumpyArray<seqwin::Kmer> kmers,
           NumpyArray<seqwin::Node> nodes,
           NumpyArray<seqwin::Edge> edges,
           NumpyArray<std::uint32_t> record_offsets,
           NumpyArray<bool> is_targets,
           std::optional<NumpyArray<double>> jaccard,
           std::optional<double> penalty_th,
           double stringency,
           double penalty_th_cap,
           double edge_w_th_mul,
           std::size_t windowsize,
           std::size_t min_len,
           std::optional<std::size_t> max_len,
           std::size_t min_nodes_floor,
           std::optional<std::size_t> max_nodes_cap,
           std::size_t n_cpu
        ) {
            const auto* kmers_ptr = kmers.data();
            auto* nodes_ptr = nodes.mutable_data();
            const auto* edges_ptr = edges.data();
            const auto* record_offsets_ptr = record_offsets.data();
            const auto* is_targets_ptr = is_targets.data();
            if (!nodes.writeable()) {
                throw std::invalid_argument("nodes must be writable");
            }

            require_1d_size(kmers, "kmers");
            const auto n_nodes = require_1d_size(nodes, "nodes");
            const auto n_edges = require_1d_size(edges, "edges");
            const auto n_record_offsets = require_1d_size(record_offsets, "record_offsets");
            const auto n_assemblies = require_1d_size(is_targets, "is_targets");

            const double* jaccard_ptr = nullptr;
            std::size_t jaccard_rows = 0;
            std::size_t jaccard_cols = 0;
            if (jaccard) {
                if (jaccard->ndim() != 2) {
                    throw std::invalid_argument("jaccard must be a two-dimensional array");
                }
                jaccard_ptr = jaccard->data();
                jaccard_rows = static_cast<std::size_t>(jaccard->shape(0));
                jaccard_cols = static_cast<std::size_t>(jaccard->shape(1));
            }
            seqwin::FilterConfig config{
                penalty_th,
                stringency,
                penalty_th_cap,
                edge_w_th_mul,
                windowsize,
                min_len,
                max_len,
                min_nodes_floor,
                max_nodes_cap,
                n_cpu
            };

            seqwin::FilterResult result;
            {
                py::gil_scoped_release release;
                result = seqwin::filter(
                    kmers_ptr,
                    nodes_ptr,
                    n_nodes,
                    edges_ptr,
                    n_edges,
                    record_offsets_ptr,
                    n_record_offsets,
                    is_targets_ptr,
                    n_assemblies,
                    jaccard_ptr,
                    jaccard_rows,
                    jaccard_cols,
                    config
                );
            }

            return py::make_tuple(
                array_to_numpy(std::move(result.kmers)),
                array_to_numpy(std::move(result.nodes)),
                array_to_numpy(std::move(result.edges)),
                std::move(result.subgraphs),
                result.total_tar,
                result.total_neg,
                result.penalty_th,
                result.edge_weight_th,
                result.min_nodes,
                result.max_nodes
            );
        },
        py::arg("kmers").noconvert(),
        py::arg("nodes").noconvert(),
        py::arg("edges").noconvert(),
        py::arg("record_offsets").noconvert(),
        py::arg("is_targets").noconvert(),
        py::arg("jaccard").noconvert(),
        py::arg("penalty_th"),
        py::arg("stringency"),
        py::arg("penalty_th_cap"),
        py::arg("edge_w_th_mul"),
        py::arg("windowsize"),
        py::arg("min_len"),
        py::arg("max_len"),
        py::arg("min_nodes_floor"),
        py::arg("max_nodes_cap"),
        py::arg("n_cpu")
    );

}

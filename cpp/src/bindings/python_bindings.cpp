#include <cstdint>
#include <memory>
#include <limits>
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

template <typename Array>
auto array_to_numpy(Array&& values) {
    using Owner = typename std::decay<Array>::type;
    using T = typename std::remove_cv<
        typename std::remove_pointer<decltype(std::declval<Owner&>().data())>::type
    >::type;

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

            auto kmers = array_to_numpy(std::move(graph.kmers));
            auto nodes = array_to_numpy(std::move(graph.nodes));
            auto edges = array_to_numpy(std::move(graph.edges));
            auto record_offsets = array_to_numpy(std::move(graph.record_offsets));

            return py::make_tuple(kmers, nodes, edges, record_offsets, std::move(graph.record_ids));
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("n_cpu") = 1,
        py::arg("low_memory") = false
    );

    m.def("_filter_native",
        [](py::array_t<seqwin::Kmer, py::array::c_style> kmers,
           py::array_t<seqwin::Node, py::array::c_style> nodes,
           py::array_t<seqwin::Edge, py::array::c_style> edges,
           py::array_t<std::uint32_t, py::array::c_style> record_offsets,
           py::array_t<bool, py::array::c_style> is_targets,
           py::object jaccard_object,
           py::object penalty_th,
           double stringency,
           double penalty_th_cap,
           double edge_w_th_mul,
           std::size_t windowsize,
           std::size_t min_len,
           py::object max_len,
           std::size_t min_nodes_floor,
           py::object max_nodes_cap,
           std::uint64_t seed,
           std::size_t n_cpu
        ) {
            auto kmers_buf = kmers.request();
            auto nodes_buf = nodes.request();
            auto edges_buf = edges.request();
            auto offsets_buf = record_offsets.request();
            auto targets_buf = is_targets.request();
            if (nodes_buf.readonly) throw std::invalid_argument("nodes must be writable");

            py::array_t<double, py::array::c_style> jaccard;
            const double* jaccard_ptr = nullptr;
            std::size_t jaccard_rows = 0;
            std::size_t jaccard_cols = 0;
            if (!jaccard_object.is_none()) {
                jaccard = jaccard_object.cast<py::array_t<double, py::array::c_style>>();
                auto buf = jaccard.request();
                if (buf.ndim != 2) throw std::invalid_argument("jaccard must be a two-dimensional array");
                jaccard_ptr = static_cast<const double*>(buf.ptr);
                jaccard_rows = static_cast<std::size_t>(buf.shape[0]);
                jaccard_cols = static_cast<std::size_t>(buf.shape[1]);
            }
            seqwin::FilterConfig config{
                penalty_th.is_none() ? std::nullopt : std::optional<double>(penalty_th.cast<double>()),
                stringency, penalty_th_cap, edge_w_th_mul, windowsize, min_len,
                max_len.is_none() ? std::nullopt : std::optional<std::size_t>(max_len.cast<std::size_t>()),
                min_nodes_floor,
                max_nodes_cap.is_none() ? std::nullopt : std::optional<std::size_t>(max_nodes_cap.cast<std::size_t>()),
                seed, n_cpu
            };
            seqwin::FilterResult result;
            {
                py::gil_scoped_release release;
                result = seqwin::filter(
                    static_cast<const seqwin::Kmer*>(kmers_buf.ptr),
                    static_cast<seqwin::Node*>(nodes_buf.ptr), static_cast<std::size_t>(nodes_buf.shape[0]),
                    static_cast<const seqwin::Edge*>(edges_buf.ptr), static_cast<std::size_t>(edges_buf.shape[0]),
                    static_cast<const std::uint32_t*>(offsets_buf.ptr), static_cast<std::size_t>(offsets_buf.shape[0]),
                    static_cast<const bool*>(targets_buf.ptr), static_cast<std::size_t>(targets_buf.shape[0]),
                    jaccard_ptr, jaccard_rows, jaccard_cols, config
                );
            }
            auto result_kmers = array_to_numpy(std::move(result.kmers));
            auto result_nodes = array_to_numpy(std::move(result.nodes));
            auto result_edges = array_to_numpy(std::move(result.edges));
            py::object result_max_nodes = result.max_nodes
                ? py::cast(*result.max_nodes) : py::none();
            return py::make_tuple(result_kmers, result_nodes, result_edges,
                std::move(result.subgraphs), result.penalty_th, result.edge_weight_th,
                result.min_nodes, result_max_nodes);
        },
        py::arg("kmers").noconvert(), py::arg("nodes").noconvert(),
        py::arg("edges").noconvert(), py::arg("record_offsets").noconvert(),
        py::arg("is_targets").noconvert(), py::arg("jaccard").noconvert(),
        py::arg("penalty_th"), py::arg("stringency"), py::arg("penalty_th_cap"),
        py::arg("edge_w_th_mul"), py::arg("windowsize"), py::arg("min_len"),
        py::arg("max_len"), py::arg("min_nodes_floor"), py::arg("max_nodes_cap"),
        py::arg("seed"), py::arg("n_cpu")
    );

}
